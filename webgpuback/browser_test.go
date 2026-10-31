//go:build js

// Browser end-to-end runs: the go-virtio/gpu guest driver, unchanged,
// drives gpudev over pcisim inside a browser worker, and gpudev presents
// through this backend on the browser's WebGPU. On the macOS CI lane the
// adapter must be a GPU (WEBGPU_REQUIRE_GPU=1); on Linux it is
// SwiftShader, and the log's adapter line says so.
package webgpuback

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu"
	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-virtio/gpudev"
	"github.com/go-virtio/gpudev/pcisim"
)

func ctx(t *testing.T) context.Context {
	c, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	return c
}

func device(t *testing.T) *webgpu.Device {
	t.Helper()
	g, err := webgpu.Navigator()
	if err != nil {
		t.Fatal(err)
	}
	a, err := g.RequestAdapter(ctx(t), &webgpu.RequestAdapterOptions{AllowFallback: os.Getenv("WEBGPU_ALLOW_FALLBACK") == "1"})
	if err != nil {
		t.Fatal(err)
	}
	info := a.Info()
	t.Logf("adapter: vendor %q, architecture %q, description %q, fallback %v", info.Vendor, info.Architecture, info.Description, info.Fallback)
	if os.Getenv("WEBGPU_REQUIRE_GPU") == "1" && info.Fallback {
		t.Fatal("a fallback adapter where a GPU is required")
	}
	d, err := a.RequestDevice(ctx(t), &webgpu.DeviceDescriptor{Label: t.Name()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Destroy)
	if err := webgpu.InstallReplay(ctx(t), ""); err != nil {
		t.Fatal(err)
	}
	return d
}

// readTexture copies a w×h bgra8 texture (w*4 a multiple of 256) out.
func readTexture(t *testing.T, d *webgpu.Device, tex *webgpu.Texture, w, h uint32) []byte {
	t.Helper()
	buf, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: uint64(w) * uint64(h) * 4, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Destroy()
	return readBack(t, d, buf, func(enc *webgpu.CommandEncoder) error {
		return enc.CopyTextureToBuffer(webgpu.TexelCopyTextureInfo{Texture: tex},
			webgpu.TexelCopyBufferInfo{Buffer: buf, Layout: webgpu.TexelCopyBufferLayout{BytesPerRow: w * 4}},
			webgpu.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1})
	})
}

func readBack(t *testing.T, d *webgpu.Device, buf *webgpu.Buffer, copyIn func(*webgpu.CommandEncoder) error) []byte {
	t.Helper()
	enc, err := d.CreateCommandEncoder("readback")
	if err != nil {
		t.Fatal(err)
	}
	if err := copyIn(enc); err != nil {
		t.Fatal(err)
	}
	cb, err := enc.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Queue().Submit(cb); err != nil {
		t.Fatal(err)
	}
	if err := buf.MapAsync(ctx(t), webgpu.MapModeRead, 0, 0); err != nil {
		t.Fatal(err)
	}
	out := make([]byte, buf.Size())
	if err := buf.ReadMapped(0, out); err != nil {
		t.Fatal(err)
	}
	_ = buf.Unmap()
	return out
}

// TestBrowser2D: the guest's framebuffer reaches the canvas texture, pixel
// for pixel: guest RAM -> writeTexture -> resource texture ->
// copyTextureToTexture -> canvas texture.
func TestBrowser2D(t *testing.T) {
	d := device(t)
	const w, h = 64, 48
	s, err := d.NewSurface(webgpu.SurfaceDescriptor{Width: w, Height: h, Format: webgpu.TextureFormatBGRA8Unorm,
		Usage: webgpu.TextureUsageCopyDst | webgpu.TextureUsageCopySrc | webgpu.TextureUsageRenderAttachment})
	if err != nil {
		t.Fatal(err)
	}
	var shown [][]byte
	be := New(Config{Device: d, Surfaces: map[uint32]*webgpu.Surface{0: s},
		BeforePresent: func(_ uint32, canvas *webgpu.Texture) error {
			shown = append(shown, readTexture(t, d, canvas, w, h))
			return nil
		}})
	ram := &gpudev.RAM{Base: 1 << 30, Bytes: make([]byte, 8<<20)}
	tr := pcisim.New(gpudev.New(gpudev.Config{Scanouts: []gpudev.Rect{{Width: w, Height: h}}}, ram, be), ram)
	g, err := gpu.OpenVirtioGPU(tr)
	if err != nil {
		t.Fatal(err)
	}
	fb, err := g.SetupFramebuffer(0, w, h)
	if err != nil {
		t.Fatal(err)
	}
	for frame := 0; frame < 2; frame++ {
		for i := range fb.Pix {
			fb.Pix[i] = byte(i*5 + frame*3)
		}
		if err := fb.Flush(); err != nil {
			t.Fatalf("frame %d: %v (transport %v)", frame, err, tr.Err())
		}
		if len(shown) != frame+1 || !bytes.Equal(shown[frame], fb.Pix) {
			t.Fatalf("frame %d: the canvas does not hold the guest's pixels", frame)
		}
	}
}

func desc(refs []uint32, body string) []byte { return wire.AppendDescriptor(nil, refs, []byte(body)) }

// TestBrowserWebGPUBatch: a guest program's compute work crosses virtio
// as webgpu-batch frames and runs on the browser's GPU: the guest creates
// a buffer, a shader that doubles it, the layouts and the pipeline by
// descriptor, and submits one stream (write, dispatch, copy); the host
// reads the result back from the GPU.
func TestBrowserWebGPUBatch(t *testing.T) {
	d := device(t)
	be := New(Config{Device: d})
	caps := wgpubatch.Caps{Version: 1, StreamVersion: wire.StreamVersion, MaxFrame: 1 << 16, MaxObjects: 64, MaxRings: 1, MaxMemory: 1 << 20}
	ram := &gpudev.RAM{Base: 1 << 30, Bytes: make([]byte, 8<<20)}
	tr := pcisim.New(gpudev.New(gpudev.Config{Scanouts: []gpudev.Rect{{Width: 16, Height: 16}}, WebGPUBatch: &caps}, ram, be), ram)
	g, err := gpu.OpenVirtioGPUWebGPUBatch(tr)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.WebGPUBatchCaps(); err != nil {
		t.Fatal(err)
	}
	if err := g.CreateWebGPUBatchContext(1, "compute"); err != nil {
		t.Fatal(err)
	}
	const n = 16
	storage := uint32(webgpu.BufferUsageStorage | webgpu.BufferUsageCopySrc | webgpu.BufferUsageCopyDst)
	read := uint32(webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst)
	e := wgpubatch.NewEncoder(caps.MaxFrame)
	// Guest ids: 1 data, 2 readback, 3 shader, 4 bgl, 5 layout, 6 group, 7 pipeline.
	e.Create(1, uint32(wire.KindBuffer), desc(nil, fmt.Sprintf(`{"Size":%d,"Usage":%d}`, 4*n, storage)))
	e.Create(2, uint32(wire.KindBuffer), desc(nil, fmt.Sprintf(`{"Size":%d,"Usage":%d}`, 4*n, read)))
	e.Create(3, uint32(wire.KindShaderModule), desc(nil, `{"Code":"@group(0) @binding(0) var<storage, read_write> v: array<u32>;\n@compute @workgroup_size(16) fn main(@builtin(global_invocation_id) i: vec3u) { v[i.x] = v[i.x] * 2u; }"}`))
	e.Create(4, uint32(wire.KindBindGroupLayout), desc(nil, `{"Entries":[{"Binding":0,"Visibility":4,"Buffer":{"Type":"storage"}}]}`))
	e.Create(5, uint32(wire.KindPipelineLayout), desc([]uint32{4}, `{"BindGroupLayouts":[null]}`))
	e.Create(6, uint32(wire.KindBindGroup), desc([]uint32{4, 1}, `{"Entries":[{"Binding":0}]}`))
	e.Create(7, uint32(wire.KindComputePipeline), desc([]uint32{5, 3}, `{"Compute":{"EntryPoint":"main"}}`))
	data := make([]byte, 4*n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(data[4*i:], uint32(i+1))
	}
	e.Stream([]uint32{
		uint32(wire.WriteBuffer), 1, 0, 0, 0, 4 * n,
		uint32(wire.BeginComputePass),
		uint32(wire.SetPipeline), 7,
		uint32(wire.SetBindGroup), 0, 6, 0,
		uint32(wire.Dispatch), 1, 1, 1,
		uint32(wire.EndPass),
		uint32(wire.CopyBufferToBuffer), 1, 0, 0, 2, 0, 0, 4 * n, 0,
		uint32(wire.Submit),
	}, data)
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	if err := g.SubmitWebGPUBatch(1, f, gpu.Fence{On: true, ID: 1}); err != nil {
		t.Fatalf("submit: %v (transport %v)", err, tr.Err())
	}
	// The host reads the readback buffer: the only map-readable buffer of
	// the context.
	var rb *webgpu.Buffer
	for _, c := range be.contexts() {
		for _, o := range c.objs {
			if b, ok := o.(*webgpu.Buffer); ok && b.Usage()&webgpu.BufferUsageMapRead != 0 {
				rb = b
			}
		}
	}
	if rb == nil {
		t.Fatal("the readback buffer was not created")
	}
	if err := rb.MapAsync(ctx(t), webgpu.MapModeRead, 0, 0); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 4*n)
	if err := rb.ReadMapped(0, got); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if v := binary.LittleEndian.Uint32(got[4*i:]); v != uint32(2*(i+1)) {
			t.Fatalf("v[%d] = %d, want %d", i, v, 2*(i+1))
		}
	}
}
