// The remote device is the go-gfx/webgpu API outside the browser.

//go:build !js

package e2e

import (
	"testing"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu"
	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-virtio/gpudev"
)

// virtioSink carries a go-gfx/webgpu remote device over the go-virtio/gpu
// guest driver: each call is one wgpubatch frame in a fenced SUBMIT_3D.
type virtioSink struct {
	t     *testing.T
	g     *gpu.VirtioGPU
	ctx   uint32
	e     *wgpubatch.Encoder
	fence uint64
}

func (s *virtioSink) send(build func(*wgpubatch.Encoder)) error {
	s.e.Reset()
	build(s.e)
	f, err := s.e.Frame()
	if err != nil {
		return err
	}
	s.fence++
	return s.g.SubmitWebGPUBatch(s.ctx, f, gpu.Fence{On: true, ID: s.fence})
}

func (s *virtioSink) Create(id uint32, kind wire.Kind, desc []byte) error {
	return s.send(func(e *wgpubatch.Encoder) { e.Create(id, uint32(kind), desc) })
}

func (s *virtioSink) Destroy(id uint32) error {
	return s.send(func(e *wgpubatch.Encoder) { e.Destroy(id) })
}

func (s *virtioSink) Submit(words []uint32, data []byte) error {
	return s.send(func(e *wgpubatch.Encoder) { e.Stream(words, data) })
}

// TestGuestRemoteDevice: a guest program written against the go-gfx/webgpu
// API (a remote device) runs through the go-virtio/gpu guest driver and
// gpudev. The backend holds every object the program created, under its
// own ids, and replays its streams, which name only those objects.
func TestGuestRemoteDevice(t *testing.T) {
	caps := wgpubatch.Caps{Version: 1, StreamVersion: wire.StreamVersion, MaxFrame: 1 << 16, MaxObjects: 1 << 20, MaxRings: 1, MaxMemory: 1 << 24}
	be, tr := newDevice(t, gpudev.Config{Scanouts: []gpudev.Rect{{Width: 16, Height: 16}}, WebGPUBatch: &caps})
	g, err := gpu.OpenVirtioGPUWebGPUBatch(tr)
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.WebGPUBatchCaps()
	if err != nil || got.StreamVersion != wire.StreamVersion {
		t.Fatalf("caps %+v %v", got, err)
	}
	if err := g.CreateWebGPUBatchContext(1, "remote"); err != nil {
		t.Fatal(err)
	}
	d := webgpu.NewRemoteDevice(&virtioSink{t: t, g: g, ctx: 1, e: wgpubatch.NewEncoder(got.MaxFrame)}, "guest", nil)

	buf, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 64, Usage: webgpu.BufferUsageStorage | webgpu.BufferUsageCopyDst | webgpu.BufferUsageCopySrc})
	if err != nil {
		t.Fatal(err)
	}
	out, err := d.CreateBuffer(webgpu.BufferDescriptor{Size: 64, Usage: webgpu.BufferUsageMapRead | webgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatal(err)
	}
	mod, err := d.CreateShaderModule(webgpu.ShaderModuleDescriptor{Code: "@compute @workgroup_size(16) fn main() {}"})
	if err != nil {
		t.Fatal(err)
	}
	bgl, err := d.CreateBindGroupLayout(webgpu.BindGroupLayoutDescriptor{Entries: []webgpu.BindGroupLayoutEntry{
		{Binding: 0, Visibility: webgpu.ShaderStageCompute, Buffer: &webgpu.BufferBindingLayout{Type: webgpu.BufferBindingTypeStorage}}}})
	if err != nil {
		t.Fatal(err)
	}
	pl, err := d.CreatePipelineLayout(webgpu.PipelineLayoutDescriptor{BindGroupLayouts: []*webgpu.BindGroupLayout{bgl}})
	if err != nil {
		t.Fatal(err)
	}
	bg, err := d.CreateBindGroup(webgpu.BindGroupDescriptor{Layout: bgl, Entries: []webgpu.BindGroupEntry{{Binding: 0, Buffer: buf}}})
	if err != nil {
		t.Fatal(err)
	}
	cp, err := d.CreateComputePipeline(webgpu.ComputePipelineDescriptor{Layout: pl, Compute: webgpu.ProgrammableStage{Module: mod, EntryPoint: "main"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Queue().WriteBuffer(buf, 0, make([]byte, 64)); err != nil {
		t.Fatal(err)
	}
	r := d.NewRecorder()
	p := r.BeginComputePass()
	p.SetPipeline(cp)
	p.SetBindGroup(0, bg)
	p.DispatchWorkgroups(1, 1, 1)
	if err := p.End(); err != nil {
		t.Fatal(err)
	}
	r.CopyBufferToBuffer(buf, 0, out, 0, 64)
	if _, err := r.Submit(); err != nil {
		t.Fatalf("submit: %v (transport %v)", err, tr.Err())
	}
	cp.Release()

	be.Lock()
	defer be.Unlock()
	c := be.Contexts[1]
	if c == nil || len(c.Objects) != 6 || len(c.Replays) != 2 {
		t.Fatalf("backend: %d objects, %d replays", len(c.Objects), len(c.Replays))
	}
	// The bind group's descriptor names the backend's ids.
	for _, o := range c.Objects {
		if wire.Kind(o.Kind) != wire.KindBindGroup {
			continue
		}
		refs, _, _ := wire.SplitDescriptor(o.Desc)
		for _, id := range refs {
			if _, ok := c.Objects[id]; !ok {
				t.Errorf("bind group references %d, not a backend object", id)
			}
		}
	}
}
