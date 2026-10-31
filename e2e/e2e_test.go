// Package e2e drives gpudev with the go-virtio/gpu guest driver, in
// process: the driver brings the device up over pcisim's emulated
// virtio-pci transport, and everything it sends crosses real virtqueues
// in guest RAM. These runs are the validation of the device model
// against an independent driver; they do not involve a browser or GPU.
package e2e

import (
	"bytes"
	"testing"

	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu"
	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-virtio/gpudev"
	"github.com/go-virtio/gpudev/memback"
	"github.com/go-virtio/gpudev/pcisim"
)

func newDevice(t *testing.T, cfg gpudev.Config) (*memback.Backend, *pcisim.Transport) {
	t.Helper()
	ram := &gpudev.RAM{Base: 1 << 30, Bytes: make([]byte, 32<<20)}
	be := memback.New()
	return be, pcisim.New(gpudev.New(cfg, ram, be), ram)
}

// TestGuest2D: the guest's own framebuffer path (DisplayInfo,
// SetupFramebuffer, Flush) shows exactly the pixels it drew.
func TestGuest2D(t *testing.T) {
	be, tr := newDevice(t, gpudev.Config{Scanouts: []gpudev.Rect{{Width: 320, Height: 200}}})
	g, err := gpu.OpenVirtioGPU(tr)
	if err != nil {
		t.Fatal(err)
	}
	ds, err := g.DisplayInfo()
	if err != nil {
		t.Fatal(err)
	}
	if !ds[0].Enabled || ds[0].Width != 320 || ds[0].Height != 200 || ds[1].Enabled {
		t.Fatalf("displays %+v", ds[:2])
	}
	fb, err := g.SetupFramebuffer(0, 320, 200)
	if err != nil {
		t.Fatal(err)
	}
	for frame := 0; frame < 3; frame++ {
		for i := range fb.Pix {
			fb.Pix[i] = byte(i*7 + frame)
		}
		if err := fb.Flush(); err != nil {
			t.Fatal(err)
		}
		be.Lock()
		got := be.Scanouts[0]
		ok := got != nil && got.Width == 320 && got.Height == 200 && bytes.Equal(got.Pix, fb.Pix)
		be.Unlock()
		if !ok {
			t.Fatalf("frame %d: the scanout does not show what the guest drew", frame)
		}
	}
	if be.Flushes != 3 || tr.Err() != nil {
		t.Fatalf("flushes %d, transport error %v", be.Flushes, tr.Err())
	}
}

// TestGuestWebGPUBatch: the guest's webgpu-batch calls (capset query,
// context creation, fenced SUBMIT_3D of wgpubatch frames) against the
// device; the backend receives the objects and a stream whose ids were
// translated from the guest's namespace to its own.
func TestGuestWebGPUBatch(t *testing.T) {
	caps := wgpubatch.Caps{Version: 1, StreamVersion: wire.StreamVersion, MaxFrame: 1 << 16, MaxObjects: 256, MaxRings: 4, MaxMemory: 64 << 20}
	be, tr := newDevice(t, gpudev.Config{Scanouts: []gpudev.Rect{{Width: 64, Height: 64}}, WebGPUBatch: &caps})
	g, err := gpu.OpenVirtioGPUWebGPUBatch(tr)
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.WebGPUBatchCaps()
	if err != nil || got != caps {
		t.Fatalf("caps %+v %v", got, err)
	}
	if err := g.CreateWebGPUBatchContext(3, "e2e"); err != nil {
		t.Fatal(err)
	}
	e := wgpubatch.NewEncoder(got.MaxFrame)
	e.Create(10, uint32(wire.KindBuffer), wire.AppendDescriptor(nil, nil, []byte(`{"Size":16,"Usage":136}`)))
	e.Create(11, uint32(wire.KindBuffer), wire.AppendDescriptor(nil, nil, []byte(`{"Size":16,"Usage":136}`)))
	e.Stream([]uint32{
		uint32(wire.WriteBuffer), 10, 0, 0, 0, 4,
		uint32(wire.CopyBufferToBuffer), 10, 0, 0, 11, 0, 0, 4, 0,
		uint32(wire.Submit),
	}, []byte{9, 8, 7, 6})
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	for seq := uint64(1); seq <= 3; seq++ {
		fence := gpu.Fence{On: true, ID: seq, Ringed: true, Ring: 1}
		if seq > 1 {
			e.Reset()
			e.Stream([]uint32{uint32(wire.CopyBufferToBuffer), 11, 0, 0, 10, 0, 0, 4, 0, uint32(wire.Submit)}, nil)
			f, _ = e.Frame()
		}
		if err := g.SubmitWebGPUBatch(3, f, fence); err != nil {
			t.Fatalf("submit %d: %v", seq, err)
		}
	}
	be.Lock()
	defer be.Unlock()
	c := be.Contexts[3]
	if c == nil || len(c.Objects) != 2 || len(c.Replays) != 3 {
		t.Fatalf("backend context %+v", c)
	}
	if a := c.Replays[0][1].Args; a[0] != memback.FirstID || a[3] != memback.FirstID+1 {
		t.Errorf("copy names %d -> %d, want the backend's ids", a[0], a[3])
	}
	if tr.Err() != nil {
		t.Fatal(tr.Err())
	}
}
