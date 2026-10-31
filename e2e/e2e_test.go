// Package e2e drives gpudev with the go-virtio/gpu guest driver, in
// process: the driver brings the device up over pcisim's emulated
// virtio-pci transport, and everything it sends crosses real virtqueues
// in guest RAM. These runs are the validation of the device model
// against an independent driver; they do not involve a browser or GPU.
package e2e

import (
	"bytes"
	"testing"

	"github.com/go-virtio/gpu"
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
