package gpudev_test

import (
	"testing"

	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-virtio/gpudev"
)

// FuzzHandleControl runs a sequence of guest requests (the input split at
// 0xFF 0xFE markers) against a device with a small host-memory budget.
// Whatever the requests, nothing panics, every response is a header or
// nothing, and the resources the backend holds never exceed the budget.
func FuzzHandleControl(f *testing.F) {
	full := gpudev.Rect{Width: 16, Height: 16}
	seq := [][]byte{
		create(1, gpudev.FormatB8G8R8A8, 16, 16),
		attach(1, [2]uint64{base, 512}, [2]uint64{base + 4096, 512}),
		transfer(1, gpudev.Rect{X: 1, Y: 2, Width: 3, Height: 9}, 2*64+4),
		setScanout(0, 1, full),
		flush(1, full),
		cmd(gpudev.CmdGetCapset, 30, 1),
		cmd(gpudev.CmdResourceUnref, 1, 0),
	}
	var joined []byte
	for _, s := range seq {
		joined = append(append(joined, s...), 0xFF, 0xFE)
	}
	f.Add(joined)
	f.Add(cmd(gpudev.CmdGetDisplayInfo))
	e := wgpubatch.NewEncoder(4096)
	e.Create(1, 1, desc())
	e.Create(2, 8, desc(1))
	e.Stream([]uint32{30, 1, 0, 0, 0, 4, 40}, []byte{1, 2, 3, 4})
	e.BindResource(wgpubatch.BindResource{ID: 1, ResourceID: 1, Size: 8})
	e.Destroy(2)
	fr, _ := e.Frame()
	var batch []byte
	for _, s := range [][]byte{create(1, gpudev.FormatB8G8R8A8, 16, 16), attach(1, [2]uint64{base, 1024}),
		ctxCreate(1, 0, 30), ctxCmd(gpudev.CmdCtxAttachResource, 1, 1, 0), submit(1, fr)} {
		batch = append(append(batch, s...), 0xFF, 0xFE)
	}
	f.Add(batch)
	caps := wgpubatch.Caps{Version: 1, StreamVersion: 1, MaxFrame: 4096, MaxObjects: 16, MaxRings: 1, MaxMemory: 1 << 16}
	const budget = 64 << 10
	f.Fuzz(func(t *testing.T, in []byte) {
		r := newRig(gpudev.Config{Scanouts: []gpudev.Rect{{Width: 32, Height: 32}}, MaxHostMem: budget, MaxBackingEntries: 64, WebGPUBatch: &caps})
		resp := make([]byte, 512)
		for len(in) > 0 {
			req := in
			for i := 0; i+1 < len(in); i++ {
				if in[i] == 0xFF && in[i+1] == 0xFE {
					req, in = in[:i], in[i+2:]
					break
				}
			}
			if len(req) == len(in) {
				in = nil
			}
			if n := r.d.HandleControl(req, resp); n != 0 && n < gpudev.HdrSize {
				t.Fatalf("response of %d bytes", n)
			}
		}
		var held uint64
		for _, img := range r.be.Resources {
			held += uint64(len(img.Pix))
		}
		if held > budget {
			t.Fatalf("backend holds %d bytes over a %d budget", held, budget)
		}
	})
}
