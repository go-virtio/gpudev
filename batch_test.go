package gpudev_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-virtio/gpudev"
	"github.com/go-virtio/gpudev/memback"
)

var testCaps = wgpubatch.Caps{Version: 1, StreamVersion: wire.StreamVersion, MaxFrame: 1 << 16, MaxObjects: 8, MaxRings: 2, MaxMemory: 1 << 20}

func batchRig() *rig {
	caps := testCaps
	return newRig(gpudev.Config{Scanouts: []gpudev.Rect{{Width: 16, Height: 16}}, WebGPUBatch: &caps, MaxContexts: 2})
}

func ctxCreate(ctx, nlen, init uint32) []byte {
	b := cmd(gpudev.CmdCtxCreate, nlen, init)
	le.PutUint32(b[16:], ctx)
	return append(b, make([]byte, 64)...)
}

func ctxCmd(typ, ctx uint32, words ...uint32) []byte {
	b := cmd(typ, words...)
	le.PutUint32(b[16:], ctx)
	return b
}

func submit(ctx uint32, frame []byte) []byte {
	b := ctxCmd(gpudev.CmdSubmit3D, ctx, uint32(len(frame)), 0)
	return append(b, frame...)
}

func frame(t *testing.T, build func(e *wgpubatch.Encoder)) []byte {
	t.Helper()
	e := wgpubatch.NewEncoder(testCaps.MaxFrame)
	build(e)
	f, err := e.Frame()
	if err != nil {
		t.Fatal(err)
	}
	return append([]byte(nil), f...)
}

func desc(refs ...uint32) []byte { return wire.AppendDescriptor(nil, refs, []byte(`{}`)) }

func TestContexts(t *testing.T) {
	r := batchRig()
	r.want(t, "ctx 0", ctxCreate(0, 0, 30), gpudev.RespErrInvalidContextID)
	r.want(t, "long name", ctxCreate(1, 65, 30), gpudev.RespErrInvalidParameter)
	r.want(t, "venus", ctxCreate(1, 0, 4), gpudev.RespErrInvalidParameter)
	r.want(t, "create", ctxCreate(1, 4, 30), gpudev.RespOKNoData)
	r.want(t, "duplicate", ctxCreate(1, 0, 30), gpudev.RespErrInvalidContextID)
	r.want(t, "device's choice", ctxCreate(2, 0, 0), gpudev.RespOKNoData)
	r.want(t, "too many", ctxCreate(3, 0, 30), gpudev.RespErrOutOfMemory)
	r.want(t, "unknown ctx", ctxCmd(gpudev.CmdCtxDestroy, 9), gpudev.RespErrInvalidContextID)
	r.want(t, "destroy", ctxCmd(gpudev.CmdCtxDestroy, 2), gpudev.RespOKNoData)
	if r.be.Contexts[2] != nil {
		t.Error("destroyed context not closed")
	}
	r.be.Fail = errors.New("down")
	r.want(t, "backend refuses", ctxCreate(2, 0, 30), gpudev.RespErrOutOfMemory)
	r.be.Fail = nil

	r.want(t, "attach unknown", ctxCmd(gpudev.CmdCtxAttachResource, 1, 5, 0), gpudev.RespErrInvalidResourceID)
	r.want(t, "res", create(5, gpudev.FormatB8G8R8A8, 4, 4), gpudev.RespOKNoData)
	r.want(t, "attach", ctxCmd(gpudev.CmdCtxAttachResource, 1, 5, 0), gpudev.RespOKNoData)
	r.want(t, "detach", ctxCmd(gpudev.CmdCtxDetachResource, 1, 5, 0), gpudev.RespOKNoData)
	r.want(t, "attach again", ctxCmd(gpudev.CmdCtxAttachResource, 1, 5, 0), gpudev.RespOKNoData)
	r.want(t, "unref attached", cmd(gpudev.CmdResourceUnref, 5, 0), gpudev.RespOKNoData)
}

// TestSubmit: guest ids are translated to the backend's before the
// backend sees a descriptor or a stream; memback refuses any stream that
// names an id it did not hand out, so these pass only if every id was
// translated.
func TestSubmit(t *testing.T) {
	r := batchRig()
	r.want(t, "create", ctxCreate(1, 0, 30), gpudev.RespOKNoData)
	f := frame(t, func(e *wgpubatch.Encoder) {
		e.Create(1, uint32(wire.KindBuffer), desc())
		e.Create(2, uint32(wire.KindBindGroup), desc(1, 1))
		e.Stream([]uint32{uint32(wire.WriteBuffer), 1, 0, 0, 0, 4, uint32(wire.Submit)}, []byte{1, 2, 3, 4})
	})
	req := submit(1, f)
	le.PutUint32(req[4:], gpudev.FlagFence|gpudev.FlagInfoRingIdx)
	req[20] = 1
	r.want(t, "submit", req, gpudev.RespOKNoData)
	c := r.be.Contexts[1]
	if len(c.Objects) != 2 || len(c.Replays) != 1 || c.Replays[0][0].Args[0] != memback.FirstID {
		t.Fatalf("objects %v replays %v", c.Objects, c.Replays)
	}
	refs, _, _ := wire.SplitDescriptor(c.Objects[memback.FirstID+1].Desc)
	if refs[0] != memback.FirstID || refs[1] != memback.FirstID {
		t.Errorf("descriptor references not translated: %v", refs)
	}
	r.want(t, "destroy", submit(1, frame(t, func(e *wgpubatch.Encoder) { e.Destroy(2) })), gpudev.RespOKNoData)
	if len(c.Objects) != 1 {
		t.Error("DESTROY did not reach the backend")
	}

	ring := submit(1, frame(t, func(e *wgpubatch.Encoder) {}))
	le.PutUint32(ring[4:], gpudev.FlagInfoRingIdx)
	ring[20] = 2
	r.want(t, "ring past MaxRings", ring, gpudev.RespErrInvalidParameter)
	short := submit(1, f)
	le.PutUint32(short[gpudev.HdrSize:], uint32(len(f)+1))
	r.want(t, "size past the request", short, gpudev.RespErrInvalidParameter)
	r.want(t, "unknown ctx", submit(9, f), gpudev.RespErrInvalidContextID)
}

// TestSubmitRefuses: each bad frame loses its context.
func TestSubmitRefuses(t *testing.T) {
	cases := map[string]func(e *wgpubatch.Encoder){
		"duplicate id":    func(e *wgpubatch.Encoder) { e.Create(1, 1, desc()); e.Create(1, 1, desc()) },
		"bad kind":        func(e *wgpubatch.Encoder) { e.Create(1, 99, desc()) },
		"bad descriptor":  func(e *wgpubatch.Encoder) { e.Create(1, 1, []byte{1}) },
		"unknown ref":     func(e *wgpubatch.Encoder) { e.Create(1, 8, desc(5)) },
		"destroy unknown": func(e *wgpubatch.Encoder) { e.Destroy(3) },
		"stream unknown": func(e *wgpubatch.Encoder) {
			e.Stream([]uint32{uint32(wire.WriteBuffer), 4, 0, 0, 0, 0}, nil)
		},
		"id past MaxObjects": func(e *wgpubatch.Encoder) { e.Create(8, 1, desc()) },
		"bind unknown object": func(e *wgpubatch.Encoder) {
			e.BindResource(wgpubatch.BindResource{ID: 3, ResourceID: 5, Size: 1})
		},
		"bind unattached": func(e *wgpubatch.Encoder) {
			e.Create(1, 1, desc())
			e.BindResource(wgpubatch.BindResource{ID: 1, ResourceID: 6, Size: 1})
		},
		"bind unknown resource": func(e *wgpubatch.Encoder) {
			e.Create(1, 1, desc())
			e.BindResource(wgpubatch.BindResource{ID: 1, ResourceID: 7, Size: 1})
		},
		"bind past backing": func(e *wgpubatch.Encoder) {
			e.Create(1, 1, desc())
			e.BindResource(wgpubatch.BindResource{ID: 1, ResourceID: 5, Offset: 60, Size: 5})
		},
		"bind offset past backing": func(e *wgpubatch.Encoder) {
			e.Create(1, 1, desc())
			e.BindResource(wgpubatch.BindResource{ID: 1, ResourceID: 5, Offset: 65, Size: 1})
		},
	}
	for name, build := range cases {
		r := batchRig()
		r.want(t, name+": ctx", ctxCreate(1, 0, 30), gpudev.RespOKNoData)
		r.want(t, name+": res", create(5, gpudev.FormatB8G8R8A8, 4, 4), gpudev.RespOKNoData)
		r.want(t, name+": res 6", create(6, gpudev.FormatB8G8R8A8, 4, 4), gpudev.RespOKNoData)
		r.want(t, name+": backing", attach(5, [2]uint64{base, 64}), gpudev.RespOKNoData)
		r.want(t, name+": attach", ctxCmd(gpudev.CmdCtxAttachResource, 1, 5, 0), gpudev.RespOKNoData)
		r.want(t, name, submit(1, frame(t, build)), gpudev.RespErrInvalidParameter)
		r.want(t, name+": lost", submit(1, frame(t, func(*wgpubatch.Encoder) {})), gpudev.RespErrInvalidContextID)
	}
	r := batchRig()
	r.want(t, "ctx", ctxCreate(1, 0, 30), gpudev.RespOKNoData)
	r.want(t, "malformed frame", submit(1, []byte("not a frame at all")), gpudev.RespErrInvalidParameter)

	for name, build := range map[string]func(e *wgpubatch.Encoder){
		"backend create": func(e *wgpubatch.Encoder) { e.Create(1, 1, desc()) },
		"backend replay": func(e *wgpubatch.Encoder) { e.Stream(nil, nil) },
	} {
		r := batchRig()
		r.want(t, name+": ctx", ctxCreate(1, 0, 30), gpudev.RespOKNoData)
		r.be.Fail = errors.New("down")
		r.want(t, name, submit(1, frame(t, build)), gpudev.RespErrInvalidParameter)
	}
	r = batchRig()
	r.want(t, "ctx", ctxCreate(1, 0, 30), gpudev.RespOKNoData)
	r.want(t, "res", create(5, gpudev.FormatB8G8R8A8, 4, 4), gpudev.RespOKNoData)
	r.want(t, "backing", attach(5, [2]uint64{base, 64}), gpudev.RespOKNoData)
	r.want(t, "attach", ctxCmd(gpudev.CmdCtxAttachResource, 1, 5, 0), gpudev.RespOKNoData)
	r.want(t, "obj", submit(1, frame(t, func(e *wgpubatch.Encoder) { e.Create(1, 1, desc()) })), gpudev.RespOKNoData)
	r.be.Fail = errors.New("down")
	r.want(t, "backend bind", submit(1, frame(t, func(e *wgpubatch.Encoder) {
		e.BindResource(wgpubatch.BindResource{ID: 1, ResourceID: 5, Size: 8})
	})), gpudev.RespErrInvalidParameter)
}

// TestWindow: a bound window reads and writes the backing across entries,
// within its bounds only, and ends when the backing is detached.
func TestWindow(t *testing.T) {
	r := batchRig()
	ram := r.mem.Bytes
	r.want(t, "ctx", ctxCreate(1, 0, 30), gpudev.RespOKNoData)
	r.want(t, "res", create(5, gpudev.FormatB8G8R8A8, 4, 4), gpudev.RespOKNoData)
	r.want(t, "backing", attach(5, [2]uint64{base, 16}, [2]uint64{base + 1024, 16}), gpudev.RespOKNoData)
	r.want(t, "attach", ctxCmd(gpudev.CmdCtxAttachResource, 1, 5, 0), gpudev.RespOKNoData)
	r.want(t, "bind", submit(1, frame(t, func(e *wgpubatch.Encoder) {
		e.Create(1, 1, desc())
		e.BindResource(wgpubatch.BindResource{ID: 1, ResourceID: 5, Offset: 8, Size: 16})
	})), gpudev.RespOKNoData)
	w := r.be.Contexts[1].Binds[memback.FirstID]
	if w.Len() != 16 {
		t.Fatalf("window of %d bytes", w.Len())
	}
	if err := w.WriteAt([]byte("ABCDEFGHIJKLMNOP"), 0); err != nil {
		t.Fatal(err)
	}
	if string(ram[8:16]) != "ABCDEFGH" || string(ram[1024:1032]) != "IJKLMNOP" {
		t.Fatalf("written across entries: %q %q", ram[8:16], ram[1024:1032])
	}
	p := make([]byte, 4)
	if err := w.ReadAt(p, 6); err != nil || string(p) != "GHIJ" {
		t.Fatalf("read %q %v", p, err)
	}
	if err := w.ReadAt(p, 13); !errors.Is(err, gpudev.ErrWindow) {
		t.Error("read past the window")
	}
	if err := w.WriteAt(p, 17); !errors.Is(err, gpudev.ErrWindow) {
		t.Error("write past the window")
	}
	r.mem.calls, r.mem.failAfter = 0, 1
	if err := w.ReadAt(p, 0); err == nil {
		t.Error("vanished memory read")
	}
	r.mem.failAfter = 0
	r.want(t, "detach", cmd(gpudev.CmdResourceDetachBacking, 5, 0), gpudev.RespOKNoData)
	if err := w.ReadAt(p, 0); !errors.Is(err, gpudev.ErrWindow) {
		t.Error("window outlived its backing")
	}
	if !bytes.Equal(p, []byte("GHIJ")) {
		t.Error("a refused read wrote")
	}
}
