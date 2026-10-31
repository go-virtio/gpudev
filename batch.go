package gpudev

import (
	"errors"
	"fmt"

	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpu/wgpubatch"
)

// batchContext is a webgpu-batch context: the backend's half, and the
// translation of the guest's object ids (its namespace, private to the
// context) to the backend's.
type batchContext struct {
	be       Context
	ids      map[uint32]uint32 // guest id -> backend id
	attached map[uint32]bool   // resources attached with CTX_ATTACH_RESOURCE
	lost     bool              // a frame failed: the context accepts nothing more
}

// dispatch3D handles the context commands (description.tex:727-766).
func (d *Device) dispatch3D(h hdr, body []byte) response {
	caps := d.cfg.WebGPUBatch
	if caps == nil {
		return errResp(RespErrUnspec)
	}
	if h.typ == CmdCtxCreate {
		return d.ctxCreate(h.ctx, le.Uint32(body[0:]), le.Uint32(body[4:]))
	}
	c := d.ctxs[h.ctx]
	if c == nil {
		return errResp(RespErrInvalidContextID)
	}
	switch h.typ {
	case CmdCtxDestroy:
		c.be.Close()
		delete(d.ctxs, h.ctx)
		return okNoData
	case CmdCtxAttachResource, CmdCtxDetachResource:
		id := le.Uint32(body[0:])
		if d.res[id] == nil {
			return errResp(RespErrInvalidResourceID)
		}
		if h.typ == CmdCtxAttachResource {
			c.attached[id] = true
		} else {
			delete(c.attached, id)
		}
		return okNoData
	}
	// SUBMIT_3D: struct virtio_gpu_cmd_submit {le32 size; le32 padding},
	// then the frame.
	if c.lost {
		return errResp(RespErrInvalidContextID)
	}
	size := uint64(le.Uint32(body[0:]))
	if size > uint64(len(body)-8) || h.flags&FlagInfoRingIdx != 0 && uint32(h.ring) >= caps.MaxRings {
		return errResp(RespErrInvalidParameter)
	}
	if err := d.runFrame(c, body[8:8+size], caps); err != nil {
		c.lost = true
		return errResp(RespErrInvalidParameter)
	}
	return okNoData
}

// ctxCreate: the low 8 bits of context_init select the capset; 0 lets
// the device choose, and this device has one kind of context.
func (d *Device) ctxCreate(ctx, nlen, init uint32) response {
	if ctx == 0 || d.ctxs[ctx] != nil {
		return errResp(RespErrInvalidContextID)
	}
	if capset := init & 0xff; nlen > 64 || capset != 0 && capset != wgpubatch.CapsetID {
		return errResp(RespErrInvalidParameter)
	}
	if len(d.ctxs) >= d.cfg.MaxContexts {
		return errResp(RespErrOutOfMemory)
	}
	be, err := d.be.NewContext(ctx)
	if err != nil {
		return errResp(RespErrOutOfMemory)
	}
	d.ctxs[ctx] = &batchContext{be: be, ids: map[uint32]uint32{}, attached: map[uint32]bool{}}
	return okNoData
}

var errFrame = errors.New("gpudev: frame refused")

// runFrame executes a frame's messages in order. The frame is host memory
// (HandleControl's contract), so translating ids in place is safe.
func (d *Device) runFrame(c *batchContext, frame []byte, caps *wgpubatch.Caps) error {
	lim := caps.Limits()
	msgs, err := wgpubatch.Decode(frame, lim)
	if err != nil {
		return err
	}
	host := func(id uint32) (uint32, bool) { h, ok := c.ids[id]; return h, ok }
	for _, m := range msgs {
		switch m.Type {
		case wgpubatch.MsgCreate:
			cr, _ := m.Create(lim) // checked by Decode
			// Decode bounded the id below MaxObjects, which bounds the
			// live count too.
			if _, live := c.ids[cr.ID]; live || !wire.Kind(cr.Kind).Valid() {
				return fmt.Errorf("%w: CREATE %d", errFrame, cr.ID)
			}
			if err := wire.RemapDescriptor(cr.Descriptor, host); err != nil {
				return err
			}
			id, err := c.be.Create(cr.Kind, cr.Descriptor)
			if err != nil {
				return err
			}
			c.ids[cr.ID] = id
		case wgpubatch.MsgDestroy:
			ids, _ := m.Destroy(lim)
			for _, id := range ids {
				h, live := c.ids[id]
				if !live {
					return fmt.Errorf("%w: DESTROY %d", errFrame, id)
				}
				c.be.Destroy(h)
				delete(c.ids, id)
			}
		case wgpubatch.MsgStream:
			s, _ := m.Stream()
			if err := wire.Remap(s.Words, uint32(len(s.Data)), host); err != nil {
				return err
			}
			if err := c.be.Replay(s.Words, s.Data); err != nil {
				return err
			}
		case wgpubatch.MsgBindResource:
			b, _ := m.BindResource(lim)
			h, live := c.ids[b.ID]
			r := d.res[b.ResourceID]
			if !live || r == nil || !c.attached[b.ResourceID] || b.Offset > r.length || b.Size > r.length-b.Offset {
				return fmt.Errorf("%w: BIND_RESOURCE %d", errFrame, b.ID)
			}
			if err := c.be.Bind(h, &window{d: d, r: r, off: b.Offset, n: b.Size}); err != nil {
				return err
			}
		}
	}
	return nil
}

// window is a range of a resource's backing. It is checked against the
// backing at every access: a detached backing ends it. A backend uses it
// only during the Context calls the device makes (under the device's
// lock), never from another goroutine.
type window struct {
	d      *Device
	r      *resource
	off, n uint64
}

// ErrWindow: an access outside a window, or after its backing went away.
var ErrWindow = errors.New("gpudev: access outside the bound window")

func (w *window) Len() uint64 { return w.n }

func (w *window) check(p []byte, off uint64) error {
	if off > w.n || uint64(len(p)) > w.n-off || w.off+w.n > w.r.length {
		return ErrWindow
	}
	return nil
}

func (w *window) ReadAt(p []byte, off uint64) error {
	if err := w.check(p, off); err != nil {
		return err
	}
	return w.d.copyBacking(w.r, w.off+off, p, false)
}

func (w *window) WriteAt(p []byte, off uint64) error {
	if err := w.check(p, off); err != nil {
		return err
	}
	return w.d.copyBacking(w.r, w.off+off, p, true)
}
