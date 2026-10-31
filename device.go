// Package gpudev is a virtio-gpu device model: the host side of the
// control queue that go-virtio/gpu, the Linux virtio_gpu driver and any
// other guest driver talk to. It executes the 2D commands (resources,
// backing, transfers, scanouts, flushes), answers the capset queries,
// and hosts webgpu-batch contexts (go-virtio/gpu/wgpubatch) whose
// go-gfx/webgpu command streams it validates and hands to a Backend.
//
// The guest is untrusted. Every id, size, offset, count and rectangle a
// request carries is bounded in 64-bit arithmetic before it reaches the
// Backend or guest memory; quotas bound what a guest can make the host
// hold. References: the virtio spec, device-types/gpu/description.tex;
// Linux include/uapi/linux/virtio_gpu.h; QEMU hw/display/virtio-gpu.c
// (the reference 2D device, whose checks are cited where they are
// followed).
package gpudev

import (
	"encoding/binary"
	"sort"
	"sync"

	"github.com/go-virtio/gpu/wgpubatch"
)

var le = binary.LittleEndian

// Config describes the device a guest sees and the bounds it enforces.
// Zero values take the defaults documented on each field.
type Config struct {
	// Scanouts are the enabled displays' sizes, at most MaxScanouts.
	Scanouts []Rect
	// MaxHostMem bounds the bytes of all 2D resources (default 256 MiB,
	// QEMU's max_hostmem default).
	MaxHostMem uint64
	// MaxResources bounds the number of live resources (default 4096).
	MaxResources int
	// MaxBackingEntries bounds ATTACH_BACKING's nr_entries (default
	// 16384, QEMU virtio_gpu_create_mapping_iov).
	MaxBackingEntries uint32
	// WebGPUBatch, when set, offers the webgpu-batch capset with these
	// limits and accepts its contexts.
	WebGPUBatch *wgpubatch.Caps
	// MaxContexts bounds live contexts (default 16).
	MaxContexts int
}

func (c Config) withDefaults() Config {
	if c.MaxHostMem == 0 {
		c.MaxHostMem = 256 << 20
	}
	if c.MaxResources == 0 {
		c.MaxResources = 4096
	}
	if c.MaxBackingEntries == 0 {
		c.MaxBackingEntries = 16384
	}
	if c.MaxContexts == 0 {
		c.MaxContexts = 16
	}
	if len(c.Scanouts) > MaxScanouts {
		c.Scanouts = c.Scanouts[:MaxScanouts]
	}
	return c
}

// Device is one virtio-gpu device. Its methods are safe for concurrent
// use; commands execute one at a time.
type Device struct {
	mu       sync.Mutex
	cfg      Config
	mem      Memory
	be       Backend
	res      map[uint32]*resource
	hostmem  uint64
	scanouts [MaxScanouts]scanout
	ctxs     map[uint32]*batchContext
}

type resource struct {
	format  Format
	w, h    uint32
	backing []memEntry // nil: no backing attached
	starts  []uint64   // starts[i]: offset of backing[i] in the flattened backing
	length  uint64     // total backing bytes
}

type memEntry struct {
	addr uint64
	len  uint32
}

type scanout struct {
	res uint32
	r   Rect
}

// New returns a device over guest memory mem presenting through be.
func New(cfg Config, mem Memory, be Backend) *Device {
	return &Device{cfg: cfg.withDefaults(), mem: mem, be: be,
		res: map[uint32]*resource{}, ctxs: map[uint32]*batchContext{}}
}

// Features are the device feature bits (VERSION_1 is the transport's).
// With webgpu-batch: VIRGL, the "3D available" gate guests look for, and
// CONTEXT_INIT, which selects the capset.
func (d *Device) Features() uint64 {
	if d.cfg.WebGPUBatch != nil {
		return FeatureVirgl | FeatureContextInit
	}
	return 0
}

// NumScanouts and NumCapsets are the virtio_gpu_config fields.
func (d *Device) NumScanouts() uint32 { return uint32(len(d.cfg.Scanouts)) }

// NumCapsets is virtio_gpu_config.num_capsets.
func (d *Device) NumCapsets() uint32 {
	if d.cfg.WebGPUBatch != nil {
		return 1
	}
	return 0
}

// response is what a command produced: a response type and its body.
type response struct {
	typ  uint32
	body []byte
}

func errResp(t uint32) response { return response{typ: t} }

var okNoData = response{typ: RespOKNoData}

// HandleControl executes one control-queue request and writes the
// response into resp, returning its length (0 when resp cannot hold it:
// the guest gave a response buffer too short for the answer).
//
// req must be host memory, not a view of guest memory: the device
// checks a value and then uses it, and a guest rewriting shared memory
// between the two would defeat the check (virglrenderer copies its
// commands out for the same reason, vkr_ring.c:312). The pcisim
// transport gathers every request into a fresh buffer.
func (d *Device) HandleControl(req, resp []byte) int {
	var h hdr
	r := errResp(RespErrUnspec)
	if len(req) >= HdrSize {
		h = parseHdr(req)
		d.mu.Lock()
		r = d.dispatch(h, req)
		d.mu.Unlock()
	}
	n := HdrSize + len(r.body)
	if len(resp) < n {
		return 0
	}
	clear(resp[:HdrSize])
	le.PutUint32(resp[0:], r.typ)
	// description.tex:275-297: a fenced request's response carries the
	// fence, and its timeline (ctx, ring) when one was named.
	if h.flags&FlagFence != 0 {
		le.PutUint32(resp[4:], h.flags&(FlagFence|FlagInfoRingIdx))
		le.PutUint64(resp[8:], h.fence)
		le.PutUint32(resp[16:], h.ctx)
		resp[20] = h.ring
	}
	copy(resp[HdrSize:], r.body)
	return n
}

type hdr struct {
	typ, flags uint32
	fence      uint64
	ctx        uint32
	ring       uint8
}

func parseHdr(b []byte) hdr {
	return hdr{typ: le.Uint32(b[0:]), flags: le.Uint32(b[4:]), fence: le.Uint64(b[8:]),
		ctx: le.Uint32(b[16:]), ring: b[20]}
}

// bodySize is each command's struct size after the header. A request
// shorter than its struct is refused, as QEMU's VIRTIO_GPU_FILL_CMD does;
// trailing bytes are ignored.
var bodySize = map[uint32]int{
	CmdGetDisplayInfo: 0, CmdResourceCreate2D: 16, CmdResourceUnref: 8, CmdSetScanout: 24,
	CmdResourceFlush: 24, CmdTransferToHost2D: 32, CmdResourceAttachBacking: 8,
	CmdResourceDetachBacking: 8, CmdGetCapsetInfo: 8, CmdGetCapset: 8,
	CmdCtxCreate: 72, CmdCtxDestroy: 0, CmdCtxAttachResource: 8, CmdCtxDetachResource: 8,
	CmdSubmit3D: 8,
}

func (d *Device) dispatch(h hdr, req []byte) response {
	body := req[HdrSize:]
	n, known := bodySize[h.typ]
	if !known || len(body) < n {
		return errResp(RespErrUnspec)
	}
	switch h.typ {
	case CmdGetDisplayInfo:
		return d.displayInfo()
	case CmdResourceCreate2D:
		return d.create2D(le.Uint32(body[0:]), Format(le.Uint32(body[4:])), le.Uint32(body[8:]), le.Uint32(body[12:]))
	case CmdResourceUnref:
		return d.unref(le.Uint32(body[0:]))
	case CmdSetScanout:
		return d.setScanout(le.Uint32(body[16:]), le.Uint32(body[20:]), rectAt(body))
	case CmdResourceFlush:
		return d.flush(le.Uint32(body[16:]), rectAt(body))
	case CmdTransferToHost2D:
		return d.transfer(le.Uint32(body[24:]), rectAt(body), le.Uint64(body[16:]))
	case CmdResourceAttachBacking:
		return d.attach(le.Uint32(body[0:]), le.Uint32(body[4:]), body[8:])
	case CmdResourceDetachBacking:
		return d.detach(le.Uint32(body[0:]))
	case CmdGetCapsetInfo:
		return d.capsetInfo(le.Uint32(body[0:]))
	case CmdGetCapset:
		return d.capset(le.Uint32(body[0:]), le.Uint32(body[4:]))
	}
	return d.dispatch3D(h, body)
}

func rectAt(b []byte) Rect {
	return Rect{le.Uint32(b[0:]), le.Uint32(b[4:]), le.Uint32(b[8:]), le.Uint32(b[12:])}
}

func (d *Device) displayInfo() response {
	body := make([]byte, MaxScanouts*24) // struct virtio_gpu_display_one pmodes[16]
	for i, s := range d.cfg.Scanouts {
		o := body[24*i:]
		le.PutUint32(o[8:], s.Width)
		le.PutUint32(o[12:], s.Height)
		le.PutUint32(o[16:], 1) // enabled
	}
	return response{RespOKDisplayInfo, body}
}

// create2D: QEMU virtio-gpu.c:281-330 (id 0, duplicate id, unknown
// format refused; host memory bounded), plus zero sizes and a resource
// count bound.
func (d *Device) create2D(id uint32, f Format, w, h uint32) response {
	if id == 0 || d.res[id] != nil {
		return errResp(RespErrInvalidResourceID)
	}
	if !f.Valid() || w == 0 || h == 0 {
		return errResp(RespErrInvalidParameter)
	}
	// w*h fits in 64 bits; w*h*4 may not, so bound w*h first.
	px := uint64(w) * uint64(h)
	if len(d.res) >= d.cfg.MaxResources || px > (d.cfg.MaxHostMem-d.hostmem)/BytesPerPixel {
		return errResp(RespErrOutOfMemory)
	}
	size := px * BytesPerPixel
	if err := d.be.CreateResource(id, f, w, h); err != nil {
		return errResp(RespErrOutOfMemory)
	}
	d.res[id] = &resource{format: f, w: w, h: h}
	d.hostmem += size
	return okNoData
}

func (d *Device) unref(id uint32) response {
	r := d.res[id]
	if r == nil {
		return errResp(RespErrInvalidResourceID)
	}
	for i := range d.scanouts {
		if d.scanouts[i].res == id {
			d.scanouts[i] = scanout{}
			_ = d.be.SetScanout(uint32(i), 0, Rect{})
		}
	}
	for _, c := range d.ctxs {
		delete(c.attached, id)
	}
	d.be.DestroyResource(id)
	delete(d.res, id)
	d.hostmem -= uint64(r.w) * uint64(r.h) * BytesPerPixel
	return okNoData
}

// setScanout: QEMU virtio-gpu.c:825-872 and its bounds check :714-732
// (a rectangle of at least 16×16 inside the resource).
func (d *Device) setScanout(sid, id uint32, rect Rect) response {
	if uint64(sid) >= uint64(len(d.cfg.Scanouts)) {
		return errResp(RespErrInvalidScanoutID)
	}
	if id == 0 {
		d.scanouts[sid] = scanout{}
		if d.be.SetScanout(sid, 0, Rect{}) != nil {
			return errResp(RespErrUnspec)
		}
		return okNoData
	}
	r := d.res[id]
	if r == nil {
		return errResp(RespErrInvalidResourceID)
	}
	if rect.Width < 16 || rect.Height < 16 || !rect.within(r.w, r.h) {
		return errResp(RespErrInvalidParameter)
	}
	if d.be.SetScanout(sid, id, rect) != nil {
		return errResp(RespErrUnspec)
	}
	d.scanouts[sid] = scanout{res: id, r: rect}
	return okNoData
}

// flush: QEMU virtio-gpu.c:651-684: the rectangle must lie inside the
// resource; each scanout showing it gets the intersection.
func (d *Device) flush(id uint32, rect Rect) response {
	r := d.res[id]
	if r == nil {
		return errResp(RespErrInvalidResourceID)
	}
	if !rect.within(r.w, r.h) {
		return errResp(RespErrInvalidParameter)
	}
	for i, s := range d.scanouts {
		if s.res != id {
			continue
		}
		if in, ok := rect.intersect(s.r); ok {
			if d.be.Flush(uint32(i), id, in) != nil {
				return errResp(RespErrUnspec)
			}
		}
	}
	return okNoData
}

// transfer copies rect from the backing to the resource. Row y of the
// rectangle starts at backing offset off + y*stride, stride = width*4
// (QEMU virtio-gpu.c:580-595). Unlike QEMU, which truncates the 64-bit
// offset to uint32_t (:531, :582) and relies on iov_to_buf to stop short,
// the whole span is checked against the backing in 64-bit arithmetic and
// a transfer that does not fit is refused.
func (d *Device) transfer(id uint32, rect Rect, off uint64) response {
	r := d.res[id]
	if r == nil || r.backing == nil {
		return errResp(RespErrInvalidResourceID)
	}
	if !rect.within(r.w, r.h) {
		return errResp(RespErrInvalidParameter)
	}
	if rect.Width == 0 || rect.Height == 0 {
		return okNoData
	}
	stride := uint64(r.w) * BytesPerPixel
	row := uint64(rect.Width) * BytesPerPixel
	span := uint64(rect.Height-1)*stride + row // < 2^64: h, w < 2^32
	if off > r.length || span > r.length-off {
		return errResp(RespErrInvalidParameter)
	}
	src, srcStride, err := d.gather(r, off, span, stride, row, rect.Height)
	if err != nil {
		return errResp(RespErrUnspec)
	}
	if d.be.Upload(id, rect, src, srcStride) != nil {
		return errResp(RespErrUnspec)
	}
	return okNoData
}

// gather returns the rows of a transfer. When the span lies in one
// backing entry it is a view of guest memory (no copy: the Backend
// uploads straight from it); otherwise the rows are packed into a
// staging buffer.
func (d *Device) gather(r *resource, off, span, stride, row uint64, rows uint32) ([]byte, uint64, error) {
	i := entryAt(r, off)
	if off+span <= r.starts[i]+uint64(r.backing[i].len) {
		b, err := d.mem.Slice(r.backing[i].addr+(off-r.starts[i]), span)
		return b, stride, err
	}
	out := make([]byte, uint64(rows)*row)
	for y := uint64(0); y < uint64(rows); y++ {
		if err := d.copyBacking(r, off+y*stride, out[y*row:(y+1)*row], false); err != nil {
			return nil, 0, err
		}
	}
	return out, row, nil
}

// entryAt is the backing entry holding flattened offset off (< length).
func entryAt(r *resource, off uint64) int {
	return sort.Search(len(r.starts), func(i int) bool { return r.starts[i] > off }) - 1
}

// copyBacking copies len(p) bytes between p and the flattened backing at
// off, across entries (into the backing when write). The caller has
// bounded off+len(p) by r.length.
func (d *Device) copyBacking(r *resource, off uint64, p []byte, write bool) error {
	for len(p) > 0 {
		i := entryAt(r, off)
		in := off - r.starts[i]
		n := min(uint64(len(p)), uint64(r.backing[i].len)-in)
		b, err := d.mem.Slice(r.backing[i].addr+in, n)
		if err != nil {
			return err
		}
		if write {
			copy(b, p)
		} else {
			copy(p, b)
		}
		p, off = p[n:], off+n
	}
	return nil
}

// attach: QEMU virtio-gpu.c:993-1013 (nr_entries bounded, the entries
// must all be in the request). Each entry must also lie in guest memory
// and be non-empty; a resource takes one backing at a time.
func (d *Device) attach(id, n uint32, ents []byte) response {
	r := d.res[id]
	if r == nil {
		return errResp(RespErrInvalidResourceID)
	}
	if r.backing != nil || n == 0 || n > d.cfg.MaxBackingEntries || uint64(len(ents)) < 16*uint64(n) {
		return errResp(RespErrUnspec)
	}
	backing := make([]memEntry, n)
	starts := make([]uint64, n)
	var total uint64
	for i := range backing {
		e := ents[16*i:]
		backing[i] = memEntry{addr: le.Uint64(e[0:]), len: le.Uint32(e[8:])}
		if backing[i].len == 0 {
			return errResp(RespErrUnspec)
		}
		if _, err := d.mem.Slice(backing[i].addr, uint64(backing[i].len)); err != nil {
			return errResp(RespErrUnspec)
		}
		starts[i] = total
		total += uint64(backing[i].len) // <= 16384 * 2^32: no overflow
	}
	r.backing, r.starts, r.length = backing, starts, total
	return okNoData
}

func (d *Device) detach(id uint32) response {
	r := d.res[id]
	if r == nil {
		return errResp(RespErrInvalidResourceID)
	}
	r.backing, r.starts, r.length = nil, nil, 0
	return okNoData
}

func (d *Device) capsetInfo(index uint32) response {
	if index >= d.NumCapsets() {
		return errResp(RespErrInvalidParameter)
	}
	body := make([]byte, 16)
	le.PutUint32(body[0:], wgpubatch.CapsetID)
	le.PutUint32(body[4:], wgpubatch.Version)
	le.PutUint32(body[8:], wgpubatch.CapsSize)
	return response{RespOKCapsetInfo, body}
}

func (d *Device) capset(id, version uint32) response {
	if d.cfg.WebGPUBatch == nil || id != wgpubatch.CapsetID || version > wgpubatch.Version {
		return errResp(RespErrInvalidParameter)
	}
	return response{RespOKCapset, d.cfg.WebGPUBatch.Encode()}
}
