package gpudev_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"testing"

	"github.com/go-virtio/gpu/wgpubatch"
	"github.com/go-virtio/gpudev"
	"github.com/go-virtio/gpudev/memback"
)

var le = binary.LittleEndian

const base = 1 << 20

// flakyMem is RAM whose Slice can be made to fail, for the error paths
// of memory that disappears after it was checked.
type flakyMem struct {
	gpudev.RAM
	failAfter int // >0: the failAfter-th Slice from now on fails
	calls     int
}

func (m *flakyMem) Slice(addr, n uint64) ([]byte, error) {
	m.calls++
	if m.failAfter > 0 && m.calls >= m.failAfter {
		return nil, gpudev.ErrOutOfRange
	}
	return m.RAM.Slice(addr, n)
}

type rig struct {
	d   *gpudev.Device
	be  *memback.Backend
	mem *flakyMem
}

func newRig(cfg gpudev.Config) *rig {
	m := &flakyMem{RAM: gpudev.RAM{Base: base, Bytes: make([]byte, 1<<20)}}
	be := memback.New()
	return &rig{d: gpudev.New(cfg, m, be), be: be, mem: m}
}

func defaultRig() *rig {
	return newRig(gpudev.Config{Scanouts: []gpudev.Rect{{Width: 64, Height: 32}, {Width: 32, Height: 32}}})
}

// cmd builds a request: header with type, then little-endian words.
func cmd(typ uint32, words ...uint32) []byte {
	b := make([]byte, gpudev.HdrSize, gpudev.HdrSize+4*len(words))
	le.PutUint32(b, typ)
	for _, w := range words {
		b = le.AppendUint32(b, w)
	}
	return b
}

func lo(v uint64) uint32 { return uint32(v) }
func hi(v uint64) uint32 { return uint32(v >> 32) }

func (r *rig) call(req []byte) (uint32, []byte) {
	resp := make([]byte, 1024)
	n := r.d.HandleControl(req, resp)
	if n < gpudev.HdrSize {
		return 0, nil
	}
	return le.Uint32(resp), resp[gpudev.HdrSize:n]
}

func (r *rig) want(t *testing.T, name string, req []byte, typ uint32) {
	t.Helper()
	if got, _ := r.call(req); got != typ {
		t.Errorf("%s: response %#x, want %#x", name, got, typ)
	}
}

func create(id uint32, f gpudev.Format, w, h uint32) []byte {
	return cmd(gpudev.CmdResourceCreate2D, id, uint32(f), w, h)
}

func attach(id uint32, ents ...[2]uint64) []byte {
	b := cmd(gpudev.CmdResourceAttachBacking, id, uint32(len(ents)))
	for _, e := range ents {
		b = le.AppendUint64(b, e[0])
		b = le.AppendUint32(b, uint32(e[1]))
		b = le.AppendUint32(b, 0)
	}
	return b
}

func transfer(id uint32, r gpudev.Rect, off uint64) []byte {
	return cmd(gpudev.CmdTransferToHost2D, r.X, r.Y, r.Width, r.Height, lo(off), hi(off), id, 0)
}

func setScanout(sid, id uint32, r gpudev.Rect) []byte {
	return cmd(gpudev.CmdSetScanout, r.X, r.Y, r.Width, r.Height, sid, id)
}

func flush(id uint32, r gpudev.Rect) []byte {
	return cmd(gpudev.CmdResourceFlush, r.X, r.Y, r.Width, r.Height, id, 0)
}

func TestHeaderAndFraming(t *testing.T) {
	r := defaultRig()
	resp := make([]byte, 64)
	if n := r.d.HandleControl(make([]byte, 10), resp); n != gpudev.HdrSize || le.Uint32(resp) != gpudev.RespErrUnspec {
		t.Errorf("short request: %d %#x", n, le.Uint32(resp))
	}
	r.want(t, "unknown", cmd(0x0999), gpudev.RespErrUnspec)
	r.want(t, "short body", cmd(gpudev.CmdResourceCreate2D, 1, 1, 1), gpudev.RespErrUnspec)
	if n := r.d.HandleControl(cmd(gpudev.CmdGetDisplayInfo), make([]byte, 100)); n != 0 {
		t.Errorf("response buffer too short: wrote %d", n)
	}
	// A fenced request: the response echoes the fence and its timeline.
	req := cmd(gpudev.CmdResourceUnref, 99, 0)
	le.PutUint32(req[4:], gpudev.FlagFence|gpudev.FlagInfoRingIdx)
	le.PutUint64(req[8:], 0x1122334455)
	le.PutUint32(req[16:], 7)
	req[20] = 3
	n := r.d.HandleControl(req, resp)
	if n != gpudev.HdrSize || le.Uint32(resp[0:]) != gpudev.RespErrInvalidResourceID ||
		le.Uint32(resp[4:]) != gpudev.FlagFence|gpudev.FlagInfoRingIdx || le.Uint64(resp[8:]) != 0x1122334455 ||
		le.Uint32(resp[16:]) != 7 || resp[20] != 3 {
		t.Errorf("fenced response %x", resp[:n])
	}
	// Unfenced: flags stay clear.
	le.PutUint32(req[4:], 0)
	r.d.HandleControl(req, resp)
	if le.Uint32(resp[4:]) != 0 || le.Uint64(resp[8:]) != 0 {
		t.Errorf("unfenced response carries a fence: %x", resp[:24])
	}
}

func TestDisplayInfo(t *testing.T) {
	r := defaultRig()
	typ, body := r.call(cmd(gpudev.CmdGetDisplayInfo))
	if typ != gpudev.RespOKDisplayInfo || len(body) != 16*24 {
		t.Fatalf("%#x, %d bytes", typ, len(body))
	}
	if le.Uint32(body[8:]) != 64 || le.Uint32(body[12:]) != 32 || le.Uint32(body[16:]) != 1 ||
		le.Uint32(body[24+8:]) != 32 || le.Uint32(body[48+16:]) != 0 {
		t.Errorf("display_one %x", body[:72])
	}
	many := make([]gpudev.Rect, 20)
	if n := newRig(gpudev.Config{Scanouts: many}).d.NumScanouts(); n != gpudev.MaxScanouts {
		t.Errorf("%d scanouts", n)
	}
}

func TestCreateUnref(t *testing.T) {
	r := newRig(gpudev.Config{Scanouts: []gpudev.Rect{{Width: 16, Height: 16}}, MaxHostMem: 4096, MaxResources: 2})
	r.want(t, "id 0", create(0, gpudev.FormatB8G8R8A8, 4, 4), gpudev.RespErrInvalidResourceID)
	r.want(t, "format", create(1, 5, 4, 4), gpudev.RespErrInvalidParameter)
	r.want(t, "zero width", create(1, gpudev.FormatB8G8R8A8, 0, 4), gpudev.RespErrInvalidParameter)
	r.want(t, "zero height", create(1, gpudev.FormatB8G8R8A8, 4, 0), gpudev.RespErrInvalidParameter)
	r.want(t, "over host mem", create(1, gpudev.FormatB8G8R8A8, 33, 32), gpudev.RespErrOutOfMemory)
	r.want(t, "w*h*4 overflow", create(1, gpudev.FormatB8G8R8A8, 0xFFFFFFFF, 0xFFFFFFFF), gpudev.RespErrOutOfMemory)
	r.want(t, "ok", create(1, gpudev.FormatB8G8R8A8, 16, 16), gpudev.RespOKNoData)
	r.want(t, "duplicate", create(1, gpudev.FormatB8G8R8A8, 4, 4), gpudev.RespErrInvalidResourceID)
	r.want(t, "rest of host mem", create(2, gpudev.FormatR8G8B8X8, 16, 49), gpudev.RespErrOutOfMemory)
	r.want(t, "fits", create(2, gpudev.FormatR8G8B8X8, 16, 48), gpudev.RespOKNoData)
	r.want(t, "count", create(3, gpudev.FormatB8G8R8A8, 1, 1), gpudev.RespErrOutOfMemory)
	r.want(t, "show", setScanout(0, 1, gpudev.Rect{Width: 16, Height: 16}), gpudev.RespOKNoData)
	r.want(t, "unref unknown", cmd(gpudev.CmdResourceUnref, 9, 0), gpudev.RespErrInvalidResourceID)
	r.want(t, "unref", cmd(gpudev.CmdResourceUnref, 1, 0), gpudev.RespOKNoData)
	if _, shown := r.be.Shown[0]; shown || r.be.Resources[1] != nil {
		t.Error("unref left the resource shown or alive")
	}
	r.want(t, "host mem refunded", create(1, gpudev.FormatB8G8R8A8, 16, 16), gpudev.RespOKNoData)
	r.want(t, "unref 2", cmd(gpudev.CmdResourceUnref, 2, 0), gpudev.RespOKNoData)
	r.be.Fail = errors.New("backend down")
	r.want(t, "backend refuses", create(2, gpudev.FormatB8G8R8A8, 1, 1), gpudev.RespErrOutOfMemory)
	for _, f := range []gpudev.Format{1, 2, 3, 4, 67, 68, 121, 134} {
		if !f.Valid() {
			t.Errorf("format %d invalid", f)
		}
	}
}

func TestScanoutAndFlush(t *testing.T) {
	r := defaultRig()
	full := gpudev.Rect{Width: 64, Height: 32}
	r.want(t, "create", create(1, gpudev.FormatB8G8R8A8, 64, 32), gpudev.RespOKNoData)
	r.want(t, "scanout id", setScanout(2, 1, full), gpudev.RespErrInvalidScanoutID)
	r.want(t, "scanout huge id", setScanout(0xFFFFFFFF, 1, full), gpudev.RespErrInvalidScanoutID)
	r.want(t, "unknown resource", setScanout(0, 9, full), gpudev.RespErrInvalidResourceID)
	r.want(t, "below 16", setScanout(0, 1, gpudev.Rect{Width: 15, Height: 32}), gpudev.RespErrInvalidParameter)
	r.want(t, "below 16 high", setScanout(0, 1, gpudev.Rect{Width: 16, Height: 15}), gpudev.RespErrInvalidParameter)
	r.want(t, "outside", setScanout(0, 1, gpudev.Rect{X: 1, Width: 64, Height: 32}), gpudev.RespErrInvalidParameter)
	r.want(t, "wrap", setScanout(0, 1, gpudev.Rect{X: 0xFFFFFFF0, Width: 32, Height: 32}), gpudev.RespErrInvalidParameter)
	r.want(t, "show 0", setScanout(0, 1, full), gpudev.RespOKNoData)
	r.want(t, "show 1 (part)", setScanout(1, 1, gpudev.Rect{X: 32, Width: 32, Height: 32}), gpudev.RespOKNoData)

	r.want(t, "flush unknown", flush(9, full), gpudev.RespErrInvalidResourceID)
	r.want(t, "flush outside", flush(1, gpudev.Rect{Y: 1, Width: 64, Height: 32}), gpudev.RespErrInvalidParameter)
	// Left half: scanout 0 only (scanout 1 shows x >= 32).
	r.want(t, "flush left", flush(1, gpudev.Rect{Width: 32, Height: 32}), gpudev.RespOKNoData)
	if r.be.Flushes != 1 || r.be.Scanouts[1] != nil {
		t.Errorf("left flush reached %d scanouts", r.be.Flushes)
	}
	r.want(t, "flush all", flush(1, full), gpudev.RespOKNoData)
	if r.be.Flushes != 3 {
		t.Errorf("full flush: %d flushes", r.be.Flushes)
	}
	r.want(t, "disable", setScanout(1, 0, gpudev.Rect{}), gpudev.RespOKNoData)
	r.be.Fail = errors.New("backend down")
	r.want(t, "flush fails", flush(1, full), gpudev.RespErrUnspec)
	r.want(t, "show fails", setScanout(1, 1, full), gpudev.RespErrUnspec)
	r.want(t, "disable fails", setScanout(1, 0, gpudev.Rect{}), gpudev.RespErrUnspec)
}

func TestAttachTransfer(t *testing.T) {
	r := newRig(gpudev.Config{Scanouts: []gpudev.Rect{{Width: 16, Height: 16}}, MaxBackingEntries: 4})
	ram := r.mem.Bytes
	for i := range ram {
		ram[i] = byte(i * 13)
	}
	r.want(t, "create", create(1, gpudev.FormatB8G8R8A8, 16, 16), gpudev.RespOKNoData)
	r.want(t, "transfer without backing", transfer(1, gpudev.Rect{Width: 1, Height: 1}, 0), gpudev.RespErrInvalidResourceID)
	r.want(t, "attach unknown", attach(9, [2]uint64{base, 1024}), gpudev.RespErrInvalidResourceID)
	r.want(t, "attach none", attach(1), gpudev.RespErrUnspec)
	r.want(t, "attach too many", attach(1, [2]uint64{base, 1}, [2]uint64{base, 1}, [2]uint64{base, 1}, [2]uint64{base, 1}, [2]uint64{base, 1}), gpudev.RespErrUnspec)
	short := attach(1, [2]uint64{base, 1024})
	r.want(t, "entries missing", short[:len(short)-1], gpudev.RespErrUnspec)
	r.want(t, "empty entry", attach(1, [2]uint64{base, 0}), gpudev.RespErrUnspec)
	r.want(t, "entry outside RAM", attach(1, [2]uint64{base + 1<<20 - 10, 64}), gpudev.RespErrUnspec)
	// Two entries, the second far from the first: a transfer that spans
	// both is gathered; one inside the first is a direct view.
	r.want(t, "attach", attach(1, [2]uint64{base + 4096, 512}, [2]uint64{base + 65536, 512}), gpudev.RespOKNoData)
	r.want(t, "attach twice", attach(1, [2]uint64{base, 1024}), gpudev.RespErrUnspec)

	r.want(t, "transfer unknown", transfer(9, gpudev.Rect{Width: 1, Height: 1}, 0), gpudev.RespErrInvalidResourceID)
	r.want(t, "transfer outside", transfer(1, gpudev.Rect{X: 8, Width: 9, Height: 1}, 0), gpudev.RespErrInvalidParameter)
	r.want(t, "empty transfer", transfer(1, gpudev.Rect{Width: 0, Height: 4}, 0), gpudev.RespOKNoData)
	r.want(t, "whole backing", transfer(1, gpudev.Rect{Width: 16, Height: 16}, 0), gpudev.RespOKNoData)
	r.want(t, "past backing", transfer(1, gpudev.Rect{Width: 16, Height: 16}, 64), gpudev.RespErrInvalidParameter)
	r.want(t, "offset past backing", transfer(1, gpudev.Rect{Width: 1, Height: 1}, 1025), gpudev.RespErrInvalidParameter)
	r.want(t, "huge offset", transfer(1, gpudev.Rect{Width: 1, Height: 1}, 1<<63), gpudev.RespErrInvalidParameter)

	// Rows 0..7 at offset 0 lie in entry 0 (8*64 = 512 bytes).
	r.want(t, "direct", transfer(1, gpudev.Rect{Width: 16, Height: 8}, 0), gpudev.RespOKNoData)
	img := r.be.Resources[1].Pix
	if !bytes.Equal(img[:512], ram[4096:4608]) {
		t.Error("direct transfer: wrong pixels")
	}
	// Rows 4..11, columns 2..5: offset 4*64+2*4, crossing into entry 1.
	rect := gpudev.Rect{X: 2, Y: 4, Width: 4, Height: 8}
	r.want(t, "gathered", transfer(1, rect, 4*64+8), gpudev.RespOKNoData)
	flat := append(append([]byte(nil), ram[4096:4608]...), ram[65536:66048]...)
	for y := 4; y < 12; y++ {
		o := y*64 + 8
		if !bytes.Equal(img[o:o+16], flat[o:o+16]) {
			t.Fatalf("gathered row %d differs", y)
		}
	}
	r.be.Fail = errors.New("backend down")
	r.want(t, "upload fails", transfer(1, rect, 4*64+8), gpudev.RespErrUnspec)
	r.be.Fail = nil

	// Guest memory that vanishes after the attach check.
	r.mem.calls, r.mem.failAfter = 0, 1
	r.want(t, "direct view fails", transfer(1, gpudev.Rect{Width: 1, Height: 1}, 0), gpudev.RespErrUnspec)
	r.mem.calls, r.mem.failAfter = 0, 1
	r.want(t, "gather fails", transfer(1, rect, 4*64+8), gpudev.RespErrUnspec)
	r.mem.failAfter = 0

	r.want(t, "detach unknown", cmd(gpudev.CmdResourceDetachBacking, 9, 0), gpudev.RespErrInvalidResourceID)
	r.want(t, "detach", cmd(gpudev.CmdResourceDetachBacking, 1, 0), gpudev.RespOKNoData)
	r.want(t, "transfer after detach", transfer(1, rect, 0), gpudev.RespErrInvalidResourceID)
}

func TestCapsets(t *testing.T) {
	r := defaultRig()
	if r.d.NumCapsets() != 0 || r.d.Features() != 0 {
		t.Error("capsets or features without webgpu-batch")
	}
	r.want(t, "no capset", cmd(gpudev.CmdGetCapsetInfo, 0, 0), gpudev.RespErrInvalidParameter)
	r.want(t, "no capset data", cmd(gpudev.CmdGetCapset, 30, 1), gpudev.RespErrInvalidParameter)

	caps := wgpubatch.Caps{Version: 1, StreamVersion: 1, MaxFrame: 1 << 16, MaxObjects: 64, MaxRings: 1, MaxMemory: 1 << 20}
	r = newRig(gpudev.Config{Scanouts: []gpudev.Rect{{Width: 16, Height: 16}}, WebGPUBatch: &caps})
	if r.d.NumCapsets() != 1 || r.d.Features() != gpudev.FeatureVirgl|gpudev.FeatureContextInit {
		t.Error("webgpu-batch not offered")
	}
	typ, body := r.call(cmd(gpudev.CmdGetCapsetInfo, 0, 0))
	if typ != gpudev.RespOKCapsetInfo || le.Uint32(body[0:]) != 30 || le.Uint32(body[4:]) != 1 || le.Uint32(body[8:]) != wgpubatch.CapsSize {
		t.Errorf("capset info %#x %x", typ, body)
	}
	r.want(t, "index 1", cmd(gpudev.CmdGetCapsetInfo, 1, 0), gpudev.RespErrInvalidParameter)
	typ, body = r.call(cmd(gpudev.CmdGetCapset, 30, 1))
	if got, err := wgpubatch.DecodeCaps(body); typ != gpudev.RespOKCapset || err != nil || got != caps {
		t.Errorf("capset %#x %+v %v", typ, got, err)
	}
	r.want(t, "other id", cmd(gpudev.CmdGetCapset, 4, 0), gpudev.RespErrInvalidParameter)
	r.want(t, "future version", cmd(gpudev.CmdGetCapset, 30, 2), gpudev.RespErrInvalidParameter)
	r.want(t, "3D refused", cmd(gpudev.CmdCtxDestroy), gpudev.RespErrUnspec)
}

func TestRAM(t *testing.T) {
	m := &gpudev.RAM{Base: 100, Bytes: make([]byte, 50)}
	for _, c := range []struct {
		addr, n uint64
		ok      bool
	}{
		{100, 50, true}, {149, 1, true}, {150, 0, true}, {99, 1, false}, {150, 1, false},
		{151, 0, false}, {120, 31, false}, {120, 1 << 63, false}, {1<<64 - 1, 2, false},
	} {
		b, err := m.Slice(c.addr, c.n)
		if (err == nil) != c.ok || (c.ok && uint64(len(b)) != c.n) {
			t.Errorf("Slice(%d, %d): %d bytes, %v", c.addr, c.n, len(b), err)
		}
	}
	b, _ := m.Slice(110, 4)
	if cap(b) != 4 {
		t.Error("a view can grow past its range")
	}
}
