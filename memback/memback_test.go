package memback

import (
	"bytes"
	"errors"
	"testing"

	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpudev"
)

var errDown = errors.New("down")

func TestResources(t *testing.T) {
	b := New()
	if err := b.CreateResource(1, gpudev.FormatB8G8R8A8, 4, 2); err != nil {
		t.Fatal(err)
	}
	src := []byte{1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16}
	// 2×2 at (1,0), source stride 10.
	if err := b.Upload(1, gpudev.Rect{X: 1, Width: 2, Height: 2}, src, 10); err != nil {
		t.Fatal(err)
	}
	want := []byte{0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0,
		0, 0, 0, 0, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0}
	if !bytes.Equal(b.Resources[1].Pix, want) {
		t.Fatalf("pixels %v", b.Resources[1].Pix)
	}
	if err := b.Upload(2, gpudev.Rect{}, nil, 0); !errors.Is(err, ErrUnknown) {
		t.Error("upload to an unknown resource")
	}
	if err := b.Flush(0, 1, gpudev.Rect{}); !errors.Is(err, ErrUnknown) {
		t.Error("flush of an unshown scanout")
	}
	// Scanout shows columns 1..3; flushing column 2..3 lands at 1..2.
	if err := b.SetScanout(0, 1, gpudev.Rect{X: 1, Width: 3, Height: 2}); err != nil {
		t.Fatal(err)
	}
	if err := b.Flush(0, 1, gpudev.Rect{X: 2, Width: 2, Height: 1}); err != nil {
		t.Fatal(err)
	}
	if got := b.Scanouts[0].Pix[4:12]; !bytes.Equal(got, []byte{5, 6, 7, 8, 0, 0, 0, 0}) {
		t.Errorf("scanout %v", b.Scanouts[0].Pix)
	}
	if err := b.Flush(0, 2, gpudev.Rect{}); !errors.Is(err, ErrUnknown) {
		t.Error("flush of a resource the scanout does not show")
	}
	_ = b.SetScanout(0, 0, gpudev.Rect{})
	if len(b.Shown)+len(b.Scanouts) != 0 {
		t.Error("disable kept state")
	}
	b.DestroyResource(1)
	b.Fail = errDown
	for name, err := range map[string]error{
		"create":  b.CreateResource(3, 1, 1, 1),
		"upload":  b.Upload(3, gpudev.Rect{}, nil, 0),
		"scanout": b.SetScanout(0, 3, gpudev.Rect{}),
		"flush":   b.Flush(0, 3, gpudev.Rect{}),
	} {
		if err != errDown {
			t.Errorf("%s: %v", name, err)
		}
	}
	if _, err := b.NewContext(1); err != errDown {
		t.Errorf("context: %v", err)
	}
}

type window struct{}

func (window) Len() uint64                  { return 0 }
func (window) ReadAt([]byte, uint64) error  { return nil }
func (window) WriteAt([]byte, uint64) error { return nil }

func TestContext(t *testing.T) {
	b := New()
	gc, err := b.NewContext(7)
	if err != nil {
		t.Fatal(err)
	}
	c := gc.(*Context)
	id, err := c.Create(1, []byte("desc"))
	if err != nil || id != FirstID || string(c.Objects[id].Desc) != "desc" {
		t.Fatalf("create %d %v", id, err)
	}
	ok := []uint32{uint32(wire.WriteBuffer), id, 0, 0, 0, 4, uint32(wire.Submit)}
	if err := c.Replay(ok, make([]byte, 4)); err != nil || len(c.Replays) != 1 {
		t.Fatalf("replay: %v", err)
	}
	if err := c.Replay([]uint32{uint32(wire.WriteBuffer), 5, 0, 0, 0, 4}, make([]byte, 4)); err == nil {
		t.Error("a stream naming an unknown object replayed")
	}
	if err := c.Bind(id, window{}); err != nil || c.Binds[id] == nil {
		t.Error("bind")
	}
	if err := c.Bind(id+1, window{}); !errors.Is(err, ErrUnknown) {
		t.Error("bind of an unknown object")
	}
	c.Destroy(id)
	if len(c.Objects)+len(c.Binds) != 0 {
		t.Error("destroy kept state")
	}
	b.Fail = errDown
	if _, err := c.Create(1, nil); err != errDown {
		t.Error("create while failing")
	}
	if err := c.Replay(ok, nil); err != errDown {
		t.Error("replay while failing")
	}
	if err := c.Bind(id, window{}); err != errDown {
		t.Error("bind while failing")
	}
	c.Close()
	if !c.Closed || b.Contexts[7] != nil {
		t.Error("close")
	}
}
