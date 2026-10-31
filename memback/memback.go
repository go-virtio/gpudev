// Package memback is an in-memory gpudev.Backend: resources are byte
// images, scanouts are framebuffers a flush composes into, and
// webgpu-batch contexts keep their objects in a table and check every
// replayed stream with go-gfx/webgpu's own reader (wire.Decode). It is
// the backend of the tests, the fuzz targets and the in-process
// end-to-end runs; it renders nothing.
package memback

import (
	"errors"
	"sync"

	"github.com/go-virtio/gpudev"
)

// Image is a resource or scanout: Width×Height pixels, 4 bytes each, rows
// packed (stride Width*4), in the resource's byte order.
type Image struct {
	Format        gpudev.Format
	Width, Height uint32
	Pix           []byte
}

// Backend is the in-memory backend. Its fields are for inspection; lock
// it (it is a sync.Locker) while reading them during device activity.
type Backend struct {
	sync.Mutex
	Resources map[uint32]*Image
	// Scanouts holds what each scanout shows: the image a flush last
	// composed (nil until the first flush of an enabled scanout).
	Scanouts map[uint32]*Image
	Shown    map[uint32]Shown // scanout -> resource and rect
	Flushes  int
	Contexts map[uint32]*Context
	// Fail, when set, makes every call that can fail return it.
	Fail error
}

// Shown is what SET_SCANOUT bound to a scanout.
type Shown struct {
	Resource uint32
	Rect     gpudev.Rect
}

// New returns an empty backend.
func New() *Backend {
	return &Backend{Resources: map[uint32]*Image{}, Scanouts: map[uint32]*Image{},
		Shown: map[uint32]Shown{}, Contexts: map[uint32]*Context{}}
}

// ErrUnknown: a call naming something the backend does not hold. The
// Device never makes one; tests that drive the backend directly can.
var ErrUnknown = errors.New("memback: unknown resource or object")

// CreateResource implements gpudev.Backend.
func (b *Backend) CreateResource(id uint32, f gpudev.Format, w, h uint32) error {
	b.Lock()
	defer b.Unlock()
	if b.Fail != nil {
		return b.Fail
	}
	b.Resources[id] = &Image{Format: f, Width: w, Height: h, Pix: make([]byte, uint64(w)*uint64(h)*gpudev.BytesPerPixel)}
	return nil
}

// DestroyResource implements gpudev.Backend.
func (b *Backend) DestroyResource(id uint32) {
	b.Lock()
	defer b.Unlock()
	delete(b.Resources, id)
}

// Upload implements gpudev.Backend.
func (b *Backend) Upload(id uint32, r gpudev.Rect, src []byte, stride uint64) error {
	b.Lock()
	defer b.Unlock()
	if b.Fail != nil {
		return b.Fail
	}
	img := b.Resources[id]
	if img == nil {
		return ErrUnknown
	}
	row := uint64(r.Width) * gpudev.BytesPerPixel
	dst := uint64(img.Width) * gpudev.BytesPerPixel
	for y := uint64(0); y < uint64(r.Height); y++ {
		o := (uint64(r.Y)+y)*dst + uint64(r.X)*gpudev.BytesPerPixel
		copy(img.Pix[o:o+row], src[y*stride:y*stride+row])
	}
	return nil
}

// SetScanout implements gpudev.Backend.
func (b *Backend) SetScanout(scanout, id uint32, r gpudev.Rect) error {
	b.Lock()
	defer b.Unlock()
	if b.Fail != nil {
		return b.Fail
	}
	if id == 0 {
		delete(b.Shown, scanout)
		delete(b.Scanouts, scanout)
		return nil
	}
	b.Shown[scanout] = Shown{id, r}
	return nil
}

// Flush implements gpudev.Backend: it copies r of the resource into the
// scanout image, at r's position relative to the scanout's rect.
func (b *Backend) Flush(scanout, id uint32, r gpudev.Rect) error {
	b.Lock()
	defer b.Unlock()
	if b.Fail != nil {
		return b.Fail
	}
	sh, ok := b.Shown[scanout]
	img := b.Resources[id]
	if !ok || img == nil || sh.Resource != id {
		return ErrUnknown
	}
	out := b.Scanouts[scanout]
	if out == nil {
		out = &Image{Format: img.Format, Width: sh.Rect.Width, Height: sh.Rect.Height,
			Pix: make([]byte, uint64(sh.Rect.Width)*uint64(sh.Rect.Height)*gpudev.BytesPerPixel)}
		b.Scanouts[scanout] = out
	}
	row := uint64(r.Width) * gpudev.BytesPerPixel
	for y := uint64(0); y < uint64(r.Height); y++ {
		so := ((uint64(r.Y)+y)*uint64(img.Width) + uint64(r.X)) * gpudev.BytesPerPixel
		do := ((uint64(r.Y-sh.Rect.Y)+y)*uint64(out.Width) + uint64(r.X-sh.Rect.X)) * gpudev.BytesPerPixel
		copy(out.Pix[do:do+row], img.Pix[so:so+row])
	}
	b.Flushes++
	return nil
}
