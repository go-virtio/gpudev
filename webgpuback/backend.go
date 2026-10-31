//go:build js

// Package webgpuback is the browser backend of gpudev, on go-gfx/webgpu.
//
// 2D (option 2 of go-virtio/gpu docs/WEBGPU.md): each resource is a GPU
// texture; TRANSFER_TO_HOST_2D is one Queue.WriteTexture straight from
// the rows the device hands over (a view of guest RAM when contiguous);
// RESOURCE_FLUSH copies the scanout's rectangle of the resource into the
// scanout's canvas texture on the GPU and presents it. A canvas texture
// starts cleared every frame, so a flush copies the whole scanout, not
// only the damaged rectangle: one GPU copy, no CPU work.
//
// webgpu-batch (option 4): contexts create objects from go-gfx/webgpu
// wire descriptors (DecodeDescriptor) and submit streams with
// Device.Replay, naming only their own objects; a context's backend ids
// are the objects' replay ids (ReplayID), so the device's translation of
// guest ids lands directly on what replay.js knows.
package webgpuback

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"syscall/js"

	"github.com/go-gfx/webgpu"
	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpudev"
)

// Config is what a Backend presents to.
type Config struct {
	Device *webgpu.Device
	// Surfaces are the scanouts' canvases, configured with this device,
	// with COPY_DST in their usage and the format of the resources they
	// will show (bgra8unorm for go-virtio/gpu's framebuffer).
	Surfaces map[uint32]*webgpu.Surface
	// BeforePresent, if set, runs after a flush's copy with the canvas
	// texture, before the frame is presented (tests read it back here).
	BeforePresent func(scanout uint32, canvas *webgpu.Texture) error
	// Present receives each presented frame, an ImageBitmap; nil drops it.
	Present func(scanout uint32, frame js.Value)
}

// Backend implements gpudev.Backend.
type Backend struct {
	cfg   Config
	mu    sync.Mutex
	res   map[uint32]*webgpu.Texture
	shown map[uint32]gpudev.Rect // scanout -> rectangle of the resource shown
	ctxs  map[uint32]*Context
}

// New returns a backend presenting through cfg.
func New(cfg Config) *Backend {
	return &Backend{cfg: cfg, res: map[uint32]*webgpu.Texture{}, shown: map[uint32]gpudev.Rect{}, ctxs: map[uint32]*Context{}}
}

// ErrFormat: a 2D format WebGPU has no texture format for (the A8/X8-first
// orders need a swizzle pass, not implemented).
var ErrFormat = errors.New("webgpuback: format needs a swizzle")

func textureFormat(f gpudev.Format) (webgpu.TextureFormat, error) {
	switch f {
	case gpudev.FormatB8G8R8A8, gpudev.FormatB8G8R8X8:
		return webgpu.TextureFormatBGRA8Unorm, nil
	case gpudev.FormatR8G8B8A8, gpudev.FormatR8G8B8X8:
		return webgpu.TextureFormatRGBA8Unorm, nil
	}
	return "", ErrFormat
}

// CreateResource implements gpudev.Backend.
func (b *Backend) CreateResource(id uint32, f gpudev.Format, w, h uint32) error {
	tf, err := textureFormat(f)
	if err != nil {
		return err
	}
	t, err := b.cfg.Device.CreateTexture(webgpu.TextureDescriptor{
		Label: fmt.Sprintf("virtio-gpu resource %d", id), Size: webgpu.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
		Format: tf, Usage: webgpu.TextureUsageCopyDst | webgpu.TextureUsageCopySrc | webgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return err
	}
	b.mu.Lock()
	b.res[id] = t
	b.mu.Unlock()
	return nil
}

// DestroyResource implements gpudev.Backend.
func (b *Backend) DestroyResource(id uint32) {
	b.mu.Lock()
	t := b.res[id]
	delete(b.res, id)
	b.mu.Unlock()
	if t != nil {
		t.Destroy()
	}
}

// Upload implements gpudev.Backend: one writeTexture.
func (b *Backend) Upload(id uint32, r gpudev.Rect, src []byte, stride uint64) error {
	b.mu.Lock()
	t := b.res[id]
	b.mu.Unlock()
	if t == nil || stride > 1<<32-1 {
		return fmt.Errorf("webgpuback: upload to resource %d", id)
	}
	return b.cfg.Device.Queue().WriteTexture(
		webgpu.TexelCopyTextureInfo{Texture: t, Origin: webgpu.Origin3D{X: r.X, Y: r.Y}},
		src, webgpu.TexelCopyBufferLayout{BytesPerRow: uint32(stride)},
		webgpu.Extent3D{Width: r.Width, Height: r.Height, DepthOrArrayLayers: 1})
}

// SetScanout implements gpudev.Backend.
func (b *Backend) SetScanout(scanout, id uint32, r gpudev.Rect) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if id == 0 {
		delete(b.shown, scanout)
		return nil
	}
	if b.cfg.Surfaces[scanout] == nil {
		return fmt.Errorf("webgpuback: no canvas for scanout %d", scanout)
	}
	b.shown[scanout] = r
	return nil
}

// Flush implements gpudev.Backend: the whole shown rectangle is copied
// into the canvas texture and presented.
func (b *Backend) Flush(scanout, id uint32, _ gpudev.Rect) error {
	b.mu.Lock()
	t, sh, s := b.res[id], b.shown[scanout], b.cfg.Surfaces[scanout]
	b.mu.Unlock()
	if t == nil || s == nil {
		return fmt.Errorf("webgpuback: flush of resource %d on scanout %d", id, scanout)
	}
	cur, err := s.CurrentTexture()
	if err != nil {
		return err
	}
	w, h := s.Size()
	size := webgpu.Extent3D{Width: min(sh.Width, uint32(w)), Height: min(sh.Height, uint32(h)), DepthOrArrayLayers: 1}
	enc, err := b.cfg.Device.CreateCommandEncoder("virtio-gpu flush")
	if err != nil {
		return err
	}
	if err := enc.CopyTextureToTexture(
		webgpu.TexelCopyTextureInfo{Texture: t, Origin: webgpu.Origin3D{X: sh.X, Y: sh.Y}},
		webgpu.TexelCopyTextureInfo{Texture: cur}, size); err != nil {
		return err
	}
	cb, err := enc.Finish()
	if err != nil {
		return err
	}
	if err := b.cfg.Device.Queue().Submit(cb); err != nil {
		return err
	}
	if b.cfg.BeforePresent != nil {
		if err := b.cfg.BeforePresent(scanout, cur); err != nil {
			return err
		}
	}
	frame, err := s.Present()
	if err != nil {
		return err
	}
	if b.cfg.Present != nil {
		b.cfg.Present(scanout, frame)
	} else {
		frame.Call("close")
	}
	return nil
}

// NewContext implements gpudev.Backend.
func (b *Backend) NewContext(id uint32) (gpudev.Context, error) {
	c := &Context{b: b, id: id, dev: b.cfg.Device, objs: map[uint32]any{}}
	b.mu.Lock()
	b.ctxs[id] = c
	b.mu.Unlock()
	return c, nil
}

// contexts returns the live contexts (for inspection by tests).
func (b *Backend) contexts() []*Context {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]*Context, 0, len(b.ctxs))
	for _, c := range b.ctxs {
		out = append(out, c)
	}
	return out
}

// Context is a webgpu-batch context on the device.
type Context struct {
	b    *Backend
	id   uint32
	dev  *webgpu.Device
	objs map[uint32]any // replay id -> handle
}

// ErrBind: BIND_RESOURCE windows (readback into guest memory) are not
// implemented by this backend yet.
var ErrBind = errors.New("webgpuback: BIND_RESOURCE is not implemented")

func (c *Context) resolve(id uint32) (any, error) {
	if o, ok := c.objs[id]; ok {
		return o, nil
	}
	return nil, fmt.Errorf("webgpuback: object %d", id)
}

// Create implements gpudev.Context.
func (c *Context) Create(kind uint32, desc []byte) (uint32, error) {
	d, err := webgpu.DecodeDescriptor(wire.Kind(kind), desc, c.resolve)
	if err != nil {
		return 0, err
	}
	var o any
	switch d := d.(type) {
	case webgpu.BufferDescriptor:
		o, err = c.dev.CreateBuffer(d)
	case webgpu.TextureDescriptor:
		o, err = c.dev.CreateTexture(d)
	case webgpu.TextureViewCreation:
		o, err = d.Texture.CreateView(d.Descriptor)
	case webgpu.SamplerDescriptor:
		o, err = c.dev.CreateSampler(&d)
	case webgpu.ShaderModuleDescriptor:
		o, err = c.dev.CreateShaderModule(d)
	case webgpu.BindGroupLayoutDescriptor:
		o, err = c.dev.CreateBindGroupLayout(d)
	case webgpu.PipelineLayoutDescriptor:
		o, err = c.dev.CreatePipelineLayout(d)
	case webgpu.BindGroupDescriptor:
		o, err = c.dev.CreateBindGroup(d)
	case webgpu.RenderPipelineDescriptor:
		o, err = c.dev.CreateRenderPipeline(d)
	case webgpu.ComputePipelineDescriptor:
		o, err = c.dev.CreateComputePipeline(d)
	}
	if err != nil {
		return 0, err
	}
	id, err := webgpu.ReplayID(o)
	if err != nil {
		return 0, err
	}
	c.objs[id] = o
	return id, nil
}

// Destroy implements gpudev.Context.
func (c *Context) Destroy(id uint32) {
	switch o := c.objs[id].(type) {
	case *webgpu.Buffer:
		o.Destroy()
	case *webgpu.Texture:
		o.Destroy()
	case *webgpu.RenderPipeline:
		o.Release()
	case *webgpu.ComputePipeline:
		o.Release()
	}
	delete(c.objs, id)
}

// Replay implements gpudev.Context: the stream may name this context's
// objects only, and Replay returns once the GPU has executed it.
func (c *Context) Replay(words []uint32, data []byte) error {
	refs := make([]any, 0, len(c.objs))
	for _, o := range c.objs {
		switch o.(type) {
		case *webgpu.Buffer, *webgpu.TextureView, *webgpu.BindGroup, *webgpu.RenderPipeline, *webgpu.ComputePipeline:
			refs = append(refs, o)
		}
	}
	seq, err := c.dev.Replay(words, data, refs...)
	if err != nil {
		return err
	}
	return c.dev.WaitSubmitted(context.Background(), seq)
}

// Bind implements gpudev.Context.
func (c *Context) Bind(uint32, gpudev.Window) error { return ErrBind }

// Close implements gpudev.Context.
func (c *Context) Close() {
	for id := range c.objs {
		c.Destroy(id)
	}
	c.b.mu.Lock()
	delete(c.b.ctxs, c.id)
	c.b.mu.Unlock()
}
