package memback

import (
	"fmt"

	"github.com/go-gfx/webgpu/wire"
	"github.com/go-virtio/gpudev"
)

// Context is an in-memory webgpu-batch context. Host ids start at
// FirstID, far from the small ids guests use, so a stream the Device did
// not translate fails wire.Decode here.
type Context struct {
	b       *Backend
	ID      uint32
	Objects map[uint32]Object
	Replays [][]wire.Cmd
	Binds   map[uint32]gpudev.Window
	Closed  bool
	next    uint32
}

// FirstID is the first host id a Context hands out.
const FirstID = 1 << 20

// Object is what a context holds for one host id.
type Object struct {
	Kind uint32
	Desc []byte
}

// NewContext implements gpudev.Backend.
func (b *Backend) NewContext(ctx uint32) (gpudev.Context, error) {
	b.Lock()
	defer b.Unlock()
	if b.Fail != nil {
		return nil, b.Fail
	}
	c := &Context{b: b, ID: ctx, Objects: map[uint32]Object{}, Binds: map[uint32]gpudev.Window{}, next: FirstID}
	b.Contexts[ctx] = c
	return c, nil
}

// Create implements gpudev.Context; the descriptor is kept as given.
func (c *Context) Create(kind uint32, desc []byte) (uint32, error) {
	c.b.Lock()
	defer c.b.Unlock()
	if c.b.Fail != nil {
		return 0, c.b.Fail
	}
	id := c.next
	c.next++
	c.Objects[id] = Object{Kind: kind, Desc: append([]byte(nil), desc...)}
	return id, nil
}

// Destroy implements gpudev.Context.
func (c *Context) Destroy(id uint32) {
	c.b.Lock()
	defer c.b.Unlock()
	delete(c.Objects, id)
	delete(c.Binds, id)
}

// Replay implements gpudev.Context: the stream must decode against this
// context's objects, with go-gfx/webgpu's checks.
func (c *Context) Replay(words []uint32, data []byte) error {
	c.b.Lock()
	defer c.b.Unlock()
	if c.b.Fail != nil {
		return c.b.Fail
	}
	cmds, err := wire.Decode(words, uint32(len(data)), func(id uint32) bool { _, ok := c.Objects[id]; return ok })
	if err != nil {
		return fmt.Errorf("memback: %w", err)
	}
	c.Replays = append(c.Replays, cmds)
	return nil
}

// Bind implements gpudev.Context.
func (c *Context) Bind(id uint32, w gpudev.Window) error {
	c.b.Lock()
	defer c.b.Unlock()
	if _, ok := c.Objects[id]; !ok {
		return ErrUnknown
	}
	c.Binds[id] = w
	return nil
}

// Close implements gpudev.Context.
func (c *Context) Close() {
	c.b.Lock()
	defer c.b.Unlock()
	c.Closed = true
	delete(c.b.Contexts, c.ID)
}
