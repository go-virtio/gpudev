package virtq

import (
	"bytes"
	"errors"
	"testing"

	"github.com/go-virtio/gpudev"
)

const (
	base  = 0x10000
	size  = 4
	desc  = base
	avail = base + 0x100
	used  = base + 0x200
	data  = base + 0x1000
)

func ring(t *testing.T) (*gpudev.RAM, *Queue) {
	t.Helper()
	m := &gpudev.RAM{Base: base, Bytes: make([]byte, 0x4000)}
	q, err := New(m, size, desc, avail, used)
	if err != nil {
		t.Fatal(err)
	}
	return m, q
}

func setDesc(m *gpudev.RAM, i uint16, addr uint64, n uint32, flags, next uint16) {
	d := m.Bytes[desc-base+16*int(i):]
	le.PutUint64(d[0:], addr)
	le.PutUint32(d[8:], n)
	le.PutUint16(d[12:], flags)
	le.PutUint16(d[14:], next)
}

// offer puts head in the available ring and bumps its index.
func offer(m *gpudev.RAM, head uint16) {
	a := m.Bytes[avail-base:]
	idx := le.Uint16(a[2:])
	le.PutUint16(a[4+2*int(idx%size):], head)
	le.PutUint16(a[2:], idx+1)
}

func TestNew(t *testing.T) {
	m := &gpudev.RAM{Base: base, Bytes: make([]byte, 0x400)}
	for name, c := range map[string][4]uint64{
		"size 0":    {0, desc, avail, used},
		"size big":  {MaxSize + 1, desc, avail, used},
		"desc out":  {size, base + 0x3F8, avail, used},
		"avail out": {size, desc, base + 0x3FA, used},
		"used out":  {size, desc, avail, base + 0x3F0},
	} {
		if _, err := New(m, uint16(c[0]), c[1], c[2], c[3]); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
	if _, err := New(m, MaxSize, desc, avail, used); err == nil {
		t.Error("rings past memory accepted")
	}
}

func TestPopPush(t *testing.T) {
	m, q := ring(t)
	if _, ok, err := q.Pop(); ok || err != nil {
		t.Fatal("empty queue popped")
	}
	copy(m.Bytes[data-base:], "hello, world")
	setDesc(m, 2, data, 5, DescNext, 0)
	setDesc(m, 0, data+5, 7, DescNext, 1)
	setDesc(m, 1, data+0x100, 8, DescWrite, 0)
	offer(m, 2)
	c, ok, err := q.Pop()
	if err != nil || !ok || c.Head != 2 || c.ReadLen != 12 || c.WriteLen != 8 || len(c.Readable) != 2 || len(c.Writable) != 1 {
		t.Fatalf("chain %+v %v %v", c, ok, err)
	}
	req, ok := Gather(c, 64)
	if !ok || string(req) != "hello, world" {
		t.Fatalf("gathered %q", req)
	}
	if _, ok := Gather(c, 11); ok {
		t.Error("gathered past the limit")
	}
	if n := Scatter(c, []byte("0123456789")); n != 8 || !bytes.Equal(m.Bytes[data-base+0x100:][:8], []byte("01234567")) {
		t.Errorf("scattered %d", n)
	}
	q.Push(c.Head, 8)
	u := m.Bytes[used-base:]
	if le.Uint16(u[2:]) != 1 || le.Uint32(u[4:]) != 2 || le.Uint32(u[8:]) != 8 {
		t.Errorf("used ring %x", u[:12])
	}
	// Wrap-around of both indices over the 16-bit space.
	le.PutUint16(m.Bytes[avail-base+2:], 0xFFFF)
	q.last = 0xFFFF
	le.PutUint16(m.Bytes[used-base+2:], 0xFFFF)
	setDesc(m, 3, data, 1, 0, 0)
	offer(m, 3)
	if c, ok, err := q.Pop(); !ok || err != nil || c.Head != 3 {
		t.Fatalf("wrapped pop %+v %v", c, err)
	}
	q.Push(3, 0)
	if le.Uint16(u[2:]) != 0 {
		t.Error("used index did not wrap")
	}
}

func TestMalformed(t *testing.T) {
	cases := map[string]func(m *gpudev.RAM){
		"index ahead": func(m *gpudev.RAM) { le.PutUint16(m.Bytes[avail-base+2:], size+1) },
		"head out of range": func(m *gpudev.RAM) {
			offer(m, size)
		},
		"next out of range": func(m *gpudev.RAM) {
			setDesc(m, 0, data, 1, DescNext, size)
			offer(m, 0)
		},
		"loop": func(m *gpudev.RAM) {
			setDesc(m, 0, data, 1, DescNext, 1)
			setDesc(m, 1, data, 1, DescNext, 0)
			offer(m, 0)
		},
		"indirect": func(m *gpudev.RAM) {
			setDesc(m, 0, data, 16, DescIndirect, 0)
			offer(m, 0)
		},
		"buffer outside": func(m *gpudev.RAM) {
			setDesc(m, 0, base+0x3FFF, 2, 0, 0)
			offer(m, 0)
		},
		"readable after writable": func(m *gpudev.RAM) {
			setDesc(m, 0, data, 1, DescWrite|DescNext, 1)
			setDesc(m, 1, data, 1, 0, 0)
			offer(m, 0)
		},
	}
	for name, set := range cases {
		m, q := ring(t)
		set(m)
		if _, _, err := q.Pop(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	// A chain over 4 GiB: lengths are checked before memory, so use a
	// memory that maps everything.
	q, _ := New(bigMem{}, size, 0, 0x100, 0x200)
	if _, _, err := q.Pop(); !errors.Is(err, ErrRing) {
		t.Errorf("4 GiB chain: %v", err)
	}
}

// bigMem maps every address to one shared page, except that the rings
// read a chain of three 2 GiB buffers.
type bigMem struct{}

var bigRing = func() []byte {
	b := make([]byte, 0x300)
	for i := 0; i < 3; i++ {
		d := b[16*i:]
		le.PutUint32(d[8:], 1<<31)
		le.PutUint16(d[12:], DescNext)
		le.PutUint16(d[14:], uint16(i+1))
	}
	le.PutUint16(b[0x100+2:], 1) // avail idx 1, ring[0] = 0
	return b
}()

func (bigMem) Slice(addr, n uint64) ([]byte, error) {
	if addr < 0x300 && n <= 0x300-addr {
		return bigRing[addr : addr+n], nil
	}
	return nil, nil // a buffer: never touched by Pop
}
