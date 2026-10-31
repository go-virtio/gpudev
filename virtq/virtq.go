// Package virtq is the device side of a virtio split virtqueue: it takes
// descriptor chains the driver made available and returns them used.
//
// Everything in the rings is written by the guest and checked here: the
// available index may not run ahead of the queue, a chain may not loop or
// be longer than the queue, every buffer must lie in guest memory,
// device-writable buffers must follow the readable ones, and indirect
// descriptors (not negotiated) are refused.
//
// Reference: the virtio spec, split-ring.tex — struct virtq_desc
// (:185-196), the descriptor rules (:146-151, :218-223, :266-278),
// struct virtq_avail (:298), struct virtq_used (:393-409) and the used
// length rules (:456-462).
package virtq

import (
	"encoding/binary"
	"errors"

	"github.com/go-virtio/gpudev"
)

var le = binary.LittleEndian

// Descriptor flags (split-ring.tex:192-196).
const (
	DescNext     uint16 = 1
	DescWrite    uint16 = 2
	DescIndirect uint16 = 4
)

// MaxSize is the largest queue size (virtio: 32768).
const MaxSize = 32768

// ErrRing is wrapped by every error a malformed ring causes.
var ErrRing = errors.New("virtq: malformed ring")

func bad(msg string) error { return errors.Join(ErrRing, errors.New("virtq: "+msg)) }

// Queue is one split virtqueue as the device sees it.
type Queue struct {
	mem               gpudev.Memory
	size              uint16
	desc, avail, used []byte
	last              uint16 // next available-ring entry to consume
}

// New maps a queue of size entries whose rings the driver placed at the
// guest-physical addresses desc, avail and used.
func New(mem gpudev.Memory, size uint16, desc, avail, used uint64) (*Queue, error) {
	if size == 0 || size > MaxSize {
		return nil, bad("queue size")
	}
	n := uint64(size)
	d, err := mem.Slice(desc, 16*n)
	if err != nil {
		return nil, err
	}
	a, err := mem.Slice(avail, 6+2*n)
	if err != nil {
		return nil, err
	}
	u, err := mem.Slice(used, 6+8*n)
	if err != nil {
		return nil, err
	}
	return &Queue{mem: mem, size: size, desc: d, avail: a, used: u}, nil
}

// Chain is one descriptor chain: its head index and its buffers, which
// are views of guest memory.
type Chain struct {
	Head     uint16
	Readable [][]byte
	Writable [][]byte
	ReadLen  uint64
	WriteLen uint64
}

// Pop returns the next available chain, or ok false when there is none.
func (q *Queue) Pop() (c Chain, ok bool, err error) {
	idx := le.Uint16(q.avail[2:])
	if idx == q.last {
		return Chain{}, false, nil
	}
	if idx-q.last > q.size {
		return Chain{}, false, bad("available index ran ahead of the queue")
	}
	head := le.Uint16(q.avail[4+2*(q.last%q.size):])
	c, err = q.walk(head)
	if err != nil {
		return Chain{}, false, err
	}
	q.last++
	return c, true, nil
}

func (q *Queue) walk(head uint16) (Chain, error) {
	c := Chain{Head: head}
	i := head
	for n := uint16(0); ; n++ {
		if i >= q.size {
			return Chain{}, bad("descriptor index out of range")
		}
		if n == q.size {
			return Chain{}, bad("descriptor chain loops or exceeds the queue size")
		}
		d := q.desc[16*int(i):]
		addr, length := le.Uint64(d[0:]), uint64(le.Uint32(d[8:]))
		flags, next := le.Uint16(d[12:]), le.Uint16(d[14:])
		if flags&DescIndirect != 0 {
			return Chain{}, bad("indirect descriptor (not negotiated)")
		}
		b, err := q.mem.Slice(addr, length)
		if err != nil {
			return Chain{}, err
		}
		if flags&DescWrite != 0 {
			c.Writable = append(c.Writable, b)
			c.WriteLen += length
		} else {
			if len(c.Writable) > 0 {
				return Chain{}, bad("readable descriptor after a writable one")
			}
			c.Readable = append(c.Readable, b)
			c.ReadLen += length
		}
		if c.ReadLen+c.WriteLen > 1<<32 {
			return Chain{}, bad("chain longer than 4 GiB")
		}
		if flags&DescNext == 0 {
			return c, nil
		}
		i = next
	}
}

// Push returns chain head to the driver with written bytes written
// (split-ring.tex:456: the length is set before the index moves).
func (q *Queue) Push(head uint16, written uint32) {
	idx := le.Uint16(q.used[2:])
	e := q.used[4+8*int(idx%q.size):]
	le.PutUint32(e[0:], uint32(head))
	le.PutUint32(e[4:], written)
	le.PutUint16(q.used[2:], idx+1)
}

// Gather copies the readable buffers of c into one host buffer (at most
// max bytes; ok false if the chain is longer). Copying is what makes the
// request immune to the guest rewriting it while it is checked.
func Gather(c Chain, max uint64) (req []byte, ok bool) {
	if c.ReadLen > max {
		return nil, false
	}
	req = make([]byte, 0, c.ReadLen)
	for _, b := range c.Readable {
		req = append(req, b...)
	}
	return req, true
}

// Scatter copies p into the writable buffers of c and returns how many
// bytes fitted.
func Scatter(c Chain, p []byte) uint32 {
	var n uint32
	for _, b := range c.Writable {
		k := copy(b, p)
		p = p[k:]
		n += uint32(k)
	}
	return n
}
