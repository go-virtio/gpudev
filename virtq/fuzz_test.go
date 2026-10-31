package virtq

import (
	"testing"

	"github.com/go-virtio/gpudev"
)

// FuzzPop: whatever the guest wrote into the descriptor table and the
// available ring, Pop never panics, never loops, and every chain it
// returns has buffers inside guest memory whose lengths add up.
func FuzzPop(f *testing.F) {
	seed := make([]byte, 0x300)
	le.PutUint64(seed[0:], data)
	le.PutUint32(seed[8:], 8)
	le.PutUint16(seed[0x100+2:], 1)
	f.Add(seed)
	f.Fuzz(func(t *testing.T, rings []byte) {
		m := &gpudev.RAM{Base: base, Bytes: make([]byte, 0x4000)}
		copy(m.Bytes, rings)
		q, err := New(m, size, desc, avail, used)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 2*size; i++ {
			c, ok, err := q.Pop()
			if err != nil || !ok {
				return
			}
			var r, w uint64
			for _, b := range c.Readable {
				r += uint64(len(b))
			}
			for _, b := range c.Writable {
				w += uint64(len(b))
			}
			if r != c.ReadLen || w != c.WriteLen || r+w > uint64(len(m.Bytes))*size {
				t.Fatalf("chain lengths %d/%d vs %d/%d", r, w, c.ReadLen, c.WriteLen)
			}
			q.Push(c.Head, 0)
		}
	})
}
