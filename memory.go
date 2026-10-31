package gpudev

import "errors"

// Memory is the guest's physical memory as the device sees it. Every
// guest-supplied address goes through Slice, which bounds it.
type Memory interface {
	// Slice returns a view of n bytes at guest-physical addr, or an
	// error if any of them lies outside guest memory.
	Slice(addr, n uint64) ([]byte, error)
}

// ErrOutOfRange: an address range outside guest memory.
var ErrOutOfRange = errors.New("gpudev: guest address range outside memory")

// RAM is guest memory as one byte slice mapped at guest-physical Base.
type RAM struct {
	Base  uint64
	Bytes []byte
}

// Slice implements Memory with 64-bit bounds checks: no addr or n can
// wrap around or reach past the slice, on 32-bit hosts too.
func (m *RAM) Slice(addr, n uint64) ([]byte, error) {
	size := uint64(len(m.Bytes))
	if addr < m.Base || addr-m.Base > size || n > size-(addr-m.Base) {
		return nil, ErrOutOfRange
	}
	off := addr - m.Base
	return m.Bytes[off : off+n : off+n], nil
}
