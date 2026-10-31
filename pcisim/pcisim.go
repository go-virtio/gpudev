// Package pcisim puts a gpudev.Device behind an emulated virtio-pci
// modern transport, in process: it implements go-virtio/common.Transport,
// so the go-virtio/gpu guest driver (or any driver written against that
// interface) can bring the device up and drive it with no VM.
//
// Guest memory is a gpudev.RAM: AllocatePages hands out its pages, and
// the device reads the virtqueues and buffers from it by guest-physical
// address, as a hypervisor would. A doorbell write runs the queue
// synchronously, so the driver finds the used ring updated when its
// notify returns.
//
// Reference: the virtio spec, transport-pci.tex — struct virtio_pci_cap
// (:143-153), struct virtio_pci_common_cfg (:302-328), the notification
// address cap.offset + queue_notify_off * notify_off_multiplier (:551).
package pcisim

import (
	"encoding/binary"
	"errors"
	"sync"

	"github.com/go-virtio/common"
	"github.com/go-virtio/gpudev"
	"github.com/go-virtio/gpudev/virtq"
)

var le = binary.LittleEndian

// BAR0 layout.
const (
	commonOff  = 0x0000
	notifyOff  = 0x1000
	notifyLen  = 0x1000
	notifyMult = 4
	deviceOff  = 0x2000
	deviceLen  = 16 // struct virtio_gpu_config
)

// Queues: 0 controlq, 1 cursorq (virtio gpu device).
const (
	numQueues    = 2
	maxQueueSize = 64
)

// MaxRequest bounds a control request the transport gathers; longer
// chains are answered with an error without being copied.
const MaxRequest = 16 << 20

// maxResponse is larger than any response the device writes.
const maxResponse = 4096

// ErrOutOfMemory: AllocatePages ran out of guest RAM.
var ErrOutOfMemory = errors.New("pcisim: guest RAM exhausted")

type queue struct {
	size              uint16
	enable            uint16
	desc, avail, used uint64
	q                 *virtq.Queue
}

// Transport is an emulated virtio-pci GPU function.
type Transport struct {
	mu       sync.Mutex
	dev      *gpudev.Device
	ram      *gpudev.RAM
	cfg      [256]byte
	next     uint64 // next free byte of RAM for AllocatePages
	status   uint8
	devSel   uint32
	drvSel   uint32
	driver   uint64
	queueSel uint16
	queues   [numQueues]queue
	err      error
}

// New returns a transport for dev, whose guest memory is ram (the same
// RAM the Device was made with).
func New(dev *gpudev.Device, ram *gpudev.RAM) *Transport {
	t := &Transport{dev: dev, ram: ram}
	t.reset()
	c := t.cfg[:]
	le.PutUint16(c[0x00:], common.PCIVendorID)
	le.PutUint16(c[0x02:], common.PCIDeviceIDModernGPU)
	le.PutUint16(c[0x06:], common.PCIStatusCapabilityList)
	c[0x34] = 0x40
	putCap(c, 0x40, 0x50, 16, common.PCICapCommonCfg, commonOff, common.CommonCfgSize)
	putCap(c, 0x50, 0x68, 20, common.PCICapNotifyCfg, notifyOff, notifyLen)
	le.PutUint32(c[0x60:], notifyMult)
	putCap(c, 0x68, 0x00, 16, common.PCICapDeviceCfg, deviceOff, deviceLen)
	return t
}

func putCap(c []byte, at, next, length, typ uint8, off, n uint32) {
	c[at], c[at+1], c[at+2], c[at+3] = common.PCICapIDVendorSpecific, next, length, typ
	c[at+4] = 0 // BAR 0
	le.PutUint32(c[at+8:], off)
	le.PutUint32(c[at+12:], n)
}

func (t *Transport) reset() {
	t.status, t.driver, t.devSel, t.drvSel, t.queueSel = 0, 0, 0, 0, 0
	for i := range t.queues {
		t.queues[i] = queue{size: maxQueueSize}
	}
}

// Err is the first ring error the device met (it then set NEEDS_RESET
// and stopped serving queues), or nil.
func (t *Transport) Err() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.err
}

func (t *Transport) features() uint64 { return common.FeatureVersion1 | t.dev.Features() }

// ReadConfig8 reads PCI configuration space.
func (t *Transport) ReadConfig8(off uint8) (uint8, error) { return t.cfg[off], nil }

// ReadConfig16 reads PCI configuration space.
func (t *Transport) ReadConfig16(off uint8) (uint16, error) {
	if int(off)+2 > len(t.cfg) {
		return 0, errors.New("pcisim: config read past 256 bytes")
	}
	return le.Uint16(t.cfg[off:]), nil
}

// ReadConfig32 reads PCI configuration space.
func (t *Transport) ReadConfig32(off uint8) (uint32, error) {
	if int(off)+4 > len(t.cfg) {
		return 0, errors.New("pcisim: config read past 256 bytes")
	}
	return le.Uint32(t.cfg[off:]), nil
}

// AllocatePages hands out zeroed, page-aligned guest RAM.
func (t *Transport) AllocatePages(count int) (uint64, []byte, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	n := uint64(count) * uint64(common.PageSize)
	if count <= 0 || n > uint64(len(t.ram.Bytes))-t.next {
		return 0, nil, ErrOutOfMemory
	}
	mem := t.ram.Bytes[t.next : t.next+n : t.next+n]
	phys := t.ram.Base + t.next
	t.next += n
	return phys, mem, nil
}

// read handles a BAR0 read of width bytes.
func (t *Transport) read(bar uint8, off uint64, width int) uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	if bar != 0 {
		return 0
	}
	q := &t.queues[t.queueSel%numQueues]
	switch {
	case off == commonOff+common.CfgDeviceFeatureSelect:
		return uint64(t.devSel)
	case off == commonOff+common.CfgDeviceFeature:
		if t.devSel == 0 {
			return t.features() & 0xFFFFFFFF
		} else if t.devSel == 1 {
			return t.features() >> 32
		}
		return 0
	case off == commonOff+common.CfgDriverFeatureSelect:
		return uint64(t.drvSel)
	case off == commonOff+common.CfgNumQueues:
		return numQueues
	case off == commonOff+common.CfgDeviceStatus:
		return uint64(t.status)
	case off == commonOff+common.CfgQueueSelect:
		return uint64(t.queueSel)
	case off == commonOff+common.CfgQueueSize:
		return uint64(q.size)
	case off == commonOff+common.CfgQueueEnable:
		return uint64(q.enable)
	case off == commonOff+common.CfgQueueNotifyOff:
		return uint64(t.queueSel)
	case off == commonOff+common.CfgQueueDesc:
		return q.desc
	case off == commonOff+common.CfgQueueDriver:
		return q.avail
	case off == commonOff+common.CfgQueueDevice:
		return q.used
	case off >= deviceOff && off+uint64(width) <= deviceOff+deviceLen:
		var c [deviceLen]byte // events_read, events_clear: no events
		le.PutUint32(c[8:], t.dev.NumScanouts())
		le.PutUint32(c[12:], t.dev.NumCapsets())
		var v [8]byte
		copy(v[:], c[off-deviceOff:off-deviceOff+uint64(width)])
		return le.Uint64(v[:])
	}
	return 0
}

// write handles a BAR0 write of width bytes.
func (t *Transport) write(bar uint8, off uint64, v uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if bar != 0 {
		return
	}
	q := &t.queues[t.queueSel%numQueues]
	live := t.status&common.StatusDriverOK == 0 // queue registers are writable before DRIVER_OK
	switch {
	case off == commonOff+common.CfgDeviceFeatureSelect:
		t.devSel = uint32(v)
	case off == commonOff+common.CfgDriverFeatureSelect:
		t.drvSel = uint32(v)
	case off == commonOff+common.CfgDriverFeature:
		if t.drvSel == 0 {
			t.driver = t.driver&^0xFFFFFFFF | v&0xFFFFFFFF
		} else if t.drvSel == 1 {
			t.driver = t.driver&0xFFFFFFFF | (v&0xFFFFFFFF)<<32
		}
	case off == commonOff+common.CfgDeviceStatus:
		t.setStatus(uint8(v))
	case off == commonOff+common.CfgQueueSelect:
		t.queueSel = uint16(v)
	case off == commonOff+common.CfgQueueSize && live:
		if s := uint16(v); s > 0 && s <= maxQueueSize {
			q.size = s
		}
	case off == commonOff+common.CfgQueueDesc && live:
		q.desc = v
	case off == commonOff+common.CfgQueueDriver && live:
		q.avail = v
	case off == commonOff+common.CfgQueueDevice && live:
		q.used = v
	case off == commonOff+common.CfgQueueEnable && live:
		q.enable = uint16(v & 1)
		q.q = nil
		if q.enable == 1 {
			vq, err := virtq.New(t.ram, q.size, q.desc, q.avail, q.used)
			if err != nil {
				t.fail(err)
				return
			}
			q.q = vq
		}
	case off >= notifyOff && off < notifyOff+notifyLen:
		t.notify(uint16(v))
	}
}

// setStatus: writing 0 resets the device; FEATURES_OK latches only for a
// subset of the offered features that includes VERSION_1.
func (t *Transport) setStatus(s uint8) {
	if s == 0 {
		t.reset()
		return
	}
	if s&common.StatusFeaturesOK != 0 &&
		(t.driver&^t.features() != 0 || t.driver&common.FeatureVersion1 == 0) {
		s &^= common.StatusFeaturesOK
	}
	t.status = s | t.status&common.StatusNeedsReset
}

func (t *Transport) fail(err error) {
	if t.err == nil {
		t.err = err
	}
	t.status |= common.StatusNeedsReset
}

// notify runs queue idx: every available chain is executed and returned.
func (t *Transport) notify(idx uint16) {
	if idx >= numQueues || t.status&common.StatusDriverOK == 0 || t.status&common.StatusNeedsReset != 0 {
		return
	}
	q := t.queues[idx].q
	if q == nil {
		return
	}
	for {
		c, ok, err := q.Pop()
		if err != nil {
			t.fail(err)
			return
		}
		if !ok {
			return
		}
		var n uint32
		if idx == 0 {
			n = t.control(c)
		}
		// cursorq commands carry no response (description.tex: the device
		// only returns the buffer).
		q.Push(c.Head, n)
	}
}

// control executes one controlq chain and returns the bytes written.
func (t *Transport) control(c virtq.Chain) uint32 {
	resp := make([]byte, min(c.WriteLen, maxResponse))
	req, ok := virtq.Gather(c, MaxRequest)
	var n int
	if ok {
		// The device lock is separate: HandleControl does not call back.
		n = t.dev.HandleControl(req, resp)
	} else if len(resp) >= gpudev.HdrSize {
		clear(resp[:gpudev.HdrSize])
		le.PutUint32(resp, gpudev.RespErrUnspec)
		n = gpudev.HdrSize
	}
	return virtq.Scatter(c, resp[:n])
}

// Read8 .. Write64 implement common.BARMemoryAccessor.
func (t *Transport) Read8(bar uint8, off uint64) (uint8, error) {
	return uint8(t.read(bar, off, 1)), nil
}

// Read16 reads BAR memory.
func (t *Transport) Read16(bar uint8, off uint64) (uint16, error) {
	return uint16(t.read(bar, off, 2)), nil
}

// Read32 reads BAR memory.
func (t *Transport) Read32(bar uint8, off uint64) (uint32, error) {
	return uint32(t.read(bar, off, 4)), nil
}

// Read64 reads BAR memory.
func (t *Transport) Read64(bar uint8, off uint64) (uint64, error) { return t.read(bar, off, 8), nil }

// Write8 writes BAR memory.
func (t *Transport) Write8(bar uint8, off uint64, v uint8) error {
	t.write(bar, off, uint64(v))
	return nil
}

// Write16 writes BAR memory.
func (t *Transport) Write16(bar uint8, off uint64, v uint16) error {
	t.write(bar, off, uint64(v))
	return nil
}

// Write32 writes BAR memory.
func (t *Transport) Write32(bar uint8, off uint64, v uint32) error {
	t.write(bar, off, uint64(v))
	return nil
}

// Write64 writes BAR memory.
func (t *Transport) Write64(bar uint8, off uint64, v uint64) error {
	t.write(bar, off, v)
	return nil
}

var _ common.Transport = (*Transport)(nil)
