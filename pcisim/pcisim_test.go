package pcisim

import (
	"encoding/binary"
	"errors"
	"testing"

	"github.com/go-virtio/common"
	"github.com/go-virtio/gpu"
	"github.com/go-virtio/gpudev"
	"github.com/go-virtio/gpudev/memback"
)

func newT(t *testing.T, ramSize int) (*Transport, *gpudev.RAM, *memback.Backend) {
	t.Helper()
	ram := &gpudev.RAM{Base: 1 << 30, Bytes: make([]byte, ramSize)}
	be := memback.New()
	dev := gpudev.New(gpudev.Config{Scanouts: []gpudev.Rect{{Width: 64, Height: 48}}}, ram, be)
	return New(dev, ram), ram, be
}

// bringUp drives the modern-transport init with go-virtio/common and sets
// up both queues, returning them for hand-made chains.
func bringUp(t *testing.T, tr *Transport) (*common.ModernConfig, [2]*common.Virtqueue) {
	t.Helper()
	cfg, err := common.InitModernConfig(tr)
	if err != nil {
		t.Fatal(err)
	}
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	must(cfg.SetDeviceStatus(0))
	must(cfg.SetDeviceStatus(common.StatusAcknowledge | common.StatusDriver))
	must(cfg.SetDriverFeatures64(common.FeatureVersion1))
	must(cfg.SetDeviceStatus(common.StatusAcknowledge | common.StatusDriver | common.StatusFeaturesOK))
	var qs [2]*common.Virtqueue
	for i := range qs {
		must(cfg.SelectQueue(uint16(i)))
		q, err := common.NewVirtqueue(tr, 8, uint16(i), uint16(i))
		must(err)
		must(cfg.SetQueueSize(8))
		must(cfg.SetQueueDesc(q.BasePhys + uint64(q.Layout.DescTableOffset)))
		must(cfg.SetQueueDriver(q.BasePhys + uint64(q.Layout.AvailRingOffset)))
		must(cfg.SetQueueDevice(q.BasePhys + uint64(q.Layout.UsedRingOffset)))
		must(cfg.SetQueueEnable(1))
		qs[i] = q
	}
	must(cfg.SetDeviceStatus(common.StatusAcknowledge | common.StatusDriver | common.StatusFeaturesOK | common.StatusDriverOK))
	return cfg, qs
}

func TestGuestDriver(t *testing.T) {
	tr, _, be := newT(t, 4<<20)
	g, err := gpu.OpenVirtioGPU(tr)
	if err != nil {
		t.Fatal(err)
	}
	fb, err := g.SetupFramebuffer(0, 64, 48)
	if err != nil {
		t.Fatal(err)
	}
	fb.Pix[5] = 0xAB
	if err := fb.Flush(); err != nil || be.Scanouts[0].Pix[5] != 0xAB {
		t.Fatalf("flush: %v", err)
	}
	if v, _ := tr.Read32(0, deviceOff+8); v != 1 {
		t.Errorf("num_scanouts %d", v)
	}
	if v, _ := tr.Read32(0, deviceOff+12); v != 0 {
		t.Errorf("num_capsets %d", v)
	}
}

func TestRegisters(t *testing.T) {
	tr, _, _ := newT(t, 1<<20)
	if _, err := tr.ReadConfig16(255); err == nil {
		t.Error("16-bit config read past 256")
	}
	if _, err := tr.ReadConfig32(253); err == nil {
		t.Error("32-bit config read past 256")
	}
	if v, _ := tr.ReadConfig8(0x34); v != 0x40 {
		t.Errorf("cap pointer %#x", v)
	}
	// Other BARs and unknown offsets read 0 and ignore writes.
	_ = tr.Write32(1, 0, 7)
	if v, _ := tr.Read32(1, 0); v != 0 {
		t.Error("BAR 1 not empty")
	}
	if v, _ := tr.Read64(0, 0x3000); v != 0 {
		t.Error("unknown offset not 0")
	}
	// Feature words: select 0, 1, and an out-of-range selector.
	_ = tr.Write32(0, common.CfgDeviceFeatureSelect, 1)
	if v, _ := tr.Read32(0, common.CfgDeviceFeature); v != 1 {
		t.Errorf("features high word %#x", v)
	}
	if v, _ := tr.Read32(0, common.CfgDeviceFeatureSelect); v != 1 {
		t.Error("device feature select not read back")
	}
	_ = tr.Write32(0, common.CfgDeviceFeatureSelect, 2)
	if v, _ := tr.Read32(0, common.CfgDeviceFeature); v != 0 {
		t.Error("feature word 2 not 0")
	}
	// Driver features: an out-of-range selector writes nothing.
	_ = tr.Write32(0, common.CfgDriverFeatureSelect, 2)
	_ = tr.Write32(0, common.CfgDriverFeature, 0xFFFFFFFF)
	if v, _ := tr.Read32(0, common.CfgDriverFeatureSelect); v != 2 || tr.driver != 0 {
		t.Error("driver feature select 2 wrote features")
	}
	// FEATURES_OK refuses unoffered features and a missing VERSION_1.
	_ = tr.Write32(0, common.CfgDriverFeatureSelect, 0)
	_ = tr.Write32(0, common.CfgDriverFeature, 1) // VIRGL: not offered
	_ = tr.Write8(0, common.CfgDeviceStatus, common.StatusFeaturesOK)
	if s, _ := tr.Read8(0, common.CfgDeviceStatus); s&common.StatusFeaturesOK != 0 {
		t.Error("unoffered feature accepted")
	}
	_ = tr.Write32(0, common.CfgDriverFeature, 0)
	_ = tr.Write8(0, common.CfgDeviceStatus, common.StatusFeaturesOK)
	if s, _ := tr.Read8(0, common.CfgDeviceStatus); s&common.StatusFeaturesOK != 0 {
		t.Error("FEATURES_OK without VERSION_1")
	}
	// Queue registers: an invalid size is ignored; values read back.
	_ = tr.Write16(0, common.CfgQueueSelect, 1)
	_ = tr.Write16(0, common.CfgQueueSize, 0)
	_ = tr.Write16(0, common.CfgQueueSize, maxQueueSize+1)
	if v, _ := tr.Read16(0, common.CfgQueueSize); v != maxQueueSize {
		t.Errorf("queue size %d", v)
	}
	if v, _ := tr.Read16(0, common.CfgNumQueues); v != numQueues {
		t.Error("num queues")
	}
	if v, _ := tr.Read16(0, common.CfgQueueNotifyOff); v != 1 {
		t.Error("notify off")
	}
	if v, _ := tr.Read16(0, common.CfgQueueSelect); v != 1 {
		t.Error("queue select")
	}
	_ = tr.Write64(0, common.CfgQueueDesc, 0x1234)
	_ = tr.Write64(0, common.CfgQueueDriver, 0x2234)
	_ = tr.Write64(0, common.CfgQueueDevice, 0x3234)
	d, _ := tr.Read64(0, common.CfgQueueDesc)
	a, _ := tr.Read64(0, common.CfgQueueDriver)
	u, _ := tr.Read64(0, common.CfgQueueDevice)
	if d != 0x1234 || a != 0x2234 || u != 0x3234 {
		t.Error("queue addresses not read back")
	}
	if v, _ := tr.Read16(0, common.CfgQueueEnable); v != 0 {
		t.Error("queue enabled")
	}
	// Enabling a queue whose rings lie outside RAM fails the device.
	_ = tr.Write16(0, common.CfgQueueEnable, 1)
	if tr.Err() == nil {
		t.Fatal("rings outside RAM accepted")
	}
	if s, _ := tr.Read8(0, common.CfgDeviceStatus); s&common.StatusNeedsReset == 0 {
		t.Error("NEEDS_RESET not set")
	}
	// Device config: a 16-bit read inside, a read straddling the end.
	if v, _ := tr.Read16(0, deviceOff+8); v != 1 {
		t.Errorf("num_scanouts low half %d", v)
	}
	if v, _ := tr.Read32(0, deviceOff+14); v != 0 {
		t.Error("straddling read served")
	}
	// Reset clears everything but the first error.
	_ = tr.Write8(0, common.CfgDeviceStatus, 0)
	if s, _ := tr.Read8(0, common.CfgDeviceStatus); s != 0 {
		t.Errorf("status after reset %#x", s)
	}
}

func TestAllocate(t *testing.T) {
	tr, ram, _ := newT(t, 3*int(common.PageSize))
	if _, _, err := tr.AllocatePages(0); !errors.Is(err, ErrOutOfMemory) {
		t.Error("0 pages")
	}
	p, mem, err := tr.AllocatePages(2)
	if err != nil || p != ram.Base || len(mem) != 2*int(common.PageSize) || cap(mem) != len(mem) {
		t.Fatalf("alloc: %#x %d %v", p, len(mem), err)
	}
	if _, _, err := tr.AllocatePages(2); !errors.Is(err, ErrOutOfMemory) {
		t.Error("allocated past RAM")
	}
	if p, _, _ := tr.AllocatePages(1); p != ram.Base+2*uint64(common.PageSize) {
		t.Errorf("third page at %#x", p)
	}
}

func chainOf(t *testing.T, tr *Transport, req []byte, respLen uint32) []common.ChainBuffer {
	t.Helper()
	p, mem, err := tr.AllocatePages(1)
	if err != nil {
		t.Fatal(err)
	}
	copy(mem, req)
	return []common.ChainBuffer{
		{Phys: p, Len: uint32(len(req))},
		{Phys: p + 2048, Len: respLen, Writable: true},
	}
}

func TestQueues(t *testing.T) {
	tr, ram, _ := newT(t, 40<<20)
	cfg, qs := bringUp(t, tr)
	// Queue registers no longer change after DRIVER_OK.
	_ = tr.Write16(0, common.CfgQueueSelect, 0)
	_ = tr.Write64(0, common.CfgQueueDesc, 1)
	if v, _ := tr.Read64(0, common.CfgQueueDesc); v == 1 {
		t.Error("queue desc writable after DRIVER_OK")
	}

	disp := make([]byte, gpudev.HdrSize)
	binary.LittleEndian.PutUint32(disp, gpudev.CmdGetDisplayInfo)
	run := func(q int, c []common.ChainBuffer) (uint32, []byte) {
		t.Helper()
		head, err := qs[q].AddChain(c)
		if err != nil {
			t.Fatal(err)
		}
		_ = cfg.NotifyQueue(uint16(q), uint16(q))
		id, n, ok := qs[q].PollUsed()
		if !ok || id != head {
			t.Fatalf("queue %d: chain not returned", q)
		}
		_ = qs[q].ReclaimChain(head)
		resp, _ := ram.Slice(c[len(c)-1].Phys, uint64(c[len(c)-1].Len))
		return n, resp
	}
	if n, resp := run(0, chainOf(t, tr, disp, 1024)); n != gpudev.HdrSize+16*24 || binary.LittleEndian.Uint32(resp) != gpudev.RespOKDisplayInfo {
		t.Errorf("display info: %d bytes", n)
	}
	// A response buffer too small for the answer: nothing written.
	if n, _ := run(0, chainOf(t, tr, disp, 100)); n != 0 {
		t.Errorf("short response buffer: %d bytes", n)
	}
	// The cursor queue returns chains with nothing written.
	if n, _ := run(1, chainOf(t, tr, disp, 64)); n != 0 {
		t.Errorf("cursorq wrote %d", n)
	}
	// A request over MaxRequest is answered with an error, uncopied:
	// three descriptors over the same 6 MiB region.
	big, _, _ := tr.AllocatePages(6 << 20 / int(common.PageSize))
	c := []common.ChainBuffer{{Phys: big, Len: 6 << 20}, {Phys: big, Len: 6 << 20}, {Phys: big, Len: 6 << 20}}
	c = append(c, chainOf(t, tr, nil, 64)[1])
	if n, resp := run(0, c); n != gpudev.HdrSize || binary.LittleEndian.Uint32(resp) != gpudev.RespErrUnspec {
		t.Errorf("oversized request: %d bytes", n)
	}
	c[3].Len = 8 // and no room even for the error header
	if n, _ := run(0, c); n != 0 {
		t.Errorf("oversized request, tiny response: %d", n)
	}
	// Notifications that do nothing: a queue that does not exist.
	_ = tr.Write32(0, notifyOff+8, 5)

	// A malformed ring stops the device.
	avail, _ := ram.Slice(qs[0].BasePhys+uint64(qs[0].Layout.AvailRingOffset), 4)
	binary.LittleEndian.PutUint16(avail[2:], binary.LittleEndian.Uint16(avail[2:])+20)
	_ = cfg.NotifyQueue(0, 0)
	if tr.Err() == nil {
		t.Fatal("malformed ring accepted")
	}
	_ = cfg.NotifyQueue(0, 0) // NEEDS_RESET: ignored
}

func TestNotifyBeforeReady(t *testing.T) {
	tr, _, _ := newT(t, 1<<20)
	_ = tr.Write32(0, notifyOff, 0) // before DRIVER_OK
	_ = tr.Write8(0, common.CfgDeviceStatus, common.StatusDriverOK)
	_ = tr.Write32(0, notifyOff, 0) // DRIVER_OK but the queue was never enabled
	if tr.Err() != nil {
		t.Error(tr.Err())
	}
}
