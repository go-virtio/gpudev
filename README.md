# gpudev

`github.com/go-virtio/gpudev` is the host side of virtio-gpu: a pure-Go device
model that guest drivers talk to (go-virtio/gpu, the Linux `virtio_gpu`
driver, or any other). It executes the control queue's 2D commands,
answers the capset queries, hosts webgpu-batch contexts (capset 30:
go-gfx/webgpu command streams from a guest), and presents through a
pluggable backend.

It is the counterpart of [go-virtio/gpu](https://github.com/go-virtio/gpu),
which is guest-side only. Design:
[go-virtio/gpu docs/WEBGPU.md](https://github.com/go-virtio/gpu/blob/main/docs/WEBGPU.md).

## Packages

| package | what |
|---|---|
| `gpudev` | the device model: resources, backing, `TRANSFER_TO_HOST_2D`, scanouts, flushes, the capset queries, and a `Backend` interface |
| `gpudev/virtq` | the device side of a split virtqueue, which checks everything in the rings |
| `gpudev/pcisim` | an emulated virtio-pci modern transport, in-process, that implements `go-virtio/common.Transport` |
| `gpudev/memback` | an in-memory backend (byte images, composed scanouts) for tests and fuzzing |

```go
ram := &gpudev.RAM{Base: 1 << 30, Bytes: make([]byte, 64<<20)}
dev := gpudev.New(gpudev.Config{Scanouts: []gpudev.Rect{{Width: 1024, Height: 768}}}, ram, memback.New())
tr := pcisim.New(dev, ram)

g, _ := gpu.OpenVirtioGPU(tr) // the go-virtio/gpu guest driver, unchanged
fb, _ := g.SetupFramebuffer(0, 1024, 768)
fb.Flush()
```

## Security

The guest is untrusted. Every id, size, offset, count and rectangle is
bounded in 64-bit arithmetic before it reaches the backend or guest
memory, on 32-bit hosts too:

- resource sizes are bounded by `MaxHostMem` (`w*h` is checked before
  `*4`), resource count by `MaxResources`, and backing entries by
  `MaxBackingEntries`;
- every backing entry and ring must lie in guest memory;
- a transfer's whole span must lie in the backing. QEMU truncates the
  64-bit offset instead (`virtio-gpu.c:531`);
- chains may not loop, exceed the queue size, or put readable buffers
  after writable ones; indirect descriptors are refused;
- requests are copied out of guest memory before they are decoded.

Each decoder of guest input has a fuzz target, run in CI:
`FuzzHandleControl` and `FuzzPop`.

## Status

- **2D: validated in-process against the go-virtio/gpu guest driver.** The
  driver brings the device up over `pcisim` and draws frames through real
  virtqueues; the scanout holds exactly what it drew (`e2e`). It has not
  yet run under a hypervisor.
- **webgpu-batch contexts: validated in-process against the go-virtio/gpu
  guest driver.** The driver queries the capset, creates a context and
  makes fenced, ringed `SUBMIT_3D` calls, and the device runs them
  against the in-memory backend (`e2e`). Each frame is decoded with
  `wgpubatch.Decode`. Guest object ids are private to the context and are
  translated to the backend's ids, in streams (`wire.Remap`) and in
  creation descriptors (`wire.RemapDescriptor`), before the backend sees
  them. The backend then checks each stream with go-gfx/webgpu's own
  reader. A frame that fails loses its context. `BIND_RESOURCE` gives a
  bounded window of a resource attached to that same context. Nothing
  has replayed on a GPU yet.
- **WebGPU backend:** not written yet.

100% statement coverage in every package, with tests on linux, macOS,
windows, 386 and s390x.

## License

BSD-3-Clause.
