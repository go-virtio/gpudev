package gpudev

// Backend presents what the device model has validated. The Device
// checks every guest-supplied id, size, offset and rectangle before it
// calls a Backend, so an implementation may trust its arguments: ids
// name live resources, rectangles lie inside them, and src holds the
// rows Upload describes.
type Backend interface {
	// CreateResource makes a w×h 2D resource (w, h > 0, f valid).
	CreateResource(id uint32, f Format, w, h uint32) error
	// DestroyResource releases it; the Device has already detached it
	// from every scanout.
	DestroyResource(id uint32)
	// Upload writes r of the resource from src, where row y of r starts
	// at src[y*stride] and holds r.Width*BytesPerPixel bytes. src may
	// alias guest memory: it is only valid during the call.
	Upload(id uint32, r Rect, src []byte, stride uint64) error
	// SetScanout shows rect r of resource id on scanout (id 0: disable).
	SetScanout(scanout, id uint32, r Rect) error
	// Flush presents r (resource coordinates, inside the scanout's rect)
	// of resource id on scanout.
	Flush(scanout, id uint32, r Rect) error
	// NewContext makes the backend half of a webgpu-batch context.
	NewContext(ctx uint32) (Context, error)
}

// Context is the backend half of a webgpu-batch context. The Device has
// decoded and bounded each message and translated every object id from
// the guest's namespace to the ids Create returned.
type Context interface {
	// Create makes an object of a go-gfx/webgpu wire kind from its
	// descriptor (object references already translated) and returns the
	// id the backend's replayer knows it by (nonzero).
	Create(kind uint32, desc []byte) (id uint32, err error)
	// Destroy releases an object Create returned.
	Destroy(id uint32)
	// Replay executes one go-gfx/webgpu Recorder submit whose ids are
	// all ids Create returned, and returns once it has executed: the
	// Device answers a fenced SUBMIT_3D after Replay returns.
	Replay(words []uint32, data []byte) error
	// Bind gives an object a window of guest memory (a resource's
	// backing) for upload and readback.
	Bind(id uint32, w Window) error
	// Close releases everything the context holds.
	Close()
}

// Window is a bounded range of a resource's backing, for BIND_RESOURCE.
type Window interface {
	Len() uint64
	ReadAt(p []byte, off uint64) error
	WriteAt(p []byte, off uint64) error
}
