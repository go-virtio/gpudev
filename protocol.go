package gpudev

// Wire constants of the virtio-gpu control protocol, transcribed from
// Linux include/uapi/linux/virtio_gpu.h (enum virtio_gpu_ctrl_type, the
// command structs, VIRTIO_GPU_FORMAT_*) and the virtio spec,
// device-types/gpu/description.tex :197-297. All fields are little-endian.

// Command types.
const (
	CmdGetDisplayInfo        uint32 = 0x0100
	CmdResourceCreate2D      uint32 = 0x0101
	CmdResourceUnref         uint32 = 0x0102
	CmdSetScanout            uint32 = 0x0103
	CmdResourceFlush         uint32 = 0x0104
	CmdTransferToHost2D      uint32 = 0x0105
	CmdResourceAttachBacking uint32 = 0x0106
	CmdResourceDetachBacking uint32 = 0x0107
	CmdGetCapsetInfo         uint32 = 0x0108
	CmdGetCapset             uint32 = 0x0109

	CmdCtxCreate         uint32 = 0x0200
	CmdCtxDestroy        uint32 = 0x0201
	CmdCtxAttachResource uint32 = 0x0202
	CmdCtxDetachResource uint32 = 0x0203
	CmdSubmit3D          uint32 = 0x0207

	CmdUpdateCursor uint32 = 0x0300
	CmdMoveCursor   uint32 = 0x0301
)

// Response types.
const (
	RespOKNoData      uint32 = 0x1100
	RespOKDisplayInfo uint32 = 0x1101
	RespOKCapsetInfo  uint32 = 0x1102
	RespOKCapset      uint32 = 0x1103

	RespErrUnspec            uint32 = 0x1200
	RespErrOutOfMemory       uint32 = 0x1201
	RespErrInvalidScanoutID  uint32 = 0x1202
	RespErrInvalidResourceID uint32 = 0x1203
	RespErrInvalidContextID  uint32 = 0x1204
	RespErrInvalidParameter  uint32 = 0x1205
)

// ctrl_hdr flags (description.tex:255-256).
const (
	FlagFence       uint32 = 1 << 0
	FlagInfoRingIdx uint32 = 1 << 1
)

// Feature bits (virtio_gpu.h VIRTIO_GPU_F_*).
const (
	FeatureVirgl        uint64 = 1 << 0
	FeatureEDID         uint64 = 1 << 1
	FeatureResourceBlob uint64 = 1 << 3
	FeatureContextInit  uint64 = 1 << 4
)

// HdrSize is sizeof(struct virtio_gpu_ctrl_hdr).
const HdrSize = 24

// MaxScanouts is VIRTIO_GPU_MAX_SCANOUTS (virtio_gpu.h:244).
const MaxScanouts = 16

// Format is a VIRTIO_GPU_FORMAT_* value (virtio_gpu.h:381-390). Every
// format is 4 bytes per pixel; the name gives the byte order in memory.
type Format uint32

// The 2D formats.
const (
	FormatB8G8R8A8 Format = 1
	FormatB8G8R8X8 Format = 2
	FormatA8R8G8B8 Format = 3
	FormatX8R8G8B8 Format = 4
	FormatR8G8B8A8 Format = 67
	FormatX8B8G8R8 Format = 68
	FormatA8B8G8R8 Format = 121
	FormatR8G8B8X8 Format = 134
)

// Valid reports whether f is one of the 2D formats.
func (f Format) Valid() bool {
	switch f {
	case FormatB8G8R8A8, FormatB8G8R8X8, FormatA8R8G8B8, FormatX8R8G8B8,
		FormatR8G8B8A8, FormatX8B8G8R8, FormatA8B8G8R8, FormatR8G8B8X8:
		return true
	}
	return false
}

// BytesPerPixel of every 2D format.
const BytesPerPixel = 4

// Rect is struct virtio_gpu_rect.
type Rect struct{ X, Y, Width, Height uint32 }

// within reports whether r lies inside a w×h area, in 64-bit arithmetic.
func (r Rect) within(w, h uint32) bool {
	return uint64(r.X)+uint64(r.Width) <= uint64(w) && uint64(r.Y)+uint64(r.Height) <= uint64(h)
}

// intersect returns r ∩ s and whether it is not empty.
func (r Rect) intersect(s Rect) (Rect, bool) {
	x0, y0 := max(uint64(r.X), uint64(s.X)), max(uint64(r.Y), uint64(s.Y))
	x1 := min(uint64(r.X)+uint64(r.Width), uint64(s.X)+uint64(s.Width))
	y1 := min(uint64(r.Y)+uint64(r.Height), uint64(s.Y)+uint64(s.Height))
	if x0 >= x1 || y0 >= y1 {
		return Rect{}, false
	}
	return Rect{uint32(x0), uint32(y0), uint32(x1 - x0), uint32(y1 - y0)}, true
}
