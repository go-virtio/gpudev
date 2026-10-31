package gpudev

// batchContext is a webgpu-batch context.
type batchContext struct{}

// dispatch3D handles the context commands. Contexts are not offered yet:
// every 3D command is refused.
func (d *Device) dispatch3D(hdr, []byte) response {
	return errResp(RespErrUnspec)
}
