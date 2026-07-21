package capi

/*
#cgo CPPFLAGS: -I${SRCDIR}/../../../cpp/gpu_monitor
#cgo LDFLAGS: -L${SRCDIR}/../../../build -L/usr/lib/x86_64-linux-gnu -lcumlgo -lnvidia-ml

#include "../../../cpp/gpu_monitor/gpu_monitor.h"
*/
import "C"

import "fmt"

type GPUMonitor struct {
	handle *C.GpuMonitorHandle
}

func NewGPUMonitor() (*GPUMonitor, error) {

	handle := C.gpu_monitor_create()
	if handle == nil {
		return nil, fmt.Errorf("failed to create GPU monitor")
	}

	return &GPUMonitor{
		handle: handle,
	}, nil
}

func (g *GPUMonitor) Start() error {

	ret := C.gpu_monitor_start(g.handle)
	if ret != 0 {
		return fmt.Errorf("failed to start GPU monitor")
	}

	return nil
}

func (g *GPUMonitor) Stop() {
	C.gpu_monitor_stop(g.handle)
}

func (g *GPUMonitor) Average() float64 {
	return float64(C.gpu_monitor_average(g.handle))
}

func (g *GPUMonitor) Peak() float64 {
	return float64(C.gpu_monitor_peak(g.handle))
}

func (g *GPUMonitor) Close() {
	if g.handle != nil {
		C.gpu_monitor_destroy(g.handle)
		g.handle = nil
	}
}