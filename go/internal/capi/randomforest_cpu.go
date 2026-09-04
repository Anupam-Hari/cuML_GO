package capi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cpp/random_forest_cpu
#cgo LDFLAGS: -L${SRCDIR}/../../../build -lcumlgo

#include "random_forest_cpu.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type RFCPUHandle struct {
	ptr *C.RFCPUHandle
}

func RFCPUCreate(
	nEstimators int,
	maxDepth int,
	maxFeatures float32,
	maxLeaves int,
	maxSamples float32,
) (*RFCPUHandle, error) {

	h := C.rf_cpu_create(
		C.int(nEstimators),
		C.int(maxDepth),
		C.float(maxFeatures),
		C.int(maxLeaves),
		C.float(maxSamples),
	)

	if h == nil {
		return nil, fmt.Errorf("rf_cpu_create failed")
	}

	return &RFCPUHandle{
		ptr: h,
	}, nil
}

func SetCPUThreadsCPU(h *RFCPUHandle, threads int) {
    if h == nil || h.ptr == nil || threads <= 0 {
        return
    }

    C.rf_set_cpu_threads_cpu(
        h.ptr,
        C.int(threads),
    )
}

func RFCPUFit(
	h *RFCPUHandle,
	data []float32,
	rows int,
	cols int,
	labels []int32,
	nClasses int,
) error {

	status := C.rf_cpu_fit(
		h.ptr,
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(rows),
		C.int(cols),
		(*C.int)(unsafe.Pointer(&labels[0])),
		C.int(nClasses),
	)

	if status != 0 {
		return fmt.Errorf("rf_cpu_fit failed")
	}

	return nil
}

func RFCPUPredict(
	h *RFCPUHandle,
	data []float32,
	rows int,
	cols int,
) ([]int32, error) {

	predictions := make([]int32, rows)

	status := C.rf_cpu_predict(
		h.ptr,
		(*C.float)(unsafe.Pointer(&data[0])),
		C.int(rows),
		C.int(cols),
		(*C.int)(unsafe.Pointer(&predictions[0])),
	)

	if status != 0 {
		return nil, fmt.Errorf("rf_cpu_predict failed")
	}

	return predictions, nil
}

func RFCPUFree(h *RFCPUHandle) {

	if h == nil || h.ptr == nil {
		return
	}

	C.rf_cpu_destroy(h.ptr)
	h.ptr = nil
}