package capi

/*
#cgo CFLAGS: -I/home/anupam/projects/cuml-go/cpp/knn_cpu
#cgo LDFLAGS: -L/home/anupam/projects/cuml-go/build -lcumlgo
#include "knn_cpu.h"
*/
import "C"
import "unsafe"

type KNNCPUHandle struct {
    ptr *C.KNNCPUHandle
}

func KNNCPUCreate(
	X []float32,
	y []int32,
	rows, cols, classes int,
) *KNNCPUHandle {

	ptr := C.knn_cpu_create(
		(*C.float)(unsafe.Pointer(&X[0])),
		(*C.int)(unsafe.Pointer(&y[0])),
		C.int(rows),
		C.int(cols),
		C.int(classes),
	)

	if ptr == nil {
		return nil
	}

	return &KNNCPUHandle{
		ptr: ptr,
	}
}

func KNNCPUPredict(
	h *KNNCPUHandle,
	X []float32,
	rows, k int,
) []int32 {

	if h == nil {
		panic("KNNCPUHandle is nil")
	}

	if h.ptr == nil {
		panic("KNNCPUHandle.ptr is nil")
	}

	out := make([]int32, rows)

	C.knn_cpu_predict(
		h.ptr,
		(*C.float)(unsafe.Pointer(&X[0])),
		C.int(rows),
		C.int(k),
		(*C.int)(unsafe.Pointer(&out[0])),
	)

	return out
}

func KNNCPUFree(h *KNNCPUHandle) {
    if h == nil || h.ptr == nil {
        return
    }
    C.knn_cpu_free(h.ptr)
    h.ptr = nil
}