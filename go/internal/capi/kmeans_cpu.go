package capi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cpp/kmeans_cpu
#cgo LDFLAGS: -L${SRCDIR}/../../../build -lcumlgo

#include "kmeans_cpu.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type KMeansCPUHandle struct {
	ptr *C.KMeansCPUHandle
}

func KMeansCPUCreate(
	k int,
	maxIters int,
	tol float32,
) (*KMeansCPUHandle, error) {

	h := C.kmeans_cpu_create(
		C.int(k),
		C.int(maxIters),
		C.float(tol),
	)

	if h == nil {
		return nil, fmt.Errorf("kmeans_cpu_create failed")
	}

	return &KMeansCPUHandle{
		ptr: h,
	}, nil
}

func KMeansCPUFit(
	h *KMeansCPUHandle,
	X []float32,
	rows int,
	cols int,
) error {

	if h == nil || h.ptr == nil {
		return fmt.Errorf("kmeans CPU handle is nil")
	}

	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid dimensions")
	}

	if len(X) != rows*cols {
		return fmt.Errorf(
			"invalid input size: got %d values, expected %d",
			len(X),
			rows*cols,
		)
	}

	status := C.kmeans_cpu_fit(
		h.ptr,
		(*C.float)(unsafe.Pointer(&X[0])),
		C.int(rows),
		C.int(cols),
	)

	if status != 0 {
		return fmt.Errorf("kmeans_cpu_fit failed")
	}

	return nil
}

func KMeansCPUPredict(
	h *KMeansCPUHandle,
	X []float32,
	rows int,
) ([]int32, error) {

	if h == nil || h.ptr == nil {
		return nil, fmt.Errorf("kmeans CPU handle is nil")
	}

	if rows <= 0 {
		return nil, fmt.Errorf("invalid number of rows")
	}

	if len(X) == 0 {
		return nil, fmt.Errorf("input data is empty")
	}

	/*
	 * The C++ handle stores n_features after Fit(),
	 * so predict only needs the number of rows.
	 */

	out := make([]int32, rows)

	status := C.kmeans_cpu_predict(
		h.ptr,
		(*C.float)(unsafe.Pointer(&X[0])),
		C.int(rows),
		(*C.int)(unsafe.Pointer(&out[0])),
	)

	if status != 0 {
		return nil, fmt.Errorf("kmeans_cpu_predict failed")
	}

	return out, nil
}

func KMeansCPUFree(
	h *KMeansCPUHandle,
) {
	if h == nil || h.ptr == nil {
		return
	}

	C.kmeans_cpu_free(h.ptr)
	h.ptr = nil
}