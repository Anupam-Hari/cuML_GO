package capi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cpp/kmeans
#cgo LDFLAGS: -L${SRCDIR}/../../../build -lcumlgo

#include "kmeans.h"
*/
import "C"

import (
	"fmt"
)

type KMeansHandle struct {
	ptr *C.KMeansHandle
}

func KMeansCreate(
	nClusters int,
	maxIters int,
	tolerance float32,
) (*KMeansHandle, error) {

	h := C.kmeans_create(
		C.int(nClusters),
		C.int(maxIters),
		C.float(tolerance),
	)

	if h == nil {
		return nil, fmt.Errorf("failed to create KMeans")
	}

	return &KMeansHandle{
		ptr: h,
	}, nil
}

func KMeansFit(
	h *KMeansHandle,
	data []float32,
	rows int,
	cols int,
) error {

	status := C.kmeans_fit(
		h.ptr,
		(*C.float)(&data[0]),
		C.int(rows),
		C.int(cols),
	)

	if status != 0 {
		return fmt.Errorf("kmeans_fit failed")
	}

	return nil
}

func KMeansPredict(
	h *KMeansHandle,
	data []float32,
	rows int,
	cols int,
) ([]int32, error) {

	predictions := make([]int32, rows)

	status := C.kmeans_predict(
		h.ptr,
		(*C.float)(&data[0]),
		C.int(rows),
		C.int(cols),
		(*C.int)(&predictions[0]),
	)

	if status != 0 {
		return nil, fmt.Errorf("kmeans_predict failed")
	}

	return predictions, nil
}

func KMeansDestroy(h *KMeansHandle) {
	if h == nil || h.ptr == nil {
		return
	}

	C.kmeans_destroy(h.ptr)
	h.ptr = nil
}