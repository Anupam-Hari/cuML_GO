package capi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cpp/random_forest
#cgo LDFLAGS: -L${SRCDIR}/../../../build -lcumlgo

#include "random_forest.h"
*/
import "C"

import (
	"fmt"
)

type RandomForestHandle struct {
	ptr *C.RFHandle
}

func RandomForestCreate(
	nEstimators int,
	maxDepth int,
	maxFeatures float32,
	maxLeaves int,
	maxSamples float32,
) (*RandomForestHandle, error) {

	h := C.rf_create(
		C.int(nEstimators),
		C.int(maxDepth),
		C.float(maxFeatures),
		C.int(maxLeaves),
		C.float(maxSamples),
	)

	if h == nil {
		return nil, fmt.Errorf("failed to create random forest")
	}

	return &RandomForestHandle{
		ptr: h,
	}, nil
}

func RandomForestFit(
	h *RandomForestHandle,
	data []float32,
	rows int,
	cols int,
	labels []int32,
	nClasses int,
) error {

	status := C.rf_fit(
		h.ptr,
		(*C.float)(&data[0]),
		C.int(rows),
		C.int(cols),
		(*C.int)(&labels[0]),
		C.int(nClasses),
	)

	if status != 0 {
		return fmt.Errorf("rf_fit failed")
	}

	return nil
}

func RandomForestPredict(
	h *RandomForestHandle,
	data []float32,
	rows int,
	cols int,
) ([]int32, error) {

	predictions := make([]int32, rows)

	status := C.rf_predict(
		h.ptr,
		(*C.float)(&data[0]),
		C.int(rows),
		C.int(cols),
		(*C.int)(&predictions[0]),
	)

	if status != 0 {
		return nil, fmt.Errorf("rf_predict failed")
	}

	return predictions, nil
}

func RandomForestDestroy(h *RandomForestHandle) {
	if h == nil || h.ptr == nil {
		return
	}

	C.rf_destroy(h.ptr)
	h.ptr = nil
}
