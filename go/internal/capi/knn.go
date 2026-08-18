package capi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cpp/knn
#cgo LDFLAGS: -L${SRCDIR}/../../../build -lcumlgo

#include "knn.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type KNNHandle struct {
	ptr *C.KNNHandle
}

func KNNCreate(
	k int,
) (*KNNHandle, error) {

	h := C.knn_create(
		C.int(k),
	)

	if h == nil {
		return nil, fmt.Errorf("failed to create KNN")
	}

	return &KNNHandle{
		ptr: h,
	}, nil
}

func KNNFit(
	h *KNNHandle,
	data []float32,
	rows int,
	cols int,
	labels []int32,
) error {

	status := C.knn_fit(
		h.ptr,
		(*C.float)(&data[0]),
		C.int(rows),
		C.int(cols),
		(*C.int)(&labels[0]),
	)

	if status != 0 {
		return fmt.Errorf("knn_fit failed")
	}

	return nil
}

func KNNPredict(
	h *KNNHandle,
	data []float32,
	rows int,
	cols int,
) ([]int32, error) {

	predictions := make([]int32, rows)

	status := C.knn_predict(
		h.ptr,
		(*C.float)(&data[0]),
		C.int(rows),
		C.int(cols),
		(*C.int)(&predictions[0]),
	)

	if status != 0 {
		return nil, fmt.Errorf("knn_predict failed")
	}

	return predictions, nil
}

func KNNLoadONNX(
	h *KNNHandle,
	filename string,
) error {


	cname := C.CString(filename)
	defer C.free(unsafe.Pointer(cname))

	status := C.knn_load_onnx(
		h.ptr,
		cname,
	)

	if status != 0 {
		return fmt.Errorf(
			"knn_load_onnx failed",
		)
	}

	return nil
}

func KNNPredictONNX(
	h *KNNHandle,
	data []float32,
	rows int,
	cols int,
) ([]int32, error) {

	predictions := make(
		[]int32,
		rows,
	)

	status := C.knn_predict_onnx(
		h.ptr,
		(*C.float)(&data[0]),
		C.int(rows),
		C.int(cols),
		(*C.int)(&predictions[0]),
	)

	if status != 0 {

		return nil,
			fmt.Errorf(
				"knn_predict_onnx failed",
			)
	}

	return predictions, nil
}

func KNNDestroy(h *KNNHandle) {
	if h == nil || h.ptr == nil {
		return
	}

	C.knn_destroy(h.ptr)
	h.ptr = nil
}