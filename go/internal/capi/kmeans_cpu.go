package capi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../cpp/kmeans_cpu
#cgo LDFLAGS: -L${SRCDIR}/../../../build -lcumlgo

#include "kmeans_cpu.h"
*/
import "C"
import "unsafe"

type KMeansCPUHandle C.KMeansCPUHandle

func KMeansCPUCreate(k, maxIters int, tol float32) *KMeansCPUHandle {
	return (*KMeansCPUHandle)(C.kmeans_cpu_create(
		C.int(k),
		C.int(maxIters),
		C.float(tol),
	))
}

func KMeansCPUFit(
	h *KMeansCPUHandle,
	X []float32,
	rows, cols int,
) {
	C.kmeans_cpu_fit(
		(*C.KMeansCPUHandle)(h),
		(*C.float)(unsafe.Pointer(&X[0])),
		C.int(rows),
		C.int(cols),
	)
}

func KMeansCPUPredict(
	h *KMeansCPUHandle,
	X []float32,
	rows int,
) []int32 {

	out := make([]int32, rows)

	C.kmeans_cpu_predict(
		(*C.KMeansCPUHandle)(h),
		(*C.float)(unsafe.Pointer(&X[0])),
		C.int(rows),
		(*C.int)(unsafe.Pointer(&out[0])),
	)

	return out
}

func KMeansCPUFree(h *KMeansCPUHandle) {
	C.kmeans_cpu_free((*C.KMeansCPUHandle)(h))
}