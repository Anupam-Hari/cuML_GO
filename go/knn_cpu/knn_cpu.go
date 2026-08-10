package knn_cpu

import "github.com/Anupam-Hari/cuml-go/go/internal/capi"

type KNNCPU struct {
	model *capi.KNNCPU
}

func New(
	X []float32,
	y []int32,
	nSamples, nFeatures, nClasses int,
) *KNNCPU {

	return &KNNCPU{
		model: capi.NewKNNCPU(X, y, nSamples, nFeatures, nClasses),
	}
}

func (k *KNNCPU) Predict(
	X []float32,
	nQueries, kVal int,
) []int32 {

	return k.model.Predict(X, nQueries, kVal)
}

func (k *KNNCPU) Free() {
	k.model.Free()
}