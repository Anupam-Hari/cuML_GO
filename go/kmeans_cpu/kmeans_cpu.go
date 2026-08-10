package kmeans_cpu

import "github.com/Anupam-Hari/cuml-go/go/internal/capi"

type KMeansCPU struct {
	model *capi.KMeansCPUHandle
}

func New(
	nClusters int,
	maxIters int,
	tolerance float32,
) *KMeansCPU {

	return &KMeansCPU{
		model: capi.KMeansCPUCreate(
			nClusters,
			maxIters,
			tolerance,
		),
	}
}

func (k *KMeansCPU) Fit(
	X []float32,
	nSamples int,
	nFeatures int,
) {

	capi.KMeansCPUFit(
		k.model,
		X,
		nSamples,
		nFeatures,
	)
}

func (k *KMeansCPU) Predict(
	X []float32,
	nSamples int,
) []int32 {

	return capi.KMeansCPUPredict(
		k.model,
		X,
		nSamples,
	)
}

func (k *KMeansCPU) Free() {
	capi.KMeansCPUFree(k.model)
}