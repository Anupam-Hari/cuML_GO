package kmeans_cpu

import (
	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
)

type KMeansCPU struct {
	model *capi.KMeansCPUHandle
}

func New(
	nClusters int,
	maxIters int,
	tolerance float32,
) (*KMeansCPU, error) {

	model, err := capi.KMeansCPUCreate(
		nClusters,
		maxIters,
		tolerance,
	)
	if err != nil {
		return nil, err
	}

	return &KMeansCPU{
		model: model,
	}, nil
}

func (k *KMeansCPU) Fit(
	X []float32,
	nSamples int,
	nFeatures int,
) error {

	return capi.KMeansCPUFit(
		k.model,
		X,
		nSamples,
		nFeatures,
	)
}

func (k *KMeansCPU) Predict(
	X []float32,
	nSamples int,
) ([]int32, error) {

	return capi.KMeansCPUPredict(
		k.model,
		X,
		nSamples,
	)
}

func (k *KMeansCPU) Free() {
	if k == nil || k.model == nil {
		return
	}

	capi.KMeansCPUFree(k.model)
	k.model = nil
}