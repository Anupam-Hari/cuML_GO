package kmeans

import (
	"fmt"
	"unsafe"

	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
	"github.com/Anupam-Hari/cuml-go/go/internal/matrix"
)

type KMeans struct {
	gpu *capi.KMeansHandle
	cpu *capi.KMeansCPUHandle

	nClusters int
	maxIters  int
	tolerance float32

	backend int
}

type Option func(*KMeans)

const (
	BackendCPU = 0
	BackendGPU = 1
)

func WithBackend(b int) Option {
	return func(kmeans *KMeans) {
		kmeans.backend = b
	}
}

func WithNClusters(nClusters int) Option {
	return func(kmeans *KMeans) {
		kmeans.nClusters = nClusters
	}
}

func WithMaxIters(maxIters int) Option {
	return func(kmeans *KMeans) {
		kmeans.maxIters = maxIters
	}
}

func WithTolerance(tolerance float32) Option {
	return func(kmeans *KMeans) {
		kmeans.tolerance = tolerance
	}
}

func New(opts ...Option) (*KMeans, error) {
	kmeans := &KMeans{
		nClusters: 8,
		maxIters:  300,
		tolerance: 1e-4,
		backend:   BackendGPU,
	}

	for _, opt := range opts {
		opt(kmeans)
	}

	if kmeans.backend == BackendGPU {
		h, err := capi.KMeansCreate(
			kmeans.nClusters,
			kmeans.maxIters,
			kmeans.tolerance,
		)
		if err != nil {
			return nil, err
		}
		kmeans.gpu = h
	}

	return kmeans, nil
}

func (kmeans *KMeans) Fit(X [][]float32) error {

	dense, err := matrix.From2D(X)
	if err != nil {
		return err
	}

	if kmeans.backend == BackendGPU {

		return capi.KMeansFit(
			kmeans.gpu,
			dense.Data,
			dense.Rows,
			dense.Cols,
		)
	}

	// CPU path
	if kmeans.cpu == nil {

		cpu, err := capi.KMeansCPUCreate(
			kmeans.nClusters,
			kmeans.maxIters,
			kmeans.tolerance,
		)
		if err != nil {
			return err
		}

		kmeans.cpu = cpu
	}

	return capi.KMeansCPUFit(
		kmeans.cpu,
		dense.Data,
		dense.Rows,
		dense.Cols,
	)
}

func (kmeans *KMeans) Predict(X [][]float32) ([]int, error) {

	dense, err := matrix.From2D(X)
	if err != nil {
		return nil, err
	}

	if kmeans.backend == BackendGPU {

		pred, err := capi.KMeansPredict(
			kmeans.gpu,
			dense.Data,
			dense.Rows,
			dense.Cols,
		)
		if err != nil {
			return nil, err
		}

		return matrix.FromCInt32(pred), nil
	}

	// CPU path
	if kmeans.cpu == nil {
		return nil, fmt.Errorf("kmeans CPU model is not fitted")
	}

	pred, err := capi.KMeansCPUPredict(
		kmeans.cpu,
		dense.Data,
		dense.Rows,
	)
	if err != nil {
		return nil, err
	}

	return matrix.FromCInt32(pred), nil
}

func (kmeans *KMeans) Close() {

	if kmeans.gpu != nil {
		capi.KMeansDestroy(kmeans.gpu)
		kmeans.gpu = nil
	}

	if kmeans.cpu != nil {
		capi.KMeansCPUFree(kmeans.cpu)
		kmeans.cpu = nil
	}
}

// Prevent accidental copying of a live model.
func (kmeans *KMeans) Handle() unsafe.Pointer {
	if kmeans.backend == BackendGPU {
		return unsafe.Pointer(kmeans.gpu)
	}
	return unsafe.Pointer(kmeans.cpu)
}