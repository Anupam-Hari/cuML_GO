package kmeans

import (
	"fmt"
	"unsafe"

	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
	"github.com/Anupam-Hari/cuml-go/go/internal/matrix"
)

type KMeans struct {
	handle *capi.KMeansHandle

	nClusters int
	maxIters  int
	tolerance float32
}

type Option func(*KMeans)

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
	}

	for _, opt := range opts {
		opt(kmeans)
	}

	handle, err := capi.KMeansCreate(
		kmeans.nClusters,
		kmeans.maxIters,
		kmeans.tolerance,
	)
	if err != nil {
		return nil, err
	}

	kmeans.handle = handle

	return kmeans, nil
}

func (kmeans *KMeans) Fit(X [][]float32) error {
	if kmeans.handle == nil {
		return fmt.Errorf("KMeans is closed")
	}

	dense, err := matrix.From2D(X)
	if err != nil {
		return err
	}

	return capi.KMeansFit(
		kmeans.handle,
		dense.Data,
		dense.Rows,
		dense.Cols,
	)
}

func (kmeans *KMeans) Predict(X [][]float32) ([]int, error) {
	if kmeans.handle == nil {
		return nil, fmt.Errorf("KMeans is closed")
	}

	dense, err := matrix.From2D(X)
	if err != nil {
		return nil, err
	}

	predictions, err := capi.KMeansPredict(
		kmeans.handle,
		dense.Data,
		dense.Rows,
		dense.Cols,
	)
	if err != nil {
		return nil, err
	}

	return matrix.FromCInt32(predictions), nil
}

func (kmeans *KMeans) Close() {
	if kmeans.handle != nil {
		capi.KMeansDestroy(kmeans.handle)
		kmeans.handle = nil
	}
}

// Prevent accidental copying of a live model.
func (kmeans *KMeans) Handle() unsafe.Pointer {
	return unsafe.Pointer(kmeans.handle)
}