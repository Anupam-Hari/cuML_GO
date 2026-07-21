package knn

import (
	"fmt"
	"unsafe"

	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
	"github.com/Anupam-Hari/cuml-go/go/internal/matrix"
)

type KNN struct {
	handle *capi.KNNHandle

	k int
}

type Option func(*KNN)

func WithK(k int) Option {
	return func(knn *KNN) {
		knn.k = k
	}
}

func New(opts ...Option) (*KNN, error) {
	knn := &KNN{
		k: 5,
	}

	for _, opt := range opts {
		opt(knn)
	}

	handle, err := capi.KNNCreate(
		knn.k,
	)
	if err != nil {
		return nil, err
	}

	knn.handle = handle

	return knn, nil
}

func (knn *KNN) Fit(X [][]float32, y []int) error {
	if knn.handle == nil {
		return fmt.Errorf("KNN is closed")
	}

	dense, err := matrix.From2D(X)
	if err != nil {
		return err
	}

	if len(y) != dense.Rows {
		return fmt.Errorf("number of labels does not match number of rows")
	}

	labels := matrix.ToCInt(y)

	return capi.KNNFit(
		knn.handle,
		dense.Data,
		dense.Rows,
		dense.Cols,
		labels,
	)
}

func (knn *KNN) Predict(X [][]float32) ([]int, error) {
	if knn.handle == nil {
		return nil, fmt.Errorf("KNN is closed")
	}

	dense, err := matrix.From2D(X)
	if err != nil {
		return nil, err
	}

	predictions, err := capi.KNNPredict(
		knn.handle,
		dense.Data,
		dense.Rows,
		dense.Cols,
	)
	if err != nil {
		return nil, err
	}

	return matrix.FromCInt32(predictions), nil
}

func (knn *KNN) Close() {
	if knn.handle != nil {
		capi.KNNDestroy(knn.handle)
		knn.handle = nil
	}
}

// Prevent accidental copying of a live model.
func (knn *KNN) Handle() unsafe.Pointer {
	return unsafe.Pointer(knn.handle)
}