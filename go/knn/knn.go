package knn

import (
	"fmt"
	"unsafe"

	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
	"github.com/Anupam-Hari/cuml-go/go/internal/matrix"
)

type KNN struct {
    gpu *capi.KNNHandle
    cpu *capi.KNNCPUHandle

    k       int
    backend int
}

type Option func(*KNN)

const (
    BackendCPU = 0
    BackendGPU = 1
)

func WithBackend(b int) Option {
    return func(knn *KNN) {
        knn.backend = b
    }
}

func WithK(k int) Option {
	return func(knn *KNN) {
		knn.k = k
	}
}

func New(opts ...Option) (*KNN, error) {
    knn := &KNN{
        k:       5,
        backend: BackendGPU,
    }

    for _, opt := range opts {
        opt(knn)
    }

    if knn.backend == BackendGPU {
        h, err := capi.KNNCreate(knn.k)
        if err != nil {
            return nil, err
        }
        knn.gpu = h
    }

    return knn, nil
}

func (knn *KNN) Fit(X [][]float32, y []int) error {

    dense, err := matrix.From2D(X)
    if err != nil {
        return err
    }

    labels := matrix.ToCInt(y)

    if knn.backend == BackendGPU {

        return capi.KNNFit(
            knn.gpu,
            dense.Data,
            dense.Rows,
            dense.Cols,
            labels,
        )
    }

	nClasses, err := matrix.NumClasses(y)
	if err != nil {
		return err
	}

    // CPU path
    knn.cpu = capi.KNNCPUCreate(
        dense.Data,
        labels,
        dense.Rows,
        dense.Cols,
        nClasses,
    )

    if knn.cpu == nil {
        return fmt.Errorf("KNNCPUCreate failed")
    }

    return nil
}

func (knn *KNN) Predict(X [][]float32) ([]int, error) {

    dense, err := matrix.From2D(X)
    if err != nil {
        return nil, err
    }

    if knn.backend == BackendGPU {

        pred, err := capi.KNNPredict(
            knn.gpu,
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
    pred := capi.KNNCPUPredict(
        knn.cpu,
        dense.Data,
        dense.Rows,
        knn.k,
    )

    return matrix.FromCInt32(pred), nil
}

func (knn *KNN) Close() {

    if knn.gpu != nil {
        capi.KNNDestroy(knn.gpu)
        knn.gpu = nil
    }

    if knn.cpu != nil {
        capi.KNNCPUFree(knn.cpu)
        knn.cpu = nil
    }
}

// Prevent accidental copying of a live model.
func (knn *KNN) Handle() unsafe.Pointer {
    if knn.backend == BackendGPU {
        return unsafe.Pointer(knn.gpu)
    }
    return unsafe.Pointer(knn.cpu)
}