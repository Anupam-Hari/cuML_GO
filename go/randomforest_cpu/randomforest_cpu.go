package randomforest_cpu

import (
    "fmt"

    "github.com/Anupam-Hari/cuml-go/go/internal/capi"
    "github.com/Anupam-Hari/cuml-go/go/internal/matrix"
)

type RandomForestCPU struct {
    handle *capi.RFCPUHandle
}

func New(
    nEstimators int,
    maxDepth int,
    maxFeatures float32,
    maxLeaves int,
    maxSamples float32,
) (*RandomForestCPU, error) {

    h, err := capi.RFCPUCreate(
        nEstimators,
        maxDepth,
        maxFeatures,
        maxLeaves,
        maxSamples,
    )
    if err != nil {
        return nil, err
    }

    return &RandomForestCPU{
        handle: h,
    }, nil
}

func (rf *RandomForestCPU) Fit(
    X [][]float32,
    y []int,
) error {

    dense, err := matrix.From2D(X)
    if err != nil {
        return err
    }

    labels := matrix.ToCInt(y)

    nClasses, err := matrix.NumClasses(y)
    if err != nil {
        return err
    }

    return capi.RFCPUFit(
        rf.handle,
        dense.Data,
        dense.Rows,
        dense.Cols,
        labels,
        nClasses,
    )
}

func (rf *RandomForestCPU) Predict(
    X [][]float32,
) ([]int, error) {

    if rf.handle == nil {
        return nil, fmt.Errorf("random forest CPU is closed")
    }

    dense, err := matrix.From2D(X)
    if err != nil {
        return nil, err
    }

    pred, err := capi.RFCPUPredict(
        rf.handle,
        dense.Data,
        dense.Rows,
        dense.Cols,
    )
    if err != nil {
        return nil, err
    }

    return matrix.FromCInt32(pred), nil
}

func (rf *RandomForestCPU) Close() {
    if rf.handle != nil {
        capi.RFCPUFree(rf.handle)
        rf.handle = nil
    }
}