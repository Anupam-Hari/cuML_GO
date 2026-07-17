package randomforest

import (
	"github.com/Anupam-Hari/cuml-go/go/internal/matrix"
	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
)

import (
	"fmt"
	"unsafe"
)

type RandomForest struct {
	handle *capi.RandomForestHandle

	nEstimators int
	maxDepth    int
	maxFeatures float32
	maxLeaves   int
	maxSamples  float32
}

type Option func(*RandomForest)

func WithEstimators(n int) Option {
	return func(rf *RandomForest) {
		rf.nEstimators = n
	}
}

func WithMaxDepth(depth int) Option {
	return func(rf *RandomForest) {
		rf.maxDepth = depth
	}
}

func WithMaxFeatures(f float32) Option {
	return func(rf *RandomForest) {
		rf.maxFeatures = f
	}
}

func WithMaxLeaves(n int) Option {
	return func(rf *RandomForest) {
		rf.maxLeaves = n
	}
}

func WithMaxSamples(f float32) Option {
	return func(rf *RandomForest) {
		rf.maxSamples = f
	}
}

func New(opts ...Option) (*RandomForest, error) {
	rf := &RandomForest{
		nEstimators: 100,
		maxDepth:    16,
		maxFeatures: 1.0,
		maxLeaves:   -1,
		maxSamples:  1.0,
	}

	for _, opt := range opts {
		opt(rf)
	}

	handle, err := capi.RandomForestCreate(
		rf.nEstimators,
		rf.maxDepth,
		rf.maxFeatures,
		rf.maxLeaves,
		rf.maxSamples,
	)
	if err != nil {
		return nil, err
	}

	rf.handle = handle

	return rf, nil
}

func (rf *RandomForest) Fit(X [][]float32, y []int) error {
	if rf.handle == nil {
		return fmt.Errorf("random forest is closed")
	}

	dense, err := matrix.From2D(X)
	if err != nil {
		return err
	}

	if len(y) != dense.Rows {
		return fmt.Errorf("number of labels does not match number of rows")
	}

	nClasses, err := matrix.NumClasses(y)
	if err != nil {
		return err
	}

	labels := matrix.ToCInt(y)

	return capi.RandomForestFit(
   	 rf.handle,
   	 dense.Data,
   	 dense.Rows,
   	 dense.Cols,
   	 labels,
   	 nClasses,
	)
}

func (rf *RandomForest) Predict(X [][]float32) ([]int, error) {
	if rf.handle == nil {
		return nil, fmt.Errorf("random forest is closed")
	}

	dense, err := matrix.From2D(X)
	if err != nil {
		return nil, err
	}

	predictions, err := capi.RandomForestPredict(
		rf.handle,
		dense.Data,
		dense.Rows,
		dense.Cols,
	)
	if err != nil {
		return nil, err
	}

	return matrix.FromCInt32(predictions), nil


}

func (rf *RandomForest) Close() {
	if rf.handle != nil {
		capi.RandomForestDestroy(rf.handle)
		rf.handle = nil
	}
}

// Prevent accidental copying of a live model.
func (rf *RandomForest) Handle() unsafe.Pointer {
	return unsafe.Pointer(rf.handle)
}
