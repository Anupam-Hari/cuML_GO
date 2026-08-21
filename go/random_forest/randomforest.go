package randomforest

import (
	"fmt"
	"unsafe"

	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
	"github.com/Anupam-Hari/cuml-go/go/internal/matrix"
)

type RandomForest struct {
	gpu *capi.RandomForestHandle
	cpu *capi.RFCPUHandle

	nEstimators int
	maxDepth    int
	maxFeatures float32
	maxLeaves   int
	maxSamples  float32

	backend int
}

type Option func(*RandomForest)

const (
	BackendCPU = 0
	BackendGPU = 1
)

func WithBackend(b int) Option {
	return func(rf *RandomForest) {
		rf.backend = b
	}
}

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
		backend:     BackendGPU,
	}

	for _, opt := range opts {
		opt(rf)
	}

	switch rf.backend {

	case BackendGPU:
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

		rf.gpu = handle

	case BackendCPU:
		handle, err := capi.RFCPUCreate(
			rf.nEstimators,
			rf.maxDepth,
			rf.maxFeatures,
			rf.maxLeaves,
			rf.maxSamples,
		)
		if err != nil {
			return nil, err
		}

		rf.cpu = handle

	default:
		return nil, fmt.Errorf(
			"invalid random forest backend: %d",
			rf.backend,
		)
	}

	return rf, nil
}

func (rf *RandomForest) Fit(
	X [][]float32,
	y []int,
) error {

	if rf == nil {
		return fmt.Errorf("random forest is nil")
	}

	dense, err := matrix.From2D(X)
	if err != nil {
		return err
	}

	if len(y) != dense.Rows {
		return fmt.Errorf(
			"number of labels does not match number of rows",
		)
	}

	nClasses, err := matrix.NumClasses(y)
	if err != nil {
		return err
	}

	labels := matrix.ToCInt(y)

	switch rf.backend {

	case BackendGPU:
		if rf.gpu == nil {
			return fmt.Errorf("random forest GPU is closed")
		}

		return capi.RandomForestFit(
			rf.gpu,
			dense.Data,
			dense.Rows,
			dense.Cols,
			labels,
			nClasses,
		)

	case BackendCPU:
		if rf.cpu == nil {
			return fmt.Errorf("random forest CPU is closed")
		}

		return capi.RFCPUFit(
			rf.cpu,
			dense.Data,
			dense.Rows,
			dense.Cols,
			labels,
			nClasses,
		)

	default:
		return fmt.Errorf(
			"invalid random forest backend: %d",
			rf.backend,
		)
	}
}

func (rf *RandomForest) Predict(
	X [][]float32,
) ([]int, error) {

	if rf == nil {
		return nil, fmt.Errorf("random forest is nil")
	}

	dense, err := matrix.From2D(X)
	if err != nil {
		return nil, err
	}

	switch rf.backend {

	case BackendGPU:
		if rf.gpu == nil {
			return nil, fmt.Errorf("random forest GPU is closed")
		}

		predictions, err := capi.RandomForestPredict(
			rf.gpu,
			dense.Data,
			dense.Rows,
			dense.Cols,
			BackendGPU,
		)
		if err != nil {
			return nil, err
		}

		return matrix.FromCInt32(predictions), nil

	case BackendCPU:
		if rf.cpu == nil {
			return nil, fmt.Errorf("random forest CPU is closed")
		}

		predictions, err := capi.RFCPUPredict(
			rf.cpu,
			dense.Data,
			dense.Rows,
			dense.Cols,
		)
		if err != nil {
			return nil, err
		}

		return matrix.FromCInt32(predictions), nil

	default:
		return nil, fmt.Errorf(
			"invalid random forest backend: %d",
			rf.backend,
		)
	}
}

func (rf *RandomForest) Close() {

	if rf == nil {
		return
	}

	if rf.gpu != nil {
		capi.RandomForestDestroy(rf.gpu)
		rf.gpu = nil
	}

	if rf.cpu != nil {
		capi.RFCPUFree(rf.cpu)
		rf.cpu = nil
	}
}

func (rf *RandomForest) Save(filename string) error {

	if rf == nil {
		return fmt.Errorf("random forest is nil")
	}

	if rf.backend != BackendGPU {
		return fmt.Errorf(
			"Save is only supported for the GPU backend",
		)
	}

	if rf.gpu == nil {
		return fmt.Errorf("random forest GPU is closed")
	}

	return capi.RandomForestSave(
		rf.gpu,
		filename,
	)
}

func Load(filename string) (*RandomForest, error) {

	handle, err := capi.RandomForestLoad(filename)
	if err != nil {
		return nil, err
	}

	return &RandomForest{
		gpu:     handle,
		backend: BackendGPU,
	}, nil
}

func SetCPUThreads(threads int) {
	capi.SetCPUThreads(threads)
}

func GetCPUThreads() int {
	return capi.GetCPUThreads()
}

// Prevent accidental copying of a live model.
func (rf *RandomForest) Handle() unsafe.Pointer {

	if rf == nil {
		return nil
	}

	switch rf.backend {

	case BackendGPU:
		return unsafe.Pointer(rf.gpu)

	case BackendCPU:
		return unsafe.Pointer(rf.cpu)

	default:
		return nil
	}
}