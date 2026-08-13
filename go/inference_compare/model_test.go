package main

import (
	"os"
	"testing"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
	randomforest "github.com/Anupam-Hari/cuml-go/go/random_forest"
)

func TestTrainAndSaveModel(t *testing.T) {

	X, y, err := dataset.LoadCSV(
		"../../benchmark/data/processed_network_traffic.csv",
		"is_malicious",
		1000000,
	)
	if err != nil {
		t.Fatal(err)
	}

	rf, err := randomforest.New(
		randomforest.WithEstimators(100),
		randomforest.WithMaxDepth(10),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer rf.Close()

	err = rf.Fit(X, y)
	if err != nil {
		t.Fatal(err)
	}

	const modelFile = "./rf_model.tl"

	err = rf.Save(modelFile)
	if err != nil {
		t.Fatal(err)
	}

	info, err := os.Stat(modelFile)
	if err != nil {
		t.Fatal(err)
	}

	if info.Size() == 0 {
		t.Fatal("saved model is empty")
	}

	t.Logf("Saved model: %s (%d bytes)", modelFile, info.Size())

	rf.Close()

	loaded, err := randomforest.Load(modelFile)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()

	gpuPred, err := loaded.Predict(
		X,
		randomforest.BackendGPU,
	)
	if err != nil {
		t.Fatal(err)
	}

	cpuPred, err := loaded.Predict(
		X,
		randomforest.BackendCPU,
	)
	if err != nil {
		t.Fatal(err)
	}

	if len(gpuPred) != len(y) {
		t.Fatal("GPU prediction length mismatch")
	}

	if len(cpuPred) != len(y) {
		t.Fatal("CPU prediction length mismatch")
	}

	for i := range gpuPred {

		if gpuPred[i] != cpuPred[i] {
			t.Fatalf(
				"prediction mismatch at sample %d: gpu=%d cpu=%d",
				i,
				gpuPred[i],
				cpuPred[i],
			)
		}
	}

	t.Logf(
		"Successfully loaded model and predicted %d samples",
		len(gpuPred),
	)
}