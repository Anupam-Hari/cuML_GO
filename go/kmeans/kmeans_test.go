package kmeans

import (
	"flag"
	"os"
	"testing"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

const onnxModel = "../../exported_models/kmeans_100000_n_clusters-8_repeat-1.onnx"

var maxRows = flag.Int(
	"rows",
	-1,
	"Maximum number of dataset rows",
)

func TestKMeans_BackendComparison(t *testing.T) {

	wd, _ := os.Getwd()
	t.Log("Working directory:", wd)

	X, _, err := dataset.LoadCSV(
		"../../benchmark/data/processed_network_traffic.csv",
		"is_malicious",
		*maxRows,
	)
	if err != nil {
		t.Fatal(err)
	}

	// ---------------- GPU ----------------

	kmeansGPU, err := New(
		WithBackend(BackendGPU),
		WithNClusters(8),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer kmeansGPU.Close()

	if err := kmeansGPU.Fit(X); err != nil {
		t.Fatal(err)
	}

	labelsGPU, err := kmeansGPU.Predict(X)
	if err != nil {
		t.Fatal(err)
	}

	// ---------------- CPU ----------------

	kmeansCPU, err := New(
		WithBackend(BackendCPU),
		WithNClusters(8),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer kmeansCPU.Close()

	if err := kmeansCPU.Fit(X); err != nil {
		t.Fatal(err)
	}

	labelsCPU, err := kmeansCPU.Predict(X)
	if err != nil {
		t.Fatal(err)
	}

	// ---------------- ONNX ----------------

	kmeansONNX, err := New(
		WithBackend(BackendGPU),
		WithNClusters(8),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer kmeansONNX.Close()

	err = kmeansONNX.LoadONNX(onnxModel)
	if err != nil {
		t.Fatal(err)
	}

	labelsONNX, err := kmeansONNX.PredictONNX(X)
	if err != nil {
		t.Fatal(err)
	}

	// ---------------- inertia ----------------

	inertiaGPU := computeInertia(X, labelsGPU)
	inertiaCPU := computeInertia(X, labelsCPU)
	inertiaONNX := computeInertia(X, labelsONNX)

	// ---------------- logs ----------------

	t.Logf("Samples         : %d", len(X))
	t.Logf("GPU Inertia     : %.6f", inertiaGPU)
	t.Logf("CPU Inertia     : %.6f", inertiaCPU)
	t.Logf("ONNX Inertia    : %.6f", inertiaONNX)

	// ---------------- sanity ----------------

	if inertiaGPU == 0 {
		t.Fatalf("invalid GPU inertia")
	}

	if inertiaCPU == 0 {
		t.Fatalf("invalid CPU inertia")
	}

	if inertiaONNX == 0 {
		t.Fatalf("invalid ONNX inertia")
	}

	if len(labelsGPU) != len(labelsCPU) {
		t.Fatalf(
			"label length mismatch GPU=%d CPU=%d",
			len(labelsGPU),
			len(labelsCPU),
		)
	}

	if len(labelsGPU) != len(labelsONNX) {
		t.Fatalf(
			"label length mismatch GPU=%d ONNX=%d",
			len(labelsGPU),
			len(labelsONNX),
		)
	}
}

func computeInertia(X [][]float32, labels []int) float64 {

	k := maxLabel(labels) + 1

	nSamples := len(X)
	nFeatures := len(X[0])

	centroids := make([][]float64, k)
	counts := make([]int, k)

	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, nFeatures)
	}

	for i := 0; i < nSamples; i++ {

		c := labels[i]
		counts[c]++

		for j := 0; j < nFeatures; j++ {
			centroids[c][j] += float64(X[i][j])
		}
	}

	for c := 0; c < k; c++ {

		if counts[c] == 0 {
			continue
		}

		for j := 0; j < nFeatures; j++ {
			centroids[c][j] /= float64(counts[c])
		}
	}

	var inertia float64

	for i := 0; i < nSamples; i++ {

		c := labels[i]

		var dist float64

		for j := 0; j < nFeatures; j++ {

			diff := float64(X[i][j]) - centroids[c][j]
			dist += diff * diff
		}

		inertia += dist
	}

	return inertia
}

func maxLabel(labels []int) int {

	m := labels[0]

	for _, v := range labels {
		if v > m {
			m = v
		}
	}

	return m
}