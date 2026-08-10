package kmeans

import (
	"flag"
	"os"
	"testing"

	"github.com/Anupam-Hari/cuml-go/go/internal/dataset"
)

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
	kmeansGPU, _ := New(
		WithBackend(BackendGPU),
		WithNClusters(8),
	)

	_ = kmeansGPU.Fit(X)
	labelsGPU, _ := kmeansGPU.Predict(X)

	// ---------------- CPU ----------------
	kmeansCPU, _ := New(
		WithBackend(BackendCPU),
		WithNClusters(8),
	)

	_ = kmeansCPU.Fit(X)
	labelsCPU, _ := kmeansCPU.Predict(X)

	// ---------------- inertia ----------------
	inertiaGPU := computeInertia(X, labelsGPU)
	inertiaCPU := computeInertia(X, labelsCPU)

	// ---------------- logs ----------------
	t.Logf("Samples        : %d", len(X))
	t.Logf("GPU Inertia    : %.6f", inertiaGPU)
	t.Logf("CPU Inertia    : %.6f", inertiaCPU)

	// sanity
	if inertiaCPU == 0 || inertiaGPU == 0 {
		t.Fatalf("invalid inertia")
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

	// compute centroids
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

	// compute inertia
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