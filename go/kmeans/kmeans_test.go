package kmeans

import (
	"flag"
	"os"
	"testing"

	benchmarkutil "github.com/Anupam-Hari/cuml-go/go/internal/benchmark"
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

	gpuMetrics, err := benchmarkutil.NewMetrics()
	if err != nil {
		t.Fatal(err)
	}

	gpuTimer := benchmarkutil.Timer{}

	gpuMetrics.Start()
	gpuTimer.Start()

	labelsGPU, err := kmeansGPU.Predict(X)

	gpuTimeMS := gpuTimer.Stop()
	gpuMetrics.Stop()
	gpuMetrics.Close()

	if err != nil {
		t.Fatal(err)
	}

	gpuThroughput := float64(len(X)) / (gpuTimeMS / 1000.0)

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

	cpuMetrics, err := benchmarkutil.NewMetrics()
	if err != nil {
		t.Fatal(err)
	}

	cpuTimer := benchmarkutil.Timer{}

	cpuMetrics.Start()
	cpuTimer.Start()

	labelsCPU, err := kmeansCPU.Predict(X)

	cpuTimeMS := cpuTimer.Stop()
	cpuMetrics.Stop()
	cpuMetrics.Close()

	if err != nil {
		t.Fatal(err)
	}

	cpuThroughput := float64(len(X)) / (cpuTimeMS / 1000.0)

	// ---------------- inertia ----------------

	inertiaGPU := computeInertia(X, labelsGPU)
	inertiaCPU := computeInertia(X, labelsCPU)

	// ---------------- logs ----------------

	t.Logf("Samples         : %d", len(X))

	t.Logf(
		"GPU Throughput  : %.3f M samples/s",
		gpuThroughput/1e6,
	)

	t.Logf(
		"CPU Throughput  : %.3f M samples/s",
		cpuThroughput/1e6,
	)

	t.Logf(
		"GPU/CPU Speedup : %.2fx",
		gpuThroughput/cpuThroughput,
	)

	t.Logf("GPU Inertia     : %.6f", inertiaGPU)
	t.Logf("CPU Inertia     : %.6f", inertiaCPU)

	// ---------------- sanity ----------------

	if inertiaGPU == 0 {
		t.Fatalf("invalid GPU inertia")
	}

	if inertiaCPU == 0 {
		t.Fatalf("invalid CPU inertia")
	}

	if len(labelsGPU) != len(labelsCPU) {
		t.Fatalf(
			"label length mismatch GPU=%d CPU=%d",
			len(labelsGPU),
			len(labelsCPU),
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