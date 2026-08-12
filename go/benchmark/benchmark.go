package main

import (
	"fmt"

	benchmarkutil "github.com/Anupam-Hari/cuml-go/go/internal/benchmark"
	"github.com/Anupam-Hari/cuml-go/go/kmeans"
    "github.com/Anupam-Hari/cuml-go/go/knn"
	"github.com/Anupam-Hari/cuml-go/go/random_forest"
)

func BenchmarkRandomForest(
	trainDataset benchmarkutil.Dataset,
	X [][]float32,
	y []int,
	predictSizes []int,
	repeats int,
) ([]BenchmarkResult, error) {

	results := []BenchmarkResult{}

	// ---------------------------------------------------------
	// CREATE GPU MODEL
	// ---------------------------------------------------------

	rf, err := randomforest.New(
		randomforest.WithEstimators(100),
		randomforest.WithMaxDepth(10),
		randomforest.WithMaxFeatures(1.0),
		randomforest.WithMaxLeaves(-1),
		randomforest.WithMaxSamples(1.0),
	)
	if err != nil {
		return results, err
	}
	defer rf.Close()

	// ---------------------------------------------------------
	// TRAIN ONCE
	//
	// Exactly 1,000,000 training rows are passed from main.
	// ---------------------------------------------------------

	fmt.Printf(
		"\nTraining Random Forest on %d rows...\n",
		trainDataset.Rows,
	)

	err = rf.Fit(
		trainDataset.X,
		trainDataset.Y,
	)
	if err != nil {
		return results, err
	}

	fmt.Println("Random Forest training complete.")

	// ---------------------------------------------------------
	// INFERENCE
	//
	// The trained model is reused for every prediction size.
	// ---------------------------------------------------------

	for _, predictRows := range predictSizes {

		if predictRows > len(X) {
			return results, fmt.Errorf(
				"prediction size %d exceeds available rows %d",
				predictRows,
				len(X),
			)
		}

		XPredict := X[:predictRows]
		yPredict := y[:predictRows]

		fmt.Printf(
			"\nRandom Forest GPU inference: %d rows\n",
			predictRows,
		)

		// -----------------------------------------------------
		// WARMUP
		//
		// These predictions are NOT measured.
		// -----------------------------------------------------

		for i := 0; i < 10; i++ {

			_, err = rf.Predict(
				XPredict,
				randomforest.BackendGPU,
			)

			if err != nil {
				return results, err
			}
		}

		fmt.Println("Warmup complete.")

		// -----------------------------------------------------
		// MEASURED INFERENCE
		// -----------------------------------------------------

		metrics, err := benchmarkutil.NewMetrics()
		if err != nil {
			return results, err
		}

		timer := benchmarkutil.Timer{}

		metrics.Start()
		timer.Start()

		var predictions []int

		for run := 0; run < repeats; run++ {

			predictions, err = rf.Predict(
				XPredict,
				randomforest.BackendGPU,
			)

			if err != nil {
				metrics.Stop()
				metrics.Close()
				return results, err
			}
		}

		totalTimeMS := timer.Stop()
		metrics.Stop()

		// -----------------------------------------------------
		// Average prediction time
		// -----------------------------------------------------

		avgPredictionTimeMS :=
			totalTimeMS / float64(repeats)

		// -----------------------------------------------------
		// Throughput
		// -----------------------------------------------------

		throughput :=
			float64(predictRows) /
				(avgPredictionTimeMS / 1000.0)

		// -----------------------------------------------------
		// Accuracy
		//
		// Only calculate this after timing is complete.
		// -----------------------------------------------------

		accuracy := computeAccuracy(
			predictions,
			yPredict,
		)

		// -----------------------------------------------------
		// CPU average
		//
		// This represents the CPU monitoring across ALL
		// measured inference repeats.
		// -----------------------------------------------------

		cpuAvg := metrics.CPUAverage()

		metrics.Close()

		// -----------------------------------------------------
		// Store result
		// -----------------------------------------------------

		results = append(
			results,
			BenchmarkResult{
				Model:               "Random Forest Inference",
				PredictRows:         predictRows,
				Backend:             "gpu",
				Runs:                repeats,
				AvgAccuracy:         accuracy,
				AvgPredictionTimeMS: avgPredictionTimeMS,
				AvgThroughput:       throughput,
				AvgCPU:              cpuAvg,
			},
		)

		fmt.Printf(
			"Prediction rows : %d\n",
			predictRows,
		)
		fmt.Printf(
			"Runs            : %d\n",
			repeats,
		)
		fmt.Printf(
			"Accuracy        : %.6f\n",
			accuracy,
		)
		fmt.Printf(
			"Prediction time : %.6f ms\n",
			avgPredictionTimeMS,
		)
		fmt.Printf(
			"Throughput      : %.6f samples/sec\n",
			throughput,
		)
		fmt.Printf(
			"CPU average     : %.6f%%\n",
			cpuAvg,
		)
	}

	return results, nil
}

func computeAccuracy(
	predictions []int,
	actual []int,
) float64 {

	if len(predictions) == 0 || len(predictions) != len(actual) {
		return 0.0
	}

	correct := 0

	for i := range predictions {
		if predictions[i] == actual[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(predictions))
}

func BenchmarkKMeans(
	trainDataset benchmarkutil.Dataset,
	X [][]float32,
	y []int,
	predictSizes []int,
	repeats int,
) ([]BenchmarkResult, error) {

	results := []BenchmarkResult{}

	// ---------------------------------------------------------
	// CREATE GPU MODEL
	// ---------------------------------------------------------

	km, err := kmeans.New(
		kmeans.WithBackend(kmeans.BackendGPU),
		kmeans.WithNClusters(8),
	)
	if err != nil {
		return results, err
	}
	defer km.Close()

	// ---------------------------------------------------------
	// TRAIN ONCE
	// ---------------------------------------------------------

	fmt.Printf(
		"\nTraining KMeans on %d rows...\n",
		trainDataset.Rows,
	)

	err = km.Fit(
		trainDataset.X,
	)
	if err != nil {
		return results, err
	}

	fmt.Println("KMeans training complete.")

	// ---------------------------------------------------------
	// INFERENCE
	// ---------------------------------------------------------

	for _, predictRows := range predictSizes {

		if predictRows > len(X) {
			return results, fmt.Errorf(
				"prediction size %d exceeds available rows %d",
				predictRows,
				len(X),
			)
		}

		XPredict := X[:predictRows]
		yPredict := y[:predictRows]

		fmt.Printf(
			"\nKMeans GPU inference: %d rows\n",
			predictRows,
		)

		// -----------------------------------------------------
		// WARMUP
		// -----------------------------------------------------

		for i := 0; i < 10; i++ {

			_, err = km.Predict(
				XPredict,
			)

			if err != nil {
				return results, err
			}
		}

		fmt.Println("Warmup complete.")

		// -----------------------------------------------------
		// MEASURED INFERENCE
		// -----------------------------------------------------

		metrics, err := benchmarkutil.NewMetrics()
		if err != nil {
			return results, err
		}

		timer := benchmarkutil.Timer{}

		metrics.Start()
		timer.Start()

		var predictions []int

		for run := 0; run < repeats; run++ {

			predictions, err = km.Predict(
				XPredict,
			)

			if err != nil {
				metrics.Stop()
				metrics.Close()
				return results, err
			}
		}

		totalTimeMS := timer.Stop()
		metrics.Stop()

		// -----------------------------------------------------
		// Average prediction time
		// -----------------------------------------------------

		avgPredictionTimeMS :=
			totalTimeMS / float64(repeats)

		// -----------------------------------------------------
		// Throughput
		// -----------------------------------------------------

		throughput :=
			float64(predictRows) /
				(avgPredictionTimeMS / 1000.0)

		// -----------------------------------------------------
		// Accuracy
		//
		// NOTE:
		// KMeans is unsupervised. Its cluster IDs are not
		// necessarily equivalent to attack_type labels.
		//
		// Do NOT use computeAccuracy() here unless your existing
		// KMeans implementation has a label-mapping scheme.
		// -----------------------------------------------------

		_ = predictions
		_ = yPredict

		accuracy := 0.0

		// -----------------------------------------------------
		// CPU average
		// -----------------------------------------------------

		cpuAvg := metrics.CPUAverage()

		metrics.Close()

		// -----------------------------------------------------
		// Store result
		// -----------------------------------------------------

		results = append(
			results,
			BenchmarkResult{
				Model:               "KMeans Inference",
				PredictRows:         predictRows,
				Backend:             "gpu",
				Runs:                repeats,
				AvgAccuracy:         accuracy,
				AvgPredictionTimeMS: avgPredictionTimeMS,
				AvgThroughput:       throughput,
				AvgCPU:              cpuAvg,
			},
		)

		fmt.Printf(
			"Prediction rows : %d\n",
			predictRows,
		)
		fmt.Printf(
			"Runs            : %d\n",
			repeats,
		)
		fmt.Printf(
			"Prediction time : %.6f ms\n",
			avgPredictionTimeMS,
		)
		fmt.Printf(
			"Throughput      : %.6f samples/sec\n",
			throughput,
		)
		fmt.Printf(
			"CPU average     : %.6f%%\n",
			cpuAvg,
		)
	}

	return results, nil
}

func BenchmarkKNN(
	trainDataset benchmarkutil.Dataset,
	X [][]float32,
	y []int,
	predictSizes []int,
	repeats int,
) ([]BenchmarkResult, error) {

	results := []BenchmarkResult{}

	// ---------------------------------------------------------
	// CREATE GPU MODEL
	// ---------------------------------------------------------

	knnModel, err := knn.New(
		knn.WithBackend(knn.BackendGPU),
		knn.WithK(5),
	)
	if err != nil {
		return results, err
	}
	defer knnModel.Close()

	// ---------------------------------------------------------
	// TRAIN ONCE
	// ---------------------------------------------------------

	fmt.Printf(
		"\nTraining KNN on %d rows...\n",
		trainDataset.Rows,
	)

	err = knnModel.Fit(
		trainDataset.X,
		trainDataset.Y,
	)
	if err != nil {
		return results, err
	}

	fmt.Println("KNN training complete.")

	// ---------------------------------------------------------
	// INFERENCE
	// ---------------------------------------------------------

	for _, predictRows := range predictSizes {

		if predictRows > len(X) {
			return results, fmt.Errorf(
				"prediction size %d exceeds available rows %d",
				predictRows,
				len(X),
			)
		}

		XPredict := X[:predictRows]
		yPredict := y[:predictRows]

		fmt.Printf(
			"\nKNN GPU inference: %d rows\n",
			predictRows,
		)

		// -----------------------------------------------------
		// WARMUP
		// -----------------------------------------------------

		for i := 0; i < 10; i++ {

			_, err = knnModel.Predict(
				XPredict,
			)

			if err != nil {
				return results, err
			}
		}

		fmt.Println("Warmup complete.")

		// -----------------------------------------------------
		// MEASURED INFERENCE
		// -----------------------------------------------------

		metrics, err := benchmarkutil.NewMetrics()
		if err != nil {
			return results, err
		}

		timer := benchmarkutil.Timer{}

		metrics.Start()
		timer.Start()

		var predictions []int

		for run := 0; run < repeats; run++ {

			predictions, err = knnModel.Predict(
				XPredict,
			)

			if err != nil {
				metrics.Stop()
				metrics.Close()
				return results, err
			}
		}

		totalTimeMS := timer.Stop()
		metrics.Stop()

		// -----------------------------------------------------
		// Average prediction time
		// -----------------------------------------------------

		avgPredictionTimeMS :=
			totalTimeMS / float64(repeats)

		// -----------------------------------------------------
		// Throughput
		// -----------------------------------------------------

		throughput :=
			float64(predictRows) /
				(avgPredictionTimeMS / 1000.0)

		// -----------------------------------------------------
		// Accuracy
		// -----------------------------------------------------

		accuracy := computeAccuracy(
			predictions,
			yPredict,
		)

		// -----------------------------------------------------
		// CPU average across all measured repeats
		// -----------------------------------------------------

		cpuAvg := metrics.CPUAverage()

		metrics.Close()

		// -----------------------------------------------------
		// Store result
		// -----------------------------------------------------

		results = append(
			results,
			BenchmarkResult{
				Model:               "KNN Inference",
				PredictRows:         predictRows,
				Backend:             "gpu",
				Runs:                repeats,
				AvgAccuracy:         accuracy,
				AvgPredictionTimeMS: avgPredictionTimeMS,
				AvgThroughput:       throughput,
				AvgCPU:              cpuAvg,
			},
		)

		fmt.Printf(
			"Prediction rows : %d\n",
			predictRows,
		)
		fmt.Printf(
			"Runs            : %d\n",
			repeats,
		)
		fmt.Printf(
			"Accuracy        : %.6f\n",
			accuracy,
		)
		fmt.Printf(
			"Prediction time : %.6f ms\n",
			avgPredictionTimeMS,
		)
		fmt.Printf(
			"Throughput      : %.6f samples/sec\n",
			throughput,
		)
		fmt.Printf(
			"CPU average     : %.6f%%\n",
			cpuAvg,
		)
	}

	return results, nil
}