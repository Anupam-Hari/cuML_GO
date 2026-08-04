package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"
)

type summaryKey struct {
	Model     string
	TrainRows int
}

type summaryValue struct {
	TestRows int
	Count    int

	TrainTime float64
	Throughput float64
	TotalTime float64
	Accuracy float64
	CPUAvg float64
	CPUPeak float64
	GPUAvg float64
	GPUPeak float64
}

func WriteResultsCSV(
	filename string,
	results []BenchmarkResult,
) error {

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{
		"Model",
		"Run",
		"TrainRows",
		"TestRows",
		"TrainTime(ms)",
		"PredictionThroughput(ops)",
		"TotalTime(ms)",
		"Accuracy",
		"CPUAvg",
		"CPUPeak",
		"GPUAvg",
		"GPUPeak",
	}

	if err := writer.Write(header); err != nil {
		return err
	}

	for _, r := range results {

		record := []string{
			r.Model,
			fmt.Sprintf("%d", r.Run),
			fmt.Sprintf("%d", r.TrainRows),
			fmt.Sprintf("%d", r.TestRows),
			fmt.Sprintf("%.3f", r.TrainTimeMS),
			fmt.Sprintf("%.3f", r.PredictionThroughput),
			fmt.Sprintf("%.3f", r.TotalTimeMS),
			fmt.Sprintf("%.6f", r.Accuracy),
			fmt.Sprintf("%.3f", r.CPUAvg),
			fmt.Sprintf("%.3f", r.CPUPeak),
			fmt.Sprintf("%.3f", r.GPUAvg),
			fmt.Sprintf("%.3f", r.GPUPeak),
		}

		if err := writer.Write(record); err != nil {
			return err
		}
	}

	writer.Flush()

	return writer.Error()

}

func WriteSummaryCSV(rawCSV, summaryCSV string) error {

	file, err := os.Open(rawCSV)
	if err != nil {
		return err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	rows, err := reader.ReadAll()
	if err != nil {
		return err
	}

	summary := make(map[summaryKey]*summaryValue)

	for i, row := range rows {
		if i == 0 {
			continue
		}

		trainRows, _ := strconv.Atoi(row[2])
		testRows, _ := strconv.Atoi(row[3])

		trainTime, _ := strconv.ParseFloat(row[4], 64)
		throughput, _ := strconv.ParseFloat(row[5], 64)
		totalTime, _ := strconv.ParseFloat(row[6], 64)
		accuracy, _ := strconv.ParseFloat(row[7], 64)
		cpuAvg, _ := strconv.ParseFloat(row[8], 64)
		cpuPeak, _ := strconv.ParseFloat(row[9], 64)
		gpuAvg, _ := strconv.ParseFloat(row[10], 64)
		gpuPeak, _ := strconv.ParseFloat(row[11], 64)

		key := summaryKey{
			Model: row[0],
			TrainRows: trainRows,
		}

		if _, ok := summary[key]; !ok {
			summary[key] = &summaryValue{
				TestRows: testRows,
			}
		}

		s := summary[key]

		s.Count++

		s.TrainTime += trainTime
		s.Throughput += throughput
		s.TotalTime += totalTime
		s.Accuracy += accuracy
		s.CPUAvg += cpuAvg
		s.CPUPeak += cpuPeak
		s.GPUAvg += gpuAvg
		s.GPUPeak += gpuPeak
	}

	keys := make([]summaryKey, 0, len(summary))
	for k := range summary {
		keys = append(keys, k)
	}

	modelOrder := map[string]int{
		"Random Forest":          0,
		"GoLearn Random Forest":  1,
		"KNN":                    2,
		"KMeans":                 3,
	}

	sort.Slice(keys, func(i, j int) bool {

		if keys[i].TrainRows != keys[j].TrainRows {
			return keys[i].TrainRows < keys[j].TrainRows
		}

		return modelOrder[keys[i].Model] < modelOrder[keys[j].Model]
	})

	out, err := os.Create(summaryCSV)
	if err != nil {
		return err
	}
	defer out.Close()

	writer := csv.NewWriter(out)
	defer writer.Flush()

	writer.Write([]string{
		"Model",
		"TrainRows",
		"TestRows",
		"Runs",
		"AvgTrainTime(ms)",
		"AvgPredictionThroughput(ops)",
		"AvgTotalTime(ms)",
		"AvgAccuracy",
		"AvgCPUAvg",
		"AvgCPUPeak",
		"AvgGPUAvg",
		"AvgGPUPeak",
	})

	for _, key := range keys {

		s := summary[key]

		record := []string{
			key.Model,
			fmt.Sprintf("%d", key.TrainRows),
			fmt.Sprintf("%d", s.TestRows),
			fmt.Sprintf("%d", s.Count),
			fmt.Sprintf("%.3f", s.TrainTime/float64(s.Count)),
			fmt.Sprintf("%.3f", s.Throughput/float64(s.Count)),
			fmt.Sprintf("%.3f", s.TotalTime/float64(s.Count)),
			fmt.Sprintf("%.6f", s.Accuracy/float64(s.Count)),
			fmt.Sprintf("%.3f", s.CPUAvg/float64(s.Count)),
			fmt.Sprintf("%.3f", s.CPUPeak/float64(s.Count)),
			fmt.Sprintf("%.3f", s.GPUAvg/float64(s.Count)),
			fmt.Sprintf("%.3f", s.GPUPeak/float64(s.Count)),
		}

		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return writer.Error()
}