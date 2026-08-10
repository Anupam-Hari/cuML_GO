package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

type summaryKey struct {
	Model       string
	PredictRows int
}

type summaryValue struct {
	Runs int

	Accuracy float64

	CPUPredTime float64
	GPUPredTime float64

	CPUThroughput float64
	GPUThroughput float64

	CPURunCPUAvg  float64
	CPURunCPUPeak float64

	GPURunCPUAvg  float64
	GPURunCPUPeak float64

	CPUCores int
}

func WriteResultsCSV(
	filename string,
	results []BenchmarkResult,
) error {

	if err := os.MkdirAll(filepath.Dir(filename), 0755); err != nil {
		return err
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)

	header := []string{
		"Model",
		"Run",
		"PredictRows",
		"Accuracy",
		"CPUPredictionTime(ms)",
		"GPUPredictionTime(ms)",
		"CPUThroughput(samples/sec)",
		"GPUThroughput(samples/sec)",
		"CPURunCPUAvg(%)",
		"CPURunCPUPeak(%)",
		"GPURunCPUAvg(%)",
		"GPURunCPUPeak(%)",
		"CPUCores",
	}

	if err := writer.Write(header); err != nil {
		file.Close()
		return err
	}

	for _, r := range results {

		record := []string{
			r.Model,
			strconv.Itoa(r.Run),
			strconv.Itoa(r.PredictRows),

			fmt.Sprintf("%.6f", r.Accuracy),

			fmt.Sprintf("%.6f", r.CPUPredictionTimeMS),
			fmt.Sprintf("%.6f", r.GPUPredictionTimeMS),

			fmt.Sprintf("%.6f", r.CPUThroughput),
			fmt.Sprintf("%.6f", r.GPUThroughput),

			fmt.Sprintf("%.6f", r.CPURunCPUAvg),
			fmt.Sprintf("%.6f", r.CPURunCPUPeak),

			fmt.Sprintf("%.6f", r.GPURunCPUAvg),
			fmt.Sprintf("%.6f", r.GPURunCPUPeak),

			strconv.Itoa(r.CPUCores),
		}

		if err := writer.Write(record); err != nil {
			file.Close()
			return err
		}
	}

	writer.Flush()

	if err := writer.Error(); err != nil {
		file.Close()
		return err
	}

	return file.Close()
}

func WriteSummaryCSV(
	rawCSV string,
	summaryCSV string,
) error {

	if err := os.MkdirAll(filepath.Dir(summaryCSV), 0755); err != nil {
		return err
	}

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

		predictRows, _ := strconv.Atoi(row[2])

		key := summaryKey{
			Model:       row[0],
			PredictRows: predictRows,
		}

		if _, ok := summary[key]; !ok {
			summary[key] = &summaryValue{}
		}

		s := summary[key]

		s.Runs++

		accuracy, _ := strconv.ParseFloat(row[3], 64)

		cpuPred, _ := strconv.ParseFloat(row[4], 64)
		gpuPred, _ := strconv.ParseFloat(row[5], 64)

		cpuThr, _ := strconv.ParseFloat(row[6], 64)
		gpuThr, _ := strconv.ParseFloat(row[7], 64)

		cpuRunAvg, _ := strconv.ParseFloat(row[8], 64)
		cpuRunPeak, _ := strconv.ParseFloat(row[9], 64)

		gpuRunAvg, _ := strconv.ParseFloat(row[10], 64)
		gpuRunPeak, _ := strconv.ParseFloat(row[11], 64)

		cpuCores, _ := strconv.Atoi(row[12])

		s.Accuracy += accuracy

		s.CPUPredTime += cpuPred
		s.GPUPredTime += gpuPred

		s.CPUThroughput += cpuThr
		s.GPUThroughput += gpuThr

		s.CPURunCPUAvg += cpuRunAvg

		if cpuRunPeak > s.CPURunCPUPeak {
			s.CPURunCPUPeak = cpuRunPeak
		}

		s.GPURunCPUAvg += gpuRunAvg

		if gpuRunPeak > s.GPURunCPUPeak {
			s.GPURunCPUPeak = gpuRunPeak
		}

		s.CPUCores = cpuCores
	}

	keys := make([]summaryKey, 0, len(summary))
	for k := range summary {
		keys = append(keys, k)
	}

	sort.Slice(keys, func(i, j int) bool {

		if keys[i].PredictRows != keys[j].PredictRows {
			return keys[i].PredictRows < keys[j].PredictRows
		}

		return keys[i].Model < keys[j].Model
	})

	out, err := os.Create(summaryCSV)
	if err != nil {
		return err
	}
	defer out.Close()

	writer := csv.NewWriter(out)

	writer.Write([]string{
		"Model",
		"PredictRows",
		"Runs",
		"AvgAccuracy",
		"AvgCPUPredictionTime(ms)",
		"AvgGPUPredictionTime(ms)",
		"AvgCPUThroughput(samples/sec)",
		"AvgGPUThroughput(samples/sec)",
		"AvgCPURunCPUAvg(%)",
		"MaxCPURunCPUPeak(%)",
		"AvgGPURunCPUAvg(%)",
		"MaxGPURunCPUPeak(%)",
		"CPUCores",
	})

	for _, key := range keys {

		s := summary[key]

		record := []string{
			key.Model,
			strconv.Itoa(key.PredictRows),
			strconv.Itoa(s.Runs),

			fmt.Sprintf("%.6f", s.Accuracy/float64(s.Runs)),

			fmt.Sprintf("%.6f", s.CPUPredTime/float64(s.Runs)),
			fmt.Sprintf("%.6f", s.GPUPredTime/float64(s.Runs)),

			fmt.Sprintf("%.6f", s.CPUThroughput/float64(s.Runs)),
			fmt.Sprintf("%.6f", s.GPUThroughput/float64(s.Runs)),

			fmt.Sprintf("%.6f", s.CPURunCPUAvg/float64(s.Runs)),
			fmt.Sprintf("%.6f", s.CPURunCPUPeak),

			fmt.Sprintf("%.6f", s.GPURunCPUAvg/float64(s.Runs)),
			fmt.Sprintf("%.6f", s.GPURunCPUPeak),

			strconv.Itoa(s.CPUCores),
		}

		if err := writer.Write(record); err != nil {
			return err
		}
	}

	writer.Flush()

	if err := writer.Error(); err != nil {
		return err
	}

	return out.Close()
}