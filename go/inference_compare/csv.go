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
	Backend string
}

type summaryValue struct {
	Runs int 
	Accuracy float64 
	PredictionTime float64 
	Throughput float64 
	CPUAvg float64
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
		"Backend", 
		"PredictRows", 
		"Accuracy", 
		"PredictionTime(ms)", 
		"Throughput(samples/sec)", 
		"CPUAvg(%)",
	}

	if err := writer.Write(header); err != nil {
		file.Close()
		return err
	}

	for _, r := range results {

		record := []string{
			r.Model, 
			strconv.Itoa(r.Run), 
			r.Backend, 
			strconv.Itoa(r.PredictRows), 
			fmt.Sprintf("%.6f", r.Accuracy), 
			fmt.Sprintf("%.6f", r.PredictionTimeMS), 
			fmt.Sprintf("%.6f", r.Throughput), 
			fmt.Sprintf("%.6f", r.CPUAvg), 
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

		predictRows, err := strconv.Atoi(row[3])
		if err != nil { 
			return fmt.Errorf("invalid PredictRows value %q: %w", row[3], err) 
		}

		key := summaryKey{
			Model:       row[0],
			PredictRows: predictRows,
			Backend: row[2],
		}

		if _, ok := summary[key]; !ok {
			summary[key] = &summaryValue{}
		}

		s := summary[key]

		s.Runs++

		accuracy, _ := strconv.ParseFloat(row[4], 64)

		predictionTime, _ := strconv.ParseFloat(row[5], 64)
		throughput, _ := strconv.ParseFloat(row[6], 64)

		cpuAvg, _ := strconv.ParseFloat(row[7], 64)

		s.Accuracy += accuracy

		s.PredictionTime += predictionTime 
		s.Throughput += throughput 
		s.CPUAvg += cpuAvg

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
		"Backend", 
		"Runs", 
		"AvgAccuracy", 
		"AvgPredictionTime(ms)", 
		"AvgThroughput(samples/sec)", 
		"AvgCPU(%)",
	})

	for _, key := range keys {

		s := summary[key]
		runs := float64(s.Runs)

		record := []string{
			key.Model, 
			strconv.Itoa(key.PredictRows), 
			key.Backend, strconv.Itoa(s.Runs), 
			fmt.Sprintf("%.6f", s.Accuracy/runs), 
			fmt.Sprintf("%.6f", s.PredictionTime/runs), 
			fmt.Sprintf("%.6f", s.Throughput/runs), 
			fmt.Sprintf("%.6f", s.CPUAvg/runs),
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