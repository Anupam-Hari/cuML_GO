package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"log"
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

const pythonSummaryFile = "/home/anupam/projects/cuml-go/go/inference_compare/python_inference_summary_ogdata.csv"

func comparePythonGoThroughput(goSummaryFile string) {

	const pythonSummaryFile = "/home/anupam/projects/cuml-go/go/inference_compare/python_inference_summary_ogdata.csv"

	type rowData struct {
		model      string
		rows       int
		throughput float64
	}

	// ------------------------------------------------------------
	// Helper to read a summary CSV
	// ------------------------------------------------------------

	readSummary := func(filename string) ([]rowData, error) {

		file, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		defer file.Close()

		reader := csv.NewReader(file)

		records, err := reader.ReadAll()
		if err != nil {
			return nil, err
		}

		if len(records) < 2 {
			return nil, fmt.Errorf("CSV contains no data: %s", filename)
		}

		header := records[0]

		modelIdx := -1
		rowsIdx := -1
		backendIdx := -1
		throughputIdx := -1

		for i, col := range header {
			switch strings.TrimSpace(col) {
			case "Model":
				modelIdx = i
			case "Backend":
				backendIdx = i
			case "PredictRows":
				rowsIdx = i
			case "AvgThroughput(samples/sec)":
				throughputIdx = i
			}
		}

		if modelIdx == -1 ||
			rowsIdx == -1 ||
			throughputIdx == -1 {

			return nil, fmt.Errorf(
				"missing required columns in %s",
				filename,
			)
		}

		var results []rowData

		for _, record := range records[1:] {

			if len(record) <= throughputIdx {
				continue
			}

			// Only CPU rows.
			if backendIdx != -1 &&
				strings.ToLower(strings.TrimSpace(record[backendIdx])) != "cpu" {
				continue
			}

			model := strings.ToLower(
				strings.TrimSpace(record[modelIdx]),
			)

			rows, err := strconv.Atoi(
				strings.TrimSpace(record[rowsIdx]),
			)
			if err != nil {
				continue
			}

			throughput, err := strconv.ParseFloat(
				strings.TrimSpace(record[throughputIdx]),
				64,
			)
			if err != nil {
				continue
			}

			results = append(results, rowData{
				model:      model,
				rows:       rows,
				throughput: throughput,
			})
		}

		return results, nil
	}

	// ------------------------------------------------------------
	// Read Python and Go summaries
	// ------------------------------------------------------------

	pythonResults, err := readSummary(pythonSummaryFile)
	if err != nil {
		log.Printf(
			"Could not read Python summary: %v",
			err,
		)
		return
	}

	goResults, err := readSummary(goSummaryFile)
	if err != nil {
		log.Printf(
			"Could not read Go summary: %v",
			err,
		)
		return
	}

	// ------------------------------------------------------------
	// Index Python results by model + rows
	// ------------------------------------------------------------

	pythonByKey := make(map[string]float64)

	for _, r := range pythonResults {

		key := fmt.Sprintf(
			"%s:%d",
			r.model,
			r.rows,
		)

		pythonByKey[key] = r.throughput
	}

	// ------------------------------------------------------------
	// Comparison
	// ------------------------------------------------------------

	fmt.Println()
	fmt.Println("==============================================================")
	fmt.Println("PYTHON vs GO — CPU INFERENCE THROUGHPUT")
	fmt.Println("==============================================================")

	fmt.Printf(
		"%-22s %10s %15s %15s %12s\n",
		"Model",
		"Rows",
		"Python M/s",
		"Go M/s",
		"Speedup",
	)

	fmt.Println(
		"----------------------------------------------------------------",
	)

	type stats struct {
		pythonTotal float64
		goTotal     float64
		count       int
	}

	modelStats := make(map[string]*stats)

	for _, goRow := range goResults {

		key := fmt.Sprintf(
			"%s:%d",
			goRow.model,
			goRow.rows,
		)

		pythonThroughput, ok := pythonByKey[key]
		if !ok {
			continue
		}

		if pythonThroughput <= 0 {
			continue
		}

		speedup := goRow.throughput / pythonThroughput

		fmt.Printf(
			"%-22s %10d %15.2f %15.2f %11.2fx\n",
			goRow.model,
			goRow.rows,
			pythonThroughput/1e6,
			goRow.throughput/1e6,
			speedup,
		)

		if modelStats[goRow.model] == nil {
			modelStats[goRow.model] = &stats{}
		}

		modelStats[goRow.model].pythonTotal += pythonThroughput
		modelStats[goRow.model].goTotal += goRow.throughput
		modelStats[goRow.model].count++
	}

	// ------------------------------------------------------------
	// Average + PASS/FAIL
	// ------------------------------------------------------------

	fmt.Println()
	fmt.Println("==============================================================")
	fmt.Println("AVERAGE CPU THROUGHPUT / SPEEDUP")
	fmt.Println("==============================================================")

	modelOrder := []string{
		"knn inference",
		"kmeans inference",
		"random forest inference",
	}

	for _, model := range modelOrder {

		s, ok := modelStats[model]

		if !ok || s.count == 0 {
			fmt.Printf(
				"%-22s NO MATCHING DATA\n",
				model,
			)
			continue
		}

		avgPython := s.pythonTotal / float64(s.count)
		avgGo := s.goTotal / float64(s.count)

		avgSpeedup := avgGo / avgPython

		status := "FAIL"

		if avgSpeedup > 1.0 {
			status = "PASS"
		}

		displayName := model

		switch model {
		case "knn inference":
			displayName = "KNN"
		case "kmeans inference":
			displayName = "KMeans"
		case "random forest inference":
			displayName = "Random Forest"
		}

		fmt.Printf(
			"%-22s %s — %.2fx speedup\n",
			displayName,
			status,
			avgSpeedup,
		)
	}

	fmt.Println("==============================================================")
	fmt.Println()
}