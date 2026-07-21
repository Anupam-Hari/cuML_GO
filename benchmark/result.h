#pragma once

#include <string>

struct BenchmarkResult
{
    std::string model;

    int train_rows;
    int test_rows;

    double train_time_ms;
    double predict_time_ms;
    double total_time_ms;

    float gpu_util_avg = 0.0f;
    float gpu_util_peak = 0.0f;
};