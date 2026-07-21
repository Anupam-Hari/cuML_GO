#pragma once

#include "data/dataset.h"
#include "result.h"

BenchmarkResult benchmark_random_forest(
    const Dataset& dataset);

BenchmarkResult benchmark_knn(
    const Dataset& dataset);

BenchmarkResult benchmark_kmeans(
    const Dataset& dataset);