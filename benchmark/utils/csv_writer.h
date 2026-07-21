#pragma once

#include <string>
#include <vector>

#include "result.h"

void write_results_csv(
    const std::string& filename,
    const std::vector<BenchmarkResult>& results);