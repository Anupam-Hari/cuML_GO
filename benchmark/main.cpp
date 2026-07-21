#include "benchmark.h"
#include "utils/csv_writer.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "data/dataset_loader.h"

#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    int max_rows = -1;

    if (argc > 1)
        max_rows = std::stoi(argv[1]);

    Dataset dataset = load_dataset(
        "benchmark/data/processed_network_traffic.csv",
        "is_malicious",
        max_rows);

    std::vector<BenchmarkResult> results;

    std::cout << "Running Random Forest benchmark...\n";
    results.push_back(benchmark_random_forest(dataset));

    std::cout << "Running KNN benchmark...\n";
    results.push_back(benchmark_knn(dataset));

    std::cout << "Running KMeans benchmark...\n";
    results.push_back(benchmark_kmeans(dataset));

    std::cout << "\nResults\n";
    std::cout << "---------------------------------------------\n";

    for (const auto& result : results)
    {
        std::cout
            << result.model
            << "\n  Train Time   : "
            << result.train_time_ms << " ms"
            << "\n  Prediction Throughput : "
            << result.prediction_throughput << " ops/s"
            << "\n  Total Time   : "
            << result.total_time_ms << " ms\n\n";
    }

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm = *std::localtime(&t);

    std::ostringstream filename;
    filename << "benchmark/results/run_"
            << std::put_time(&tm, "%d%m%y%H%M%S")
            << ".csv";

    write_results_csv(
        filename.str(),
        results);

    std::cout << "Results written to " << filename.str() << "\n";

    return 0;
}