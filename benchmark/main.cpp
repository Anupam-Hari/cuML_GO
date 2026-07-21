#include "benchmark.h"
#include "utils/csv_writer.h"

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
            << "\n  Predict Time : "
            << result.predict_time_ms << " ms"
            << "\n  Total Time   : "
            << result.total_time_ms << " ms\n\n";
    }

    write_results_csv(
        "benchmark/results/benchmark_results.csv",
        results);

    std::cout << "Results written to benchmark_results.csv\n";

    return 0;
}