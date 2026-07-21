#include "csv_writer.h"

#include <fstream>

void write_results_csv(
    const std::string& filename,
    const std::vector<BenchmarkResult>& results)
{
    std::ofstream out(filename);

    out << "Model,"
        << "TrainRows,"
        << "TestRows,"
        << "TrainTime(ms),"
        << "PredictTime(ms),"
        << "TotalTime(ms),"
        << "GPUAvg,"
        << "GPUPeak\n";

    for (const auto& r : results)
    {
        out << r.model << ","
            << r.train_rows << ","
            << r.test_rows << ","
            << r.train_time_ms << ","
            << r.predict_time_ms << ","
            << r.total_time_ms << ","
            << r.gpu_util_avg << ","
            << r.gpu_util_peak
            << '\n';
    }
}