#include "benchmark.h"

#include "timer.h"

#include "../cpp/random_forest/random_forest.h"
#include "../cpp/knn/knn.h"
#include "../cpp/kmeans/kmeans.h"

#include <algorithm>
#include <vector>

namespace
{

struct TrainTestSplit
{
    std::vector<float> X_train;
    std::vector<int> y_train;

    std::vector<float> X_test;
    std::vector<int> y_test;

    int train_rows = 0;
    int test_rows = 0;
};

TrainTestSplit split_dataset(
    const Dataset& dataset,
    float train_ratio = 0.8f)
{
    TrainTestSplit split;

    split.train_rows =
        static_cast<int>(dataset.rows * train_ratio);

    split.test_rows =
        dataset.rows - split.train_rows;

    split.X_train.reserve(split.train_rows * dataset.cols);
    split.y_train.reserve(split.train_rows);

    split.X_test.reserve(split.test_rows * dataset.cols);
    split.y_test.reserve(split.test_rows);

    for (int i = 0; i < dataset.rows; i++)
    {
        const float* row =
            dataset.X.data() + i * dataset.cols;

        if (i < split.train_rows)
        {
            split.X_train.insert(
                split.X_train.end(),
                row,
                row + dataset.cols);

            split.y_train.push_back(dataset.y[i]);
        }
        else
        {
            split.X_test.insert(
                split.X_test.end(),
                row,
                row + dataset.cols);

            split.y_test.push_back(dataset.y[i]);
        }
    }

    return split;
}

int num_classes(const Dataset& dataset)
{
    int max_class = *std::max_element(
        dataset.y.begin(),
        dataset.y.end());

    return max_class + 1;
}

}

BenchmarkResult benchmark_random_forest(
    const Dataset& dataset)
{
    auto split = split_dataset(dataset);

    BenchmarkResult result;

    result.model = "Random Forest";
    result.train_rows = split.train_rows;
    result.test_rows = split.test_rows;

    RFHandle* rf = rf_create(
        100,
        10,
        1.0f,
        -1,
        1.0f);

    Timer timer;

    timer.start();

    rf_fit(
        rf,
        split.X_train.data(),
        split.train_rows,
        dataset.cols,
        split.y_train.data(),
        num_classes(dataset));

    result.train_time_ms = timer.stop();

    std::vector<int> predictions(split.test_rows);

    timer.start();

    rf_predict(
        rf,
        split.X_test.data(),
        split.test_rows,
        dataset.cols,
        predictions.data());

    result.predict_time_ms = timer.stop();

    result.total_time_ms =
        result.train_time_ms +
        result.predict_time_ms;

    rf_destroy(rf);

    return result;
}

BenchmarkResult benchmark_knn(
    const Dataset& dataset)
{
    auto split = split_dataset(dataset);

    BenchmarkResult result;

    result.model = "KNN";
    result.train_rows = split.train_rows;
    result.test_rows = split.test_rows;

    KNNHandle* knn = knn_create(5);

    Timer timer;

    timer.start();

    knn_fit(
        knn,
        split.X_train.data(),
        split.train_rows,
        dataset.cols,
        split.y_train.data());

    result.train_time_ms = timer.stop();

    std::vector<int> predictions(split.test_rows);

    timer.start();

    knn_predict(
        knn,
        split.X_test.data(),
        split.test_rows,
        dataset.cols,
        predictions.data());

    result.predict_time_ms = timer.stop();

    result.total_time_ms =
        result.train_time_ms +
        result.predict_time_ms;

    knn_destroy(knn);

    return result;
}

BenchmarkResult benchmark_kmeans(
    const Dataset& dataset)
{
    auto split = split_dataset(dataset);

    BenchmarkResult result;

    result.model = "KMeans";
    result.train_rows = split.train_rows;
    result.test_rows = split.test_rows;

    KMeansHandle* km = kmeans_create(
        num_classes(dataset),
        8,
        1e-4f);

    Timer timer;

    timer.start();

    kmeans_fit(
        km,
        split.X_train.data(),
        split.train_rows,
        dataset.cols);

    result.train_time_ms = timer.stop();

    std::vector<int> labels(split.test_rows);

    timer.start();

    kmeans_predict(
        km,
        split.X_test.data(),
        split.test_rows,
        dataset.cols,
        labels.data());

    result.predict_time_ms = timer.stop();

    result.total_time_ms =
        result.train_time_ms +
        result.predict_time_ms;

    kmeans_destroy(km);

    return result;
}