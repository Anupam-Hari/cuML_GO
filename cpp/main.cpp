#include <iostream>
#include <vector>

#include "knn/knn.h"
#include "data/dataset_loader.h"

void test_knn()
{
    constexpr int rows = 150;
    constexpr int cols = 4;

    Dataset dataset = load_dataset("benchmark/data/processed_network_traffic.csv", "is_malicious");

    KNNHandle* knn = knn_create(5);

    int status = knn_fit(
        knn,
        dataset.X.data(),
        dataset.rows,
        dataset.cols,
        dataset.y.data());

    std::cout << "knn_fit returned: "
              << status
              << std::endl;

    std::vector<int> predictions(rows);

    status = knn_predict(
        knn,
        dataset.X.data(),
        dataset.rows,
        dataset.cols,
        predictions.data());

    std::cout << "knn_predict returned: "
              << status
              << std::endl;

    int correct = 0;

    for (int i = 0; i < dataset.rows; ++i)
    {
        if (predictions[i] == dataset.y[i])
            ++correct;
    }

    std::cout << "Accuracy: "
              << correct
              << "/"
              << rows
              << " = "
              << (100.0 * correct / rows)
              << "%"
              << std::endl;


    knn_destroy(knn);
}

int main()
{
    test_knn();
    return 0;
}