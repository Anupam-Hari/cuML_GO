#include <iostream>
#include <vector>

#include "random_forest/random_forest.h"
#include "dataset/dataset.h"

int main()
{
    Dataset dataset = load_csv("benchmark/data/iris.csv");

    std::cout << "Loaded dataset\n";
    std::cout << "Rows    : " << dataset.rows << '\n';
    std::cout << "Cols    : " << dataset.cols << '\n';
    std::cout << "Classes : " << dataset.classes << '\n';
    
    RFHandle* rf = rf_create(
        10,     // n_estimators
        5,      // max_depth
        1.0f,   // max_features
        -1,     // max_leaves
        1.0f    // max_samples
    );

    if (!rf) {
        std::cerr << "Failed to create RF\n";
        return 1;
    }

    int status = rf_fit(
        rf,
        dataset.X.data(),
        dataset.rows,
        dataset.cols,
        dataset.y.data(),
        dataset.classes
    );

    std::vector<int> predictions(dataset.rows);

    status = rf_predict(
        rf,
        dataset.X.data(),
        dataset.rows,
        dataset.cols,
        predictions.data()
    );

    std::cout << "rf_predict returned: "
              << status << std::endl;

    if (status == 0) {
        std::cout << "Predictions: ";

        for (int prediction:predictions)
             std::cout << prediction << " ";

        std::cout << std::endl;
    }

    int correct = 0;

    for (int i = 0; i < dataset.rows; ++i) {
        if (predictions[i] == dataset.y[i]) {
            ++correct;
        }
    }

    float accuracy =
        static_cast<float>(correct) / dataset.rows;

    std::cout << "\nTraining Accuracy: "
              << accuracy * 100.0f
              << "%\n";

    std::cout << "rf_fit returned: " << status << std::endl;

    rf_destroy(rf);

    return status;
}
