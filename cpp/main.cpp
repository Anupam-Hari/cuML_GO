#include <iostream>

#include "random_forest/random_forest.h"

int main()
{
    float X[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };

    int y[] = {
        0,
        0,
        1,
        1
    };

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
        X,
        4,      // rows
        2,      // cols
        y,
        2       // number of classes
    );

    int predictions[4];

    status = rf_predict(
        rf,
        X,
        4,
        2,
        predictions
    );

    std::cout << "rf_predict returned: "
              << status << std::endl;

    if (status == 0) {
        std::cout << "Predictions: ";

        for (int i = 0; i < 4; ++i)
             std::cout << predictions[i] << " ";

        std::cout << std::endl;
    }

    std::cout << "rf_fit returned: " << status << std::endl;

    rf_destroy(rf);

    return status;
}
