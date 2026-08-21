#include "random_forest_cpu.h"

#include <mlpack.hpp>

#include <cstdint>
#include <iostream>
#include <memory>

struct RFCPUHandle {

    std::unique_ptr<mlpack::RandomForest<>> model;

    int n_estimators;
    int max_depth;
    float max_features;
    int max_leaves;
    float max_samples;

    int n_classes;
    int n_features;
};


RFCPUHandle* rf_cpu_create(
    int n_estimators,
    int max_depth,
    float max_features,
    int max_leaves,
    float max_samples
)
{
    try {

        auto* handle = new RFCPUHandle();

        handle->n_estimators = n_estimators;
        handle->max_depth = max_depth;
        handle->max_features = max_features;
        handle->max_leaves = max_leaves;
        handle->max_samples = max_samples;

        handle->n_classes = 0;
        handle->n_features = 0;

        return handle;

    }
    catch (const std::exception& e) {

        std::cerr
            << "rf_cpu_create: "
            << e.what()
            << std::endl;

        return nullptr;
    }
}


void rf_cpu_destroy(
    RFCPUHandle* handle
)
{
    delete handle;
}


int rf_cpu_fit(
    RFCPUHandle* handle,
    const float* X,
    int rows,
    int cols,
    const int* y,
    int n_classes
)
{
    if (!handle || !X || !y) {
        return -1;
    }

    try {

        handle->n_classes = n_classes;
        handle->n_features = cols;

        /*
         * mlpack uses Armadillo matrices in column-major
         * layout.
         *
         * Our Go/C API receives X in row-major layout:
         *
         * X[row * cols + col]
         *
         * Therefore construct the Armadillo matrix
         * explicitly.
         */

        arma::mat data(
            cols,
            rows,
            arma::fill::none
        );

        for (int r = 0; r < rows; ++r) {

            for (int c = 0; c < cols; ++c) {

                data(c, r) =
                    static_cast<double>(
                        X[r * cols + c]
                    );
            }
        }

        arma::Row<size_t> labels(rows);

        for (int r = 0; r < rows; ++r) {

            labels[r] =
                static_cast<size_t>(y[r]);
        }

        handle->model = std::make_unique<mlpack::RandomForest<>>();

        /*
         * Train the forest.
         *
         * Start with mlpack defaults for the parameters
         * that do not map directly to the existing cuML
         * configuration. The exact parameter mapping will
         * be handled separately.
         */

        handle->model->Train(
            data,
            labels,
            static_cast<size_t>(n_classes),
            static_cast<size_t>(handle->n_estimators)
        );

        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "rf_cpu_fit: "
            << e.what()
            << std::endl;

        return -1;
    }
}


int rf_cpu_predict(
    RFCPUHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* predictions
)
{
    if (!handle || !handle->model || !X || !predictions) {
        return -1;
    }

    try {

        arma::mat data(
            cols,
            rows,
            arma::fill::none
        );

        for (int r = 0; r < rows; ++r) {

            for (int c = 0; c < cols; ++c) {

                data(c, r) =
                    static_cast<double>(
                        X[r * cols + c]
                    );
            }
        }

        arma::Row<size_t> output;

        handle->model->Classify(
            data,
            output
        );

        for (int r = 0; r < rows; ++r) {

            predictions[r] =
                static_cast<int>(output[r]);
        }

        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "rf_cpu_predict: "
            << e.what()
            << std::endl;

        return -1;
    }
}