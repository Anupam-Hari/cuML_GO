#include "random_forest_cpu.h"

#include <mlpack.hpp>
#include <omp.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <fstream>
#include <string>

struct RFCPUHandle {

    std::unique_ptr<mlpack::RandomForest<>> model;

    int n_estimators;
    int max_depth;
    float max_features;
    int max_leaves;
    float max_samples;

    int n_classes;
    int n_features;
    int cpu_threads;
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
        handle->cpu_threads = 1;

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

void rf_set_cpu_threads_cpu(RFCPUHandle* handle, int threads)
{
    if (!handle || threads <= 0) {
        return;
    }

    handle->cpu_threads = threads;
}

int rf_cpu_save(
    RFCPUHandle* handle,
    const char* filename
)
{
    if (handle == nullptr || handle->model == nullptr) {
        std::cerr << "rf_cpu_save: invalid handle" << std::endl;
        return -1;
    }

    if (filename == nullptr) {
        std::cerr << "rf_cpu_save: filename is null" << std::endl;
        return -1;
    }

    try {

        // Save the trained mlpack model.
        bool success = mlpack::data::Save(
            filename,
            *handle->model
        );

        if (!success) {
            std::cerr
                << "rf_cpu_save: failed to save model"
                << std::endl;
            return -1;
        }

        // Save wrapper metadata.
        std::ofstream meta(
            std::string(filename) + ".meta"
        );

        if (!meta) {
            std::cerr
                << "rf_cpu_save: failed to create metadata file"
                << std::endl;
            return -1;
        }

        meta << "n_estimators " << handle->n_estimators << '\n';
        meta << "max_depth " << handle->max_depth << '\n';
        meta << "max_features " << handle->max_features << '\n';
        meta << "max_leaves " << handle->max_leaves << '\n';
        meta << "max_samples " << handle->max_samples << '\n';
        meta << "n_classes " << handle->n_classes << '\n';
        meta << "n_features " << handle->n_features << '\n';
        meta << "cpu_threads " << handle->cpu_threads << '\n';

        if (!meta) {
            std::cerr
                << "rf_cpu_save: failed while writing metadata"
                << std::endl;
            return -1;
        }

        meta.close();

        return 0;

    }
    catch (const std::exception& e) {

        std::cerr
            << "rf_cpu_save: "
            << e.what()
            << std::endl;

        return -1;
    }
}

RFCPUHandle* rf_cpu_load(
    const char* filename
)
{
    if (filename == nullptr) {
        std::cerr << "rf_cpu_load: filename is null" << std::endl;
        return nullptr;
    }

    try {

        auto* handle = new RFCPUHandle();

        handle->model =
            std::make_unique<mlpack::RandomForest<>>();

        bool success = mlpack::data::Load(
            filename,
            *handle->model
        );

        if (!success) {
            std::cerr
                << "rf_cpu_load: failed to load model"
                << std::endl;

            delete handle;
            return nullptr;
        }

        std::ifstream meta(
            std::string(filename) + ".meta"
        );

        if (!meta) {
            std::cerr
                << "rf_cpu_load: failed to open metadata file"
                << std::endl;

            delete handle;
            return nullptr;
        }

        std::string key;

        if (!(meta >> key >> handle->n_estimators) ||
            key != "n_estimators") {
            std::cerr << "rf_cpu_load: invalid n_estimators metadata" << std::endl;
            delete handle;
            return nullptr;
        }

        if (!(meta >> key >> handle->max_depth) ||
            key != "max_depth") {
            std::cerr << "rf_cpu_load: invalid max_depth metadata" << std::endl;
            delete handle;
            return nullptr;
        }

        if (!(meta >> key >> handle->max_features) ||
            key != "max_features") {
            std::cerr << "rf_cpu_load: invalid max_features metadata" << std::endl;
            delete handle;
            return nullptr;
        }

        if (!(meta >> key >> handle->max_leaves) ||
            key != "max_leaves") {
            std::cerr << "rf_cpu_load: invalid max_leaves metadata" << std::endl;
            delete handle;
            return nullptr;
        }

        if (!(meta >> key >> handle->max_samples) ||
            key != "max_samples") {
            std::cerr << "rf_cpu_load: invalid max_samples metadata" << std::endl;
            delete handle;
            return nullptr;
        }

        if (!(meta >> key >> handle->n_classes) ||
            key != "n_classes") {
            std::cerr << "rf_cpu_load: invalid n_classes metadata" << std::endl;
            delete handle;
            return nullptr;
        }

        if (!(meta >> key >> handle->n_features) ||
            key != "n_features") {
            std::cerr << "rf_cpu_load: invalid n_features metadata" << std::endl;
            delete handle;
            return nullptr;
        }

        if (!(meta >> key >> handle->cpu_threads) ||
            key != "cpu_threads") {
            std::cerr << "rf_cpu_load: invalid cpu_threads metadata" << std::endl;
            delete handle;
            return nullptr;
        }

        return handle;

    }
    catch (const std::exception& e) {

        std::cerr
            << "rf_cpu_load: "
            << e.what()
            << std::endl;

        return nullptr;
    }
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

        // printf(
        //     "[RF DEBUG] BEFORE Classify: omp_max=%d\n",
        //     omp_get_max_threads()
        // );

        omp_set_dynamic(0);
        omp_set_num_threads(handle->cpu_threads);

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