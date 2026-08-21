#include <omp.h>
#include "random_forest.h"

#include "../common/cuda_utils.hpp"
#include <cuml/ensemble/randomforest.hpp>
#include <raft/core/handle.hpp>
#include <cmath>
#include <rmm/cuda_stream_pool.hpp>
#include <nvforest/treelite_importer.hpp>
#include <nvforest/forest_model.hpp>
#include <treelite/c_api.h>
#include <optional>
#include <vector>
#include <cuda_runtime.h>
#include <memory>
#include <fstream>
#include <string>

struct RFHandle {
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
    raft::handle_t handle;

    ML::RF_params params;
    TreeliteModelHandle tl_model = nullptr;
    std::optional<nvforest::forest_model> cpu_forest;
    std::optional<nvforest::forest_model> gpu_forest;

    int n_classes = 0;
};

RFHandle* rf_create(
    int n_estimators,
    int max_depth,
    float max_features,
    int max_leaves,
    float max_samples)
{
    auto* rf = new RFHandle;

    rf->stream_pool = std::make_shared<rmm::cuda_stream_pool>(4);

    new (&rf->handle) raft::handle_t(
        rmm::cuda_stream_per_thread,
        rf->stream_pool
    );

    rf->params = ML::set_rf_params(
        max_depth,
        max_leaves,
        max_features,
        128,
        1,
        2,
        0.0f,
        true,
        n_estimators,
        max_samples,
        42,                     // same random_state
        ML::CRITERION::GINI,
        4,                      // same n_streams
        4096
    );

    return rf;
}

void rf_set_cpu_threads(int threads)
{
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
}

int rf_get_cpu_threads()
{
    return omp_get_max_threads();
}

int rf_fit(
    RFHandle* handle,
    const float* X,
    int rows,
    int cols,
    const int* y,
    int n_classes
)
{
    float* d_X = nullptr;
    int* d_y = nullptr;
    handle->n_classes = n_classes;
    int device;
    cudaGetDevice(&device);

    try
    {
        d_X = static_cast<float*>(
            cuda_utils::Malloc(rows * cols * sizeof(float)));

        d_y = static_cast<int*>(
            cuda_utils::Malloc(rows * sizeof(int)));

        std::vector<float> X_col(rows * cols);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                X_col[c * rows + r] = X[r * cols + c];
            }
        }


        cuda_utils::CopyHostToDevice(
            d_X,
            X_col.data(),
            rows * cols * sizeof(float));

        cuda_utils::CopyHostToDevice(
            d_y,
            y,
            rows * sizeof(int));

        // Match Python cuML's max_features="sqrt"
        handle->params.tree_params.max_features =
            std::sqrt(static_cast<float>(cols)) /
            static_cast<float>(cols);

        float* feature_importances = nullptr;

        ML::fit_treelite(
            handle->handle,
            &handle->tl_model,
            d_X,
            rows,
            cols,
            d_y,
            n_classes,
            handle->params,
            nullptr,
            feature_importances,
            rapids_logger::level_enum::info
        );

        handle->cpu_forest.emplace(
            nvforest::import_from_treelite_handle(
                handle->tl_model,
                nvforest::preferred_tree_layout,
                0,
                std::nullopt,
                raft_proto::device_type::cpu,
                device,
                handle->handle.get_stream().value()
            )
        );

        handle->gpu_forest.emplace(
            nvforest::import_from_treelite_handle(
                handle->tl_model,
                nvforest::preferred_tree_layout,
                0,
                std::nullopt,
                raft_proto::device_type::gpu,
                device,
                handle->handle.get_stream().value()
            )
        );

        cuda_utils::Free(d_X);
        cuda_utils::Free(d_y);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "rf_fit exception: "
                  << e.what()
                  << std::endl;

        cuda_utils::Free(d_X);
        cuda_utils::Free(d_y);

        return -1;
    }   
    catch (...)
    {
        std::cerr << "rf_fit unknown exception"
                  << std::endl;

        cuda_utils::Free(d_X);
        cuda_utils::Free(d_y);

        return -1;
    }
}

int rf_predict(
    RFHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* predictions,
    int backend)
{
    float* d_X = nullptr;
    float* d_predictions = nullptr;

    try
    {
        auto* forest =
            (backend == 0)
                ? &handle->cpu_forest
                : &handle->gpu_forest;

        if (!forest->has_value()) {
            return -1;
        }

        std::vector<float> h_predictions(rows * handle->n_classes);

        if (backend == 1) {

            d_X = static_cast<float*>(
                cuda_utils::Malloc(rows * cols * sizeof(float)));

            d_predictions = static_cast<float*>(
                cuda_utils::Malloc(rows * handle->n_classes * sizeof(float)));

            cuda_utils::CopyHostToDevice(
                d_X,
                X,
                rows * cols * sizeof(float));

            forest->value().predict(
                raft_proto::handle_t(handle->handle),
                d_predictions,
                d_X,
                rows,
                raft_proto::device_type::gpu,
                raft_proto::device_type::gpu
            );

            cuda_utils::CopyDeviceToHost(
                h_predictions.data(),
                d_predictions,
                rows * handle->n_classes * sizeof(float));

            cuda_utils::Free(d_X);
            cuda_utils::Free(d_predictions);
        }
        else {

            forest->value().predict(
                raft_proto::handle_t(handle->handle),
                h_predictions.data(),
                const_cast<float*>(X),
                rows,
                raft_proto::device_type::cpu,
                raft_proto::device_type::cpu
            );
        }

        for (int r = 0; r < rows; ++r) {

            int best = 0;
            float best_score =
                h_predictions[r * handle->n_classes];

            for (int c = 1; c < handle->n_classes; ++c) {

                float score =
                    h_predictions[r * handle->n_classes + c];

                if (score > best_score) {
                    best_score = score;
                    best = c;
                }
            }

            predictions[r] = best;
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "rf_predict exception: "
                  << e.what() << std::endl;

        cuda_utils::Free(d_X);
        cuda_utils::Free(d_predictions);

        return -1;
    }
    catch (...)
    {
        std::cerr << "rf_predict unknown exception"
                  << std::endl;

        cuda_utils::Free(d_X);
        cuda_utils::Free(d_predictions);

        return -1;
    }
}

int rf_save(
    RFHandle* handle,
    const char* filename)
{
    if (handle == nullptr || handle->tl_model == nullptr) {
        std::cerr << "rf_save: invalid handle" << std::endl;
        return -1;
    }

    int status = TreeliteSerializeModelToFile(
        handle->tl_model,
        filename
    );

    if (status != 0) {
        std::cerr << "Treelite error: "
                  << TreeliteGetLastError()
                  << std::endl;
        return -1;
    }

    std::ofstream meta(std::string(filename) + ".meta");

    if (!meta) {
        std::cerr << "Failed to create metadata file" << std::endl;
        return -1;
    }

    meta << handle->n_classes;
    meta.close();

    return 0;
}


RFHandle* rf_load(
    const char* filename)
{
    auto* rf = new RFHandle;

    rf->stream_pool = std::make_shared<rmm::cuda_stream_pool>(4);

    new (&rf->handle) raft::handle_t(
        rmm::cuda_stream_per_thread,
        rf->stream_pool
    );

    int status = TreeliteDeserializeModelFromFile(
        filename,
        &rf->tl_model
    );

    std::ifstream meta(
        std::string(filename) + ".meta"
    );

    if (!meta) {
        std::cerr << "Failed to open metadata file"
                << std::endl;

        TreeliteFreeModel(rf->tl_model);
        delete rf;
        return nullptr;
    }

    meta >> rf->n_classes;

    if (!meta) {
        std::cerr << "Failed to read n_classes"
                << std::endl;

        TreeliteFreeModel(rf->tl_model);
        delete rf;
        return nullptr;
    }

    if (status != 0) {
        std::cerr << "Treelite error: "
                  << TreeliteGetLastError()
                  << std::endl;

        delete rf;
        return nullptr;
    }

    int device;
    cudaGetDevice(&device);

    try {

        rf->cpu_forest.emplace(
            nvforest::import_from_treelite_handle(
                rf->tl_model,
                nvforest::preferred_tree_layout,
                0,
                std::nullopt,
                raft_proto::device_type::cpu,
                device,
                rf->handle.get_stream().value()
            )
        );
        std::cout << "rf_load: cpu forest imported" << std::endl;

        rf->gpu_forest.emplace(
            nvforest::import_from_treelite_handle(
                rf->tl_model,
                nvforest::preferred_tree_layout,
                0,
                std::nullopt,
                raft_proto::device_type::gpu,
                device,
                rf->handle.get_stream().value()
            )
        );

        return rf;
    }
    catch (const std::exception& e) {

        std::cerr << "rf_load exception: "
                  << e.what()
                  << std::endl;

        if (rf->tl_model != nullptr)
            TreeliteFreeModel(rf->tl_model);

        delete rf;
        return nullptr;
    }
}

void rf_destroy(RFHandle* rf)
{
    if (rf == nullptr)
        return;


    if (rf->tl_model != nullptr)
        TreeliteFreeModel(rf->tl_model);

    delete rf;
}
