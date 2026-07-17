#include "random_forest.h"

#include "../common/cuda_utils.hpp"
#include <cuml/ensemble/randomforest.hpp>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_pool.hpp>

struct RFHandle {
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
    raft::handle_t handle;
    ML::RF_params params;
    ML::RandomForestClassifierF* forest;

    RFHandle()
        : stream_pool(std::make_shared<rmm::cuda_stream_pool>(1)),
          handle(rmm::cuda_stream_per_thread, stream_pool),
          forest(nullptr)
    {
    }
};

RFHandle* rf_create(
    int n_estimators,
    int max_depth,
    float max_features,
    int max_leaves,
    float max_samples)
{
    auto* rf = new RFHandle;

    rf->forest = new ML::RandomForestClassifierF();

    rf->params = ML::set_rf_params(
        max_depth,
        max_leaves,
        max_features,
        128,                     // max_n_bins
        1,                       // min_samples_leaf
        2,                       // min_samples_split
        0.0f,                    // min_impurity_decrease
        true,                    // bootstrap
        n_estimators,
        max_samples,
        0,                       // seed
        ML::CRITERION::GINI,
        1,                       // n_streams
        4096                     // max_batch_size
    );

    return rf;
}

int rf_fit(
    RFHandle* handle,
    const float* X,
    int rows,
    int cols,
    const int* y,
    int n_classes)
{
    float* d_X = nullptr;
    int* d_y = nullptr;

    try
    {
        d_X = static_cast<float*>(
            cuda_utils::Malloc(rows * cols * sizeof(float)));

        d_y = static_cast<int*>(
            cuda_utils::Malloc(rows * sizeof(int)));

        cuda_utils::CopyHostToDevice(
            d_X,
            X,
            rows * cols * sizeof(float));

        cuda_utils::CopyHostToDevice(
            d_y,
            y,
            rows * sizeof(int));

        ML::fit(
	    handle->handle,
	    handle->forest,
	    d_X,
	    rows,
	    cols,
	    d_y,
	    n_classes,
	    handle->params
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
    int* predictions)
{
    float* d_X = nullptr;
    int* d_predictions = nullptr;

    try
    {
        d_X = static_cast<float*>(
            cuda_utils::Malloc(rows * cols * sizeof(float)));

        d_predictions = static_cast<int*>(
            cuda_utils::Malloc(rows * sizeof(int)));

        cuda_utils::CopyHostToDevice(
            d_X,
            X,
            rows * cols * sizeof(float));

        ML::predict(
            handle->handle,
            handle->forest,
            d_X,
            rows,
            cols,
            d_predictions);

        cuda_utils::CopyDeviceToHost(
            predictions,
            d_predictions,
            rows * sizeof(int));

        cuda_utils::Free(d_X);
        cuda_utils::Free(d_predictions);

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

#include <iostream>

void rf_destroy(RFHandle* rf)
{
    std::cout << "destroy: begin\n";

    if (rf == nullptr)
        return;


    if (rf->forest != nullptr) {
        ML::delete_rf_metadata(rf->forest);
    }

    delete rf;
}
