#include "knn.h"

#include "../common/cuda_utils.hpp"

#include <cuml/neighbors/knn.hpp>

#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_pool.hpp>

#include <memory>
#include <vector>
#include <iostream>

struct KNNHandle {
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
    raft::handle_t handle;

    int k;

    int rows;
    int cols;

    float* d_X;
    int* d_y;

    KNNHandle()
        : stream_pool(std::make_shared<rmm::cuda_stream_pool>(1)),
          handle(rmm::cuda_stream_per_thread, stream_pool),
          k(0),
          rows(0),
          cols(0),
          d_X(nullptr),
          d_y(nullptr)
    {
    }
};

KNNHandle* knn_create(int k)
{
    auto* knn = new KNNHandle;

    knn->k = k;

    return knn;
}

int knn_fit(
    KNNHandle* handle,
    const float* X,
    int rows,
    int cols,
    const int* y)
{
    try
    {
        handle->rows = rows;
        handle->cols = cols;

        handle->d_X = static_cast<float*>(
            cuda_utils::Malloc(rows * cols * sizeof(float)));

        handle->d_y = static_cast<int*>(
            cuda_utils::Malloc(rows * sizeof(int)));

        cuda_utils::CopyHostToDevice(
            handle->d_X,
            X,
            rows * cols * sizeof(float));

        cuda_utils::CopyHostToDevice(
            handle->d_y,
            y,
            rows * sizeof(int));

        return 0;
    }
    catch (...)
    {
        cuda_utils::Free(handle->d_X);
        cuda_utils::Free(handle->d_y);

        handle->d_X = nullptr;
        handle->d_y = nullptr;

        return -1;
    }
}

int knn_predict(
    KNNHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* predictions)
{
    float* d_query = nullptr;
    int64_t* d_indices = nullptr;
    float* d_distances = nullptr;
    int* d_predictions = nullptr;

    try
    {
        if (cols != handle->cols)
            return -1;

        d_query = static_cast<float*>(
            cuda_utils::Malloc(rows * cols * sizeof(float)));

        d_indices = static_cast<int64_t*>(
            cuda_utils::Malloc(rows * handle->k * sizeof(int64_t)));

        d_distances = static_cast<float*>(
            cuda_utils::Malloc(rows * handle->k * sizeof(float)));

        d_predictions = static_cast<int*>(
            cuda_utils::Malloc(rows * sizeof(int)));

        cuda_utils::CopyHostToDevice(
            d_query,
            X,
            rows * cols * sizeof(float));

        std::vector<float*> inputs{handle->d_X};
        std::vector<int> sizes{handle->rows};

        ML::brute_force_knn(
            handle->handle,
            inputs,
            sizes,
            handle->cols,
            d_query,
            rows,
            d_indices,
            d_distances,
            handle->k,
            true,
            true
        );

        std::vector<int*> labels{handle->d_y};

        ML::knn_classify(
            handle->handle,
            d_predictions,
            d_indices,
            labels,
            handle->rows,
            rows,
            handle->k
        );

        cuda_utils::CopyDeviceToHost(
            predictions,
            d_predictions,
            rows * sizeof(int));

        cuda_utils::Free(d_query);
        cuda_utils::Free(d_indices);
        cuda_utils::Free(d_distances);
        cuda_utils::Free(d_predictions);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "knn_predict exception: "
                  << e.what() << std::endl;

        cuda_utils::Free(d_query);
        cuda_utils::Free(d_indices);
        cuda_utils::Free(d_distances);
        cuda_utils::Free(d_predictions);

        return -1;
    }
    catch (...)
    {
        std::cerr << "knn_predict unknown exception"
                  << std::endl;

        cuda_utils::Free(d_query);
        cuda_utils::Free(d_indices);
        cuda_utils::Free(d_distances);
        cuda_utils::Free(d_predictions);

        return -1;
    }
}

void knn_destroy(KNNHandle* handle)
{
    if (handle == nullptr)
        return;

    if (handle->d_X != nullptr)
    {
        cuda_utils::Free(handle->d_X);
        handle->d_X = nullptr;
    }

    if (handle->d_y != nullptr)
    {
        cuda_utils::Free(handle->d_y);
        handle->d_y = nullptr;
    }

    delete handle;
}