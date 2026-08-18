#include "kmeans.h"

#include "../common/cuda_utils.hpp"

#include <cuml/cluster/kmeans.hpp>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <raft/random/rng_state.hpp>
#include <onnxruntime_cxx_api.h>


#include <vector>
#include <memory>

struct KMeansHandle {
    std::shared_ptr<rmm::cuda_stream_pool> stream_pool;
    raft::handle_t handle;

    ML::kmeans::KMeansParams params;

    int n_clusters;
    int n_features;

    float inertia;

    std::vector<float> centroids;

    KMeansHandle()
        : stream_pool(std::make_shared<rmm::cuda_stream_pool>(1)),
          handle(rmm::cuda_stream_per_thread, stream_pool),
          n_clusters(0),
          n_features(0),
          inertia(0.0f)
    {
    }

    Ort::Env env{
        ORT_LOGGING_LEVEL_WARNING,
        "kmeans"
    };
    std::unique_ptr<Ort::Session> session;
};

KMeansHandle* kmeans_create(
    int n_clusters,
    int max_iters,
    float tolerance)
{
    auto* km = new KMeansHandle;

    km->n_clusters = n_clusters;
    km->params.init = ML::kmeans::KMeansParams::InitMethod::KMeansPlusPlus;
    km->params.n_clusters = n_clusters;
    km->params.max_iter   = max_iters;
    km->params.tol        = tolerance;
    km->params.rng_state.seed = 42;

    return km;
}

int kmeans_fit(
    KMeansHandle* handle,
    const float* X,
    int rows,
    int cols)
{
    float* d_X = nullptr;
    float* d_centroids = nullptr;

    try
    {
        handle->n_features = cols;

        handle->centroids.resize(
            handle->n_clusters * cols
        );

        d_X = static_cast<float*>(
            cuda_utils::Malloc(rows * cols * sizeof(float)));

        d_centroids = static_cast<float*>(
            cuda_utils::Malloc(
                handle->n_clusters * cols * sizeof(float)));

        cuda_utils::CopyHostToDevice(
            d_X,
            X,
            rows * cols * sizeof(float));

        int n_iter = 0;

        ML::kmeans::fit(
            handle->handle,
            handle->params,
            d_X,
            rows,
            cols,
            nullptr,          // sample weights
            d_centroids,
            handle->inertia,
            n_iter
        );

        cuda_utils::CopyDeviceToHost(
            handle->centroids.data(),
            d_centroids,
            handle->n_clusters * cols * sizeof(float));

        cuda_utils::Free(d_X);
        cuda_utils::Free(d_centroids);

        return 0;
    }
    catch (...)
    {
        cuda_utils::Free(d_X);
        cuda_utils::Free(d_centroids);
        return -1;
    }
}

int kmeans_predict(
    KMeansHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* labels)
{
    float* d_X = nullptr;
    float* d_centroids = nullptr;
    int* d_labels = nullptr;

    try
    {
        if (cols != handle->n_features)
            return -1;

        d_X = static_cast<float*>(
            cuda_utils::Malloc(rows * cols * sizeof(float)));

        d_centroids = static_cast<float*>(
            cuda_utils::Malloc(
                handle->n_clusters * cols * sizeof(float)));

        d_labels = static_cast<int*>(
            cuda_utils::Malloc(rows * sizeof(int)));

        cuda_utils::CopyHostToDevice(
            d_X,
            X,
            rows * cols * sizeof(float));

        cuda_utils::CopyHostToDevice(
            d_centroids,
            handle->centroids.data(),
            handle->n_clusters * cols * sizeof(float));

        float inertia = 0.0f;

        ML::kmeans::predict(
            handle->handle,
            handle->params,
            d_centroids,
            d_X,
            rows,
            cols,
            nullptr,      // sample weights
            false,        // normalize weights
            d_labels,
            inertia
        );

        cuda_utils::CopyDeviceToHost(
            labels,
            d_labels,
            rows * sizeof(int));

        cuda_utils::Free(d_X);
        cuda_utils::Free(d_centroids);
        cuda_utils::Free(d_labels);

        return 0;
    }
    catch (...)
    {
        cuda_utils::Free(d_X);
        cuda_utils::Free(d_centroids);
        cuda_utils::Free(d_labels);

        return -1;
    }
}

int kmeans_load_onnx(
    KMeansHandle* handle,
    const char* filename
)
{
    try {

        Ort::SessionOptions options;

        options.SetIntraOpNumThreads(1);

        handle->session =
            std::make_unique<Ort::Session>(
                handle->env,
                filename,
                options
            );

        return 0;

    }
    catch (...) {

        return -1;
    }
}

int kmeans_predict_onnx(
    KMeansHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* predictions
)
{
    try {

        std::vector<int64_t> shape = {
            rows,
            cols
        };

        auto memory =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

        auto input =
            Ort::Value::CreateTensor<float>(
                memory,
                const_cast<float*>(X),
                rows * cols,
                shape.data(),
                shape.size()
            );

        const char* input_names[] = {
            "X"
        };

        const char* output_names[] = {
            "label"
        };

        auto outputs =
            handle->session->Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input,
                1,
                output_names,
                1
            );

        auto labels =
            outputs[0]
                .GetTensorMutableData<int64_t>();

        for (int i = 0; i < rows; i++) {

            predictions[i] =
                static_cast<int>(labels[i]);
        }

        return 0;

    }
    catch (const Ort::Exception& e) {
        std::cerr
            << "ONNX exception: "
            << e.what()
            << std::endl;

        return -1;
    }

    catch (const std::exception& e) {
        std::cerr
            << "std::exception: "
            << e.what()
            << std::endl;

        return -1;
    }

    catch (...) {
        std::cerr
            << "unknown exception"
            << std::endl;

        return -1;
    }
}

void kmeans_destroy(KMeansHandle* handle)
{
    if (handle == nullptr)
        return;

    delete handle;
}