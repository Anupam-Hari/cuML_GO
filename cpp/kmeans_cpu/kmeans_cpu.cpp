#include "kmeans_cpu.h"

#include <mlpack.hpp>

#include <armadillo>
#include <limits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <omp.h>
#include <cblas.h>
#include <vector>

struct KMeansCPUHandle {

    int k;
    int max_iters;
    float tol;

    int n_features;

    std::vector<double> centroid_norms;

    /*
     * mlpack / Armadillo stores matrices as:
     *
     * rows    = features
     * columns = samples
     *
     * Therefore centroids has:
     *
     *   cols = k
     *   rows = n_features
     */
    arma::mat centroids;

    bool fitted;
};


KMeansCPUHandle* kmeans_cpu_create(
    int k,
    int max_iters,
    float tol
)
{
    try {

        if (k <= 0) {
            std::cerr
                << "kmeans_cpu_create: "
                << "k must be greater than zero"
                << std::endl;

            return nullptr;
        }

        if (max_iters <= 0) {
            std::cerr
                << "kmeans_cpu_create: "
                << "max_iters must be greater than zero"
                << std::endl;

            return nullptr;
        }

        if (tol < 0.0f) {
            std::cerr
                << "kmeans_cpu_create: "
                << "tol must be non-negative"
                << std::endl;

            return nullptr;
        }

        auto* handle = new KMeansCPUHandle();

        handle->k = k;
        handle->max_iters = max_iters;
        handle->tol = tol;

        handle->n_features = 0;
        handle->fitted = false;

        return handle;
    }
    catch (const std::exception& e) {

        std::cerr
            << "kmeans_cpu_create: "
            << e.what()
            << std::endl;

        return nullptr;
    }
}


void kmeans_cpu_free(
    KMeansCPUHandle* handle
)
{
    delete handle;
}


int kmeans_cpu_fit(
    KMeansCPUHandle* handle,
    const float* X,
    int rows,
    int cols
)
{
    if (!handle || !X) {
        return -1;
    }

    if (rows <= 0 || cols <= 0) {
        return -1;
    }

    if (handle->k > rows) {
        std::cerr
            << "kmeans_cpu_fit: "
            << "k cannot be greater than number of samples"
            << std::endl;

        return -1;
    }

    try {

        handle->n_features = cols;

        /*
         * Go/C API gives us row-major data:
         *
         * X[r * cols + c]
         *
         * mlpack/Armadillo expects:
         *
         * rows    = features
         * columns = samples
         *
         * Therefore:
         *
         * data(c, r) = X[r * cols + c]
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

        /*
         * mlpack KMeans parameters:
         *
         * KMeans<MetricType, InitialPartitionPolicy,
         *        EmptyClusterPolicy>
         *
         * Cluster() performs the actual KMeans algorithm.
         *
         * The centroids matrix will have:
         *
         *   cols = k
         *   rows = number of features
         */

        mlpack::KMeans<> kmeans(handle->max_iters);

        arma::mat centroids;

        kmeans.Cluster(
            data,
            static_cast<size_t>(handle->k),
            centroids
        );

        /*
         * Store the trained centroids in the handle.
         */

        handle->centroids = std::move(centroids);

        handle->fitted = true;

        handle->centroid_norms.resize(handle->k);

        for (int c = 0; c < handle->k; ++c) {
            handle->centroid_norms[c] =
                arma::dot(
                    handle->centroids.col(c),
                    handle->centroids.col(c)
                );
        }

        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "kmeans_cpu_fit: "
            << e.what()
            << std::endl;

        handle->fitted = false;

        return -1;
    }
}


int kmeans_cpu_predict(
    KMeansCPUHandle* handle,
    const float* X,
    int rows,
    int* predictions
)
{
    if (
        !handle ||
        !handle->fitted ||
        !X ||
        !predictions
    ) {
        return -1;
    }

    if (rows <= 0 || handle->n_features <= 0) {
        return -1;
    }

    try {

        const int d = handle->n_features;
        const int chunk_size = 256;
        const int n_chunks =
            (rows + chunk_size - 1) / chunk_size;

        const int n_threads =
            std::min(
                omp_get_max_threads(),
                n_chunks
            );

        std::vector<std::vector<double>> thread_X_double(
            n_threads
        );

        std::vector<std::vector<double>> thread_distances(
            n_threads
        );

        for (int thread = 0; thread < n_threads; ++thread) {

            thread_X_double[thread].resize(
                static_cast<size_t>(chunk_size) * d
            );

            thread_distances[thread].resize(
                static_cast<size_t>(chunk_size) * handle->k
            );
        }

        #pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();

            double* local_X =
                thread_X_double[thread_id].data();

            double* local_distances =
                thread_distances[thread_id].data();

            #pragma omp for schedule(static)
            for (int chunk_idx = 0;
                 chunk_idx < n_chunks;
                 ++chunk_idx) {

                const int start =
                    chunk_idx * chunk_size;

                const int count =
                    std::min(
                        chunk_size,
                        rows - start
                    );

                /*
                 * Convert this chunk from Go's
                 * row-major float32 format to
                 * row-major double format.
                 */
                for (int i = 0; i < count; ++i) {

                    for (int j = 0; j < d; ++j) {

                        local_X[
                            static_cast<size_t>(i) * d + j
                        ] =
                            static_cast<double>(
                                X[
                                    static_cast<size_t>(start + i) * d + j
                                ]
                            );
                    }
                }

                /*
                 * distances =
                 *
                 *     -2 * X * C^T
                 *
                 * mlpack centroids are stored as:
                 *
                 *     features x clusters
                 *
                 * so CblasTrans gives us:
                 *
                 *     clusters x features
                 */
                cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,

                    count,
                    handle->k,
                    d,

                    -2.0,

                    local_X,
                    d,

                    handle->centroids.memptr(),
                    d,

                    0.0,

                    local_distances,
                    handle->k
                );

                /*
                 * Add ||C||².
                 */
                for (int i = 0; i < count; ++i) {

                    for (int c = 0;
                         c < handle->k;
                         ++c) {

                        local_distances[
                            static_cast<size_t>(i) * handle->k + c
                        ] +=
                            handle->centroid_norms[c];
                    }
                }

                /*
                 * Find nearest centroid.
                 */
                for (int i = 0; i < count; ++i) {

                    int best_cluster = 0;

                    double best_distance =
                        local_distances[
                            static_cast<size_t>(i) * handle->k
                        ];

                    for (int c = 1;
                         c < handle->k;
                         ++c) {

                        const double distance =
                            local_distances[
                                static_cast<size_t>(i) * handle->k + c
                            ];

                        if (distance < best_distance) {

                            best_distance = distance;
                            best_cluster = c;
                        }
                    }

                    predictions[start + i] =
                        best_cluster;
                }
            }
        }

        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "kmeans_cpu_predict: "
            << e.what()
            << std::endl;

        return -1;
    }
}