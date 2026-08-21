#include "kmeans_cpu.h"

#include <mlpack.hpp>

#include <armadillo>
#include <limits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

struct KMeansCPUHandle {

    int k;
    int max_iters;
    float tol;

    int n_features;

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

    if (rows <= 0) {
        return -1;
    }

    if (handle->n_features <= 0) {
        return -1;
    }

    try {

        const int cols = handle->n_features;

        /*
         * Convert row-major Go data into Armadillo format.
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
         * Assign every sample to its nearest centroid.
         *
         * assignments:
         *
         *   one cluster ID per input sample
         */

        for (int r = 0; r < rows; ++r) {

            double best_distance = std::numeric_limits<double>::max();
            int best_cluster = 0;

            for (int c = 0; c < handle->k; ++c) {

                double distance = 0.0;

                for (int f = 0; f < cols; ++f) {

                    const double diff =
                        data(f, r) -
                        handle->centroids(f, c);

                    distance += diff * diff;
                }

                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = c;
                }
            }

            predictions[r] = best_cluster;
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