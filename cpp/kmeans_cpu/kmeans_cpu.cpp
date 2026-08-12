#include "kmeans_cpu.h"

#include <vector>
#include <cmath>
#include <limits>
#include <cstring>
#include <cblas.h>
#include <algorithm>
#include <omp.h>

struct KMeansCPUHandle {
    int k;
    int max_iters;
    float tol;

    int n_features;

    std::vector<float> centroids;
    std::vector<double> centroid_norms;
};

// static float dist_sq(const float* a, const float* b, int d) {
//     float s = 0;
//     for (int i = 0; i < d; i++) {
//         float diff = a[i] - b[i];
//         s += diff * diff;
//     }
//     return s;
// }

KMeansCPUHandle* kmeans_cpu_create(
    int k,
    int max_iters,
    float tol
) {
    KMeansCPUHandle* h = new KMeansCPUHandle;
    h->k = k;
    h->max_iters = max_iters;
    h->tol = tol;
    return h;
}

void kmeans_cpu_fit(
    KMeansCPUHandle* h,
    const float* X,
    int n_samples,
    int n_features
) {
    h->n_features = n_features;
    h->centroids.resize(h->k * n_features);

    // init → first k points
    std::memcpy(
        h->centroids.data(),
        X,
        sizeof(float) * h->k * n_features
    );

    std::vector<int> labels(n_samples);
    std::vector<float> new_centroids(
        h->k * n_features
    );
    std::vector<int> counts(h->k);

    const int chunk_size = 256;

    for (int iter = 0;
         iter < h->max_iters;
         ++iter) {

        // --------------------------------------------------
        // Convert current centroids to double
        // --------------------------------------------------

        std::vector<double> centroids_double(
            static_cast<size_t>(h->k) * n_features
        );

        for (int c = 0; c < h->k; ++c) {
            for (int j = 0; j < n_features; ++j) {

                centroids_double[
                    static_cast<size_t>(c) * n_features + j
                ] =
                    static_cast<double>(
                        h->centroids[
                            static_cast<size_t>(c) *
                            n_features + j
                        ]
                    );
            }
        }

        // --------------------------------------------------
        // Compute squared norms of current centroids
        // --------------------------------------------------

        std::vector<double> centroid_norms(h->k);

        for (int c = 0; c < h->k; ++c) {

            double sum = 0.0;

            for (int j = 0; j < n_features; ++j) {

                double value =
                    centroids_double[
                        static_cast<size_t>(c) *
                        n_features + j
                    ];

                sum += value * value;
            }

            centroid_norms[c] = sum;
        }

        // --------------------------------------------------
        // Assignment
        // --------------------------------------------------

        for (int start = 0;
             start < n_samples;
             start += chunk_size) {

            const int count =
                std::min(
                    chunk_size,
                    n_samples - start
                );

            std::vector<double> X_double(
                static_cast<size_t>(count) *
                n_features
            );

            // float -> double
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < n_features; ++j) {

                    X_double[
                        static_cast<size_t>(i) *
                        n_features + j
                    ] =
                        static_cast<double>(
                            X[
                                static_cast<size_t>(
                                    start + i
                                ) * n_features + j
                            ]
                        );
                }
            }

            std::vector<double> distances(
                static_cast<size_t>(count) *
                h->k
            );

            // -2 * X * C^T
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,

                count,
                h->k,
                n_features,

                -2.0,

                X_double.data(),
                n_features,

                centroids_double.data(),
                n_features,

                0.0,

                distances.data(),
                h->k
            );

            // Add ||C||².
            //
            // ||X-C||² =
            //     ||X||² - 2X.C + ||C||²
            //
            // ||X||² can be omitted because it is
            // identical for every centroid of a sample.

            for (int i = 0; i < count; ++i) {

                for (int c = 0;
                     c < h->k;
                     ++c) {

                    distances[
                        static_cast<size_t>(i) *
                        h->k + c
                    ] += centroid_norms[c];
                }
            }

            // Argmin
            for (int i = 0; i < count; ++i) {

                int best_k = 0;

                double best =
                    distances[
                        static_cast<size_t>(i) *
                        h->k
                    ];

                for (int c = 1;
                     c < h->k;
                     ++c) {

                    double distance =
                        distances[
                            static_cast<size_t>(i) *
                            h->k + c
                        ];

                    if (distance < best) {
                        best = distance;
                        best_k = c;
                    }
                }

                labels[start + i] = best_k;
            }
        }

        // --------------------------------------------------
        // Accumulate
        // --------------------------------------------------

        std::fill(
            new_centroids.begin(),
            new_centroids.end(),
            0
        );

        std::fill(
            counts.begin(),
            counts.end(),
            0
        );

        for (int i = 0;
             i < n_samples;
             ++i) {

            int c = labels[i];

            counts[c]++;

            for (int j = 0;
                 j < n_features;
                 ++j) {

                new_centroids[
                    static_cast<size_t>(c) *
                    n_features + j
                ] +=
                    X[
                        static_cast<size_t>(i) *
                        n_features + j
                    ];
            }
        }

        // --------------------------------------------------
        // Normalize
        // --------------------------------------------------

        for (int c = 0;
             c < h->k;
             ++c) {

            if (counts[c] == 0)
                continue;

            for (int j = 0;
                 j < n_features;
                 ++j) {

                new_centroids[
                    static_cast<size_t>(c) *
                    n_features + j
                ] /= counts[c];
            }
        }

        h->centroids = new_centroids;
    }

    // ------------------------------------------------------
    // Precompute final centroid norms for inference
    // ------------------------------------------------------

    h->centroid_norms.resize(h->k);

    for (int c = 0;
         c < h->k;
         ++c) {

        double sum = 0.0;

        for (int j = 0;
             j < n_features;
             ++j) {

            double value =
                static_cast<double>(
                    h->centroids[
                        static_cast<size_t>(c) *
                        n_features + j
                    ]
                );

            sum += value * value;
        }

        h->centroid_norms[c] = sum;
    }
}

void kmeans_cpu_predict(
    KMeansCPUHandle* h,
    const float* X,
    int n_samples,
    int* labels
) {
    const int d = h->n_features;
    const int chunk_size = 256;
    const int n_chunks =
        (n_samples + chunk_size - 1) / chunk_size;

    const int n_threads =
        std::min(omp_get_max_threads(), n_chunks);

    std::vector<std::vector<double>> thread_X_double(
        n_threads
    );

    std::vector<std::vector<double>> thread_distances(
        n_threads
    );

    for (int thread = 0;
        thread < n_threads;
        ++thread) {

        thread_X_double[thread].resize(
            static_cast<size_t>(chunk_size) * d
        );

        thread_distances[thread].resize(
            static_cast<size_t>(chunk_size) * h->k
        );
    }

    std::vector<double> centroids_double(
    static_cast<size_t>(h->k) * d
    );

    for (int c = 0; c < h->k; ++c) {
        for (int j = 0; j < d; ++j) {

            centroids_double[
                static_cast<size_t>(c) * d + j
            ] =
                static_cast<double>(
                    h->centroids[
                        static_cast<size_t>(c) * d + j
                    ]
                );
        }
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
                    n_samples - start
                );

            // Convert this thread's X chunk.
            for (int i = 0; i < count; ++i) {

                for (int j = 0; j < d; ++j) {

                    local_X[
                        static_cast<size_t>(i) * d + j
                    ] =
                        static_cast<double>(
                            X[
                                static_cast<size_t>(start + i) *
                                d + j
                            ]
                        );
                }
            }

            // -2 * X_chunk * C^T
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,

                count,
                h->k,
                d,

                -2.0,

                local_X,
                d,

                centroids_double.data(),
                d,

                0.0,

                local_distances,
                h->k
            );

            // Add ||C||².
            for (int i = 0; i < count; ++i) {

                for (int c = 0;
                    c < h->k;
                    ++c) {

                    local_distances[
                        static_cast<size_t>(i) * h->k + c
                    ] += h->centroid_norms[c];
                }
            }

            // Argmin.
            for (int i = 0; i < count; ++i) {

                int best_cluster = 0;

                double best_distance =
                    local_distances[
                        static_cast<size_t>(i) * h->k
                    ];

                for (int c = 1;
                    c < h->k;
                    ++c) {

                    double distance =
                        local_distances[
                            static_cast<size_t>(i) * h->k + c
                        ];

                    if (distance < best_distance) {

                        best_distance = distance;
                        best_cluster = c;
                    }
                }

                labels[start + i] = best_cluster;
            }
        }
    }
}

void kmeans_cpu_free(KMeansCPUHandle* h) {
    delete h;
}