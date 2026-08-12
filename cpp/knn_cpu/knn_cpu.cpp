#include "knn_cpu.h"
#include <vector>
#include <cblas.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <omp.h>

extern "C" {

struct KNNCPUHandle {
    void* impl;
};

}

struct KNNCPUImpl {
    float* X;
    int* y;
    int n_samples;
    int n_features;
    int n_classes;
    std::vector<double> X_norms;
    std::vector<double> X_double;
};

struct KNNHeap {
    double* distances;
    int* indices;
    int k;
};

static inline void heap_push(
    double* distances,
    int* indices,
    int k,
    double val,
    int val_idx
) {
    if (val >= distances[0]) {
        return;
    }

    int current = 0;

    distances[0] = val;
    indices[0] = val_idx;

    while (true) {

        const int left = 2 * current + 1;
        const int right = left + 1;

        if (left >= k) {
            break;
        }

        int swap_idx;

        if (right >= k) {

            if (distances[left] > val) {
                swap_idx = left;
            } else {
                break;
            }

        } else if (
            distances[left] >= distances[right]
        ) {

            if (val < distances[left]) {
                swap_idx = left;
            } else {
                break;
            }

        } else {

            if (val < distances[right]) {
                swap_idx = right;
            } else {
                break;
            }
        }

        distances[current] =
            distances[swap_idx];

        indices[current] =
            indices[swap_idx];

        current = swap_idx;
    }

    distances[current] = val;
    indices[current] = val_idx;
}

KNNCPUHandle* knn_cpu_create(
    const float* X,
    const int* y,
    int n_samples,
    int n_features,
    int n_classes)
{
    auto* impl = new KNNCPUImpl;
    auto* h = new KNNCPUHandle; 
    h->impl = impl;

    impl->n_samples = n_samples;
    impl->n_features = n_features;
    impl->n_classes = n_classes;

    impl->X = new float[n_samples * n_features];
    impl->y = new int[n_samples];

    memcpy(impl->X, X, sizeof(float)*n_samples*n_features);
    memcpy(impl->y, y, sizeof(int)*n_samples);

    impl->X_double.resize(
        static_cast<size_t>(n_samples) * n_features
    );

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {

            impl->X_double[
                static_cast<size_t>(i) * n_features + j
            ] =
                static_cast<double>(
                    impl->X[
                        static_cast<size_t>(i) * n_features + j
                    ]
                );
        }
    }

    impl->X_norms.resize(n_samples);

    for (int i = 0; i < n_samples; ++i) {

        double sum = 0.0;

        for (int j = 0; j < n_features; ++j) {

            double value =
                static_cast<double>(
                    impl->X[
                        static_cast<size_t>(i) * n_features + j
                    ]
                );

            sum += value * value;
        }

        impl->X_norms[i] = sum;
    }

    return h;
}

// static float squared_l2(
//     const float* a,
//     const float* b,
//     int dim)
// {
//     float dist = 0.0f;
//     for (int i = 0; i < dim; ++i) {
//         float d = a[i] - b[i];
//         dist += d * d;
//     }
//     return dist;
// }

static void compute_squared_distances(
    const double* X,
    const double* Y,
    const double* X_norms,
    const double* Y_norms,
    int nX,
    int nY,
    int nFeatures,
    double* distances
) {

    // distances = -2 * X * Y^T
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        nX,
        nY,
        nFeatures,
        -2.0,
        X,
        nFeatures,
        Y,
        nFeatures,
        0.0,
        distances,
        nY
    );

    // Add ||X||² + ||Y||²
    for (int i = 0; i < nX; ++i) {

        for (int j = 0; j < nY; ++j) {

            distances[
                static_cast<size_t>(i) * nY + j
            ] += X_norms[i] + Y_norms[j];
        }
    }
}

int knn_cpu_predict(
    KNNCPUHandle* handle,
    const float* X_query,
    int n_queries,
    int k,
    int* predictions)
{
    auto* impl = static_cast<KNNCPUImpl*>(handle->impl);

    if (k > impl->n_samples)
        return -1;

    const int chunk_size = 256;

    const int n_query_chunks =
        (n_queries + chunk_size - 1) / chunk_size;

    #pragma omp parallel for schedule(static)
    for (int q_chunk_idx = 0;
        q_chunk_idx < n_query_chunks;
        ++q_chunk_idx) {

        const int q_start =
            q_chunk_idx * chunk_size;

        const int q_count =
            std::min(
                chunk_size,
                n_queries - q_start
            );

        std::vector<double> query_double(
            static_cast<size_t>(q_count) * impl->n_features
        );

        for (int q = 0; q < q_count; ++q) {

            for (int f = 0; f < impl->n_features; ++f) {

                query_double[
                    static_cast<size_t>(q) * impl->n_features + f
                ] =
                    static_cast<double>(
                        X_query[
                            static_cast<size_t>(q_start + q) *
                            impl->n_features + f
                        ]
                    );
            }
        }

        std::vector<double> query_norms(q_count);

        for (int q = 0; q < q_count; ++q) {

            double sum = 0.0;

            for (int f = 0; f < impl->n_features; ++f) {

                double value =
                    query_double[
                        static_cast<size_t>(q) * impl->n_features + f
                    ];

                sum += value * value;
            }

            query_norms[q] = sum;
        }

        // One persistent top-k heap for every query.
        // These heaps survive across ALL training chunks.
        std::vector<double> heap_distances(
            static_cast<size_t>(q_count) * k,
            std::numeric_limits<double>::max()
        );

        std::vector<int> heap_indices(
            static_cast<size_t>(q_count) * k,
            -1
        );

        const int n_threads = omp_get_max_threads();

        std::vector<std::vector<double>> thread_heap_distances(
            n_threads
        );

        std::vector<std::vector<int>> thread_heap_indices(
            n_threads
        );

        for (int thread = 0; thread < n_threads; ++thread) {

            thread_heap_distances[thread].resize(
                static_cast<size_t>(q_count) * k
            );

            thread_heap_indices[thread].resize(
                static_cast<size_t>(q_count) * k
            );

            std::fill(
                thread_heap_distances[thread].begin(),
                thread_heap_distances[thread].end(),
                std::numeric_limits<double>::max()
            );

            std::fill(
                thread_heap_indices[thread].begin(),
                thread_heap_indices[thread].end(),
                -1
            );
        }

        std::vector<int> counts(
            static_cast<size_t>(q_count) * impl->n_classes,
            0
        );

        std::vector<std::vector<double>> thread_distances(
            n_threads
        );

        for (int thread = 0; thread < n_threads; ++thread) {

            thread_distances[thread].resize(
                static_cast<size_t>(q_count) * chunk_size
            );
        }

        // Process training samples in chunks.
        const int n_training_chunks =
            (impl->n_samples + chunk_size - 1) / chunk_size;

        #pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();

            double* local_distances =
                thread_distances[thread_id].data();

            double* local_heap_distances =
                thread_heap_distances[thread_id].data();

            int* local_heap_indices =
                thread_heap_indices[thread_id].data();

            for (int chunk_idx = 0;
                chunk_idx < n_training_chunks;
                ++chunk_idx) {

                const int t_start =
                    chunk_idx * chunk_size;

                const int t_count =
                    std::min(
                        chunk_size,
                        impl->n_samples - t_start
                    );

                compute_squared_distances(
                    query_double.data(),

                    impl->X_double.data() +
                        static_cast<size_t>(t_start) *
                        impl->n_features,

                    query_norms.data(),

                    impl->X_norms.data() + t_start,

                    q_count,
                    t_count,
                    impl->n_features,

                    local_distances
                );

                for (int q = 0; q < q_count; ++q) {

                    const size_t heap_offset =
                        static_cast<size_t>(q) * k;

                    double* heap_distances_q =
                        local_heap_distances + heap_offset;

                    int* heap_indices_q =
                        local_heap_indices + heap_offset;

                    for (int t = 0; t < t_count; ++t) {

                        const double distance =
                            local_distances[
                                static_cast<size_t>(q) * t_count + t
                            ];

                        const int global_index =
                            t_start + t;

                        heap_push(
                            heap_distances_q,
                            heap_indices_q,
                            k,
                            distance,
                            global_index
                        );
                    }
                }
            }
        }

        // Merge every thread-local heap into the main heap.
        #pragma omp parallel for schedule(static)
        for (int q = 0; q < q_count; ++q) {

            const size_t offset =
                static_cast<size_t>(q) * k;

            double* global_distances =
                heap_distances.data() + offset;

            int* global_indices =
                heap_indices.data() + offset;

            for (int thread = 0;
                thread < n_threads;
                ++thread) {

                const double* thread_distances =
                    thread_heap_distances[thread].data() +
                    offset;

                const int* thread_indices =
                    thread_heap_indices[thread].data() +
                    offset;

                for (int i = 0; i < k; ++i) {

                    heap_push(
                        global_distances,
                        global_indices,
                        k,
                        thread_distances[i],
                        thread_indices[i]
                    );
                }
            }
        }

        // All training chunks have now been processed.
        // Each heap contains the final k nearest neighbors.
        for (int q = 0; q < q_count; ++q) {

            int* query_counts =
                counts.data() +
                static_cast<size_t>(q) * impl->n_classes;

            std::fill(
                query_counts,
                query_counts + impl->n_classes,
                0
            );

            const int* indices =
                heap_indices.data() +
                static_cast<size_t>(q) * k;

            for (int i = 0; i < k; ++i) {
                const int index = indices[i];
                query_counts[impl->y[index]]++;
            }

            int best_label = 0;
            int best_count = query_counts[0];

            for (int c = 1;
                 c < impl->n_classes;
                 ++c) {

                if (query_counts[c] > best_count) {

                    best_count =
                        query_counts[c];

                    best_label = c;
                }
            }

            predictions[q_start + q] =
                best_label;
        }
    }

    return 0;
}

void knn_cpu_free(KNNCPUHandle* handle)
{
    if (!handle) return;

    auto* impl = static_cast<KNNCPUImpl*>(handle->impl);

    delete[] impl->X;
    delete[] impl->y;
    delete impl;
    delete handle;
}