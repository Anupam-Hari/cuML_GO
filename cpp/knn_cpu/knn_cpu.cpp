#include "knn_cpu.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

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
};

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

    return h;
}

static float squared_l2(
    const float* a,
    const float* b,
    int dim)
{
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

int knn_cpu_predict(
    KNNCPUHandle* handle,
    const float* X_query,
    int n_queries,
    int k,
    int* predictions)
{
    auto* impl = static_cast<KNNCPUImpl*>(handle->impl);
    if (k > impl->n_samples) return -1;

    for (int q = 0; q < n_queries; ++q) {

        std::vector<std::pair<float, int>> dists;

        const float* query = X_query + q * impl->n_features;

        for (int i = 0; i < impl->n_samples; ++i) {

            const float* train = impl->X + i * impl->n_features;

            float dist = squared_l2(query, train, impl->n_features);

            dists.emplace_back(dist, impl->y[i]);
        }

        std::nth_element(
            dists.begin(),
            dists.begin() + k,
            dists.end(),
            [](auto& a, auto& b) { return a.first < b.first; }
        );

        std::vector<int> counts(impl->n_classes, 0);

        for (int i = 0; i < k; ++i) {
            counts[dists[i].second]++;
        }

        int best_label = 0;
        int best_count = counts[0];

        for (int c = 1; c < impl->n_classes; ++c) {
            if (counts[c] > best_count) {
                best_count = counts[c];
                best_label = c;
            }
        }

        predictions[q] = best_label;
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