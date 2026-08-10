#include "kmeans_cpu.h"

#include <vector>
#include <cmath>
#include <limits>
#include <cstring>

struct KMeansCPUHandle {
    int k;
    int max_iters;
    float tol;

    int n_features;

    std::vector<float> centroids;
};

static float dist_sq(const float* a, const float* b, int d) {
    float s = 0;
    for (int i = 0; i < d; i++) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

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
    std::vector<float> new_centroids(h->k * n_features);
    std::vector<int> counts(h->k);

    for (int iter = 0; iter < h->max_iters; iter++) {

        // assign
        for (int i = 0; i < n_samples; i++) {
            float best = std::numeric_limits<float>::max();
            int best_k = 0;

            for (int c = 0; c < h->k; c++) {
                float d = dist_sq(
                    &X[i * n_features],
                    &h->centroids[c * n_features],
                    n_features
                );
                if (d < best) {
                    best = d;
                    best_k = c;
                }
            }
            labels[i] = best_k;
        }

        std::fill(new_centroids.begin(), new_centroids.end(), 0);
        std::fill(counts.begin(), counts.end(), 0);

        // accumulate
        for (int i = 0; i < n_samples; i++) {
            int c = labels[i];
            counts[c]++;

            for (int j = 0; j < n_features; j++) {
                new_centroids[c * n_features + j] +=
                    X[i * n_features + j];
            }
        }

        // normalize
        for (int c = 0; c < h->k; c++) {
            if (counts[c] == 0) continue;

            for (int j = 0; j < n_features; j++) {
                new_centroids[c * n_features + j] /= counts[c];
            }
        }

        h->centroids = new_centroids;
    }
}

void kmeans_cpu_predict(
    KMeansCPUHandle* h,
    const float* X,
    int n_samples,
    int* labels
) {
    int d = h->n_features;

    for (int i = 0; i < n_samples; i++) {

        float best = std::numeric_limits<float>::max();
        int best_k = 0;

        for (int c = 0; c < h->k; c++) {
            float d2 = dist_sq(
                &X[i * d],
                &h->centroids[c * d],
                d
            );
            if (d2 < best) {
                best = d2;
                best_k = c;
            }
        }

        labels[i] = best_k;
    }
}

void kmeans_cpu_free(KMeansCPUHandle* h) {
    delete h;
}