#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KMeansCPUHandle KMeansCPUHandle;

KMeansCPUHandle* kmeans_cpu_create(
    int k,
    int max_iters,
    float tol
);

void kmeans_cpu_fit(
    KMeansCPUHandle* handle,
    const float* X,
    int n_samples,
    int n_features
);

void kmeans_cpu_predict(
    KMeansCPUHandle* handle,
    const float* X,
    int n_samples,
    int* labels
);

void kmeans_cpu_free(
    KMeansCPUHandle* handle
);

#ifdef __cplusplus
}
#endif