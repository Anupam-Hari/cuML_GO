#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KNNCPUHandle KNNCPUHandle;

KNNCPUHandle* knn_cpu_create(
    const float* X,
    const int* y,
    int n_samples,
    int n_features,
    int n_classes
);

int knn_cpu_predict(
    KNNCPUHandle* handle,
    const float* X_query,
    int n_queries,
    int k,
    int* predictions
);

void knn_cpu_free(KNNCPUHandle* handle);

#ifdef __cplusplus
}
#endif