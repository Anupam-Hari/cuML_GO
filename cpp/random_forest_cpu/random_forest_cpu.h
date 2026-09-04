#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RFCPUHandle RFCPUHandle;

/* Lifecycle */

RFCPUHandle* rf_cpu_create(
    int n_estimators,
    int max_depth,
    float max_features,
    int max_leaves,
    float max_samples
);

void rf_cpu_destroy(RFCPUHandle* handle);

void rf_set_cpu_threads_cpu(RFCPUHandle* handle, int threads);

/* Training */

int rf_cpu_fit(
    RFCPUHandle* handle,
    const float* X,
    int rows,
    int cols,
    const int* y,
    int n_classes
);

/* Prediction */

int rf_cpu_predict(
    RFCPUHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* predictions
);

#ifdef __cplusplus
}
#endif