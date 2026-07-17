#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RFHandle RFHandle;

/* Lifecycle */
RFHandle* rf_create(
    int n_estimators,
    int max_depth,
    float max_features,
    int max_leaves,
    float max_samples
);

void rf_destroy(RFHandle* handle);

int rf_fit(
    RFHandle* handle,
    const float* X,
    int rows,
    int cols,
    const int* y,
    int n_classes
);

int rf_predict(
    RFHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* predictions
);

#ifdef __cplusplus
}
#endif
