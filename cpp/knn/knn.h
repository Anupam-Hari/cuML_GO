#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KNNHandle KNNHandle;

KNNHandle* knn_create(
    int k);

int knn_fit(
    KNNHandle* handle,
    const float* X,
    int rows,
    int cols,
    const int* y);

int knn_predict(
    KNNHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* predictions);

int knn_load_onnx(
    KNNHandle* handle,
    const char* filename
);

int knn_predict_onnx(
    KNNHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* predictions
);

void knn_destroy(
    KNNHandle* handle);

#ifdef __cplusplus
}
#endif