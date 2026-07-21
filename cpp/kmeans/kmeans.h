#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KMeansHandle KMeansHandle;

KMeansHandle* kmeans_create(
    int n_clusters,
    int max_iters,
    float tolerance
);

int kmeans_fit(
    KMeansHandle* handle,
    const float* X,
    int rows,
    int cols
);

int kmeans_predict(
    KMeansHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* labels
);

void kmeans_destroy(
    KMeansHandle* handle
);

#ifdef __cplusplus
}
#endif