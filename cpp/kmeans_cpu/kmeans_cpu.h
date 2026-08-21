#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KMeansCPUHandle KMeansCPUHandle;

KMeansCPUHandle* kmeans_cpu_create(
    int k,
    int max_iters,
    float tol
);

int kmeans_cpu_fit(
    KMeansCPUHandle* handle,
    const float* X,
    int rows,
    int cols
);

int kmeans_cpu_predict(
    KMeansCPUHandle* handle,
    const float* X,
    int rows,
    int* predictions
);

void kmeans_cpu_free(
    KMeansCPUHandle* handle
);

#ifdef __cplusplus
}
#endif

#endif