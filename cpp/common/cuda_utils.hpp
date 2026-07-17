#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace cuda_utils {

void Check(cudaError_t error);

void* Malloc(std::size_t bytes);

void Free(void* ptr);

void CopyHostToDevice(
    void* device,
    const void* host,
    std::size_t bytes);

void CopyDeviceToHost(
    void* host,
    const void* device,
    std::size_t bytes);

}
