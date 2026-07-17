#include "cuda_utils.hpp"

#include <stdexcept>

namespace cuda_utils {

void Check(cudaError_t error)
{
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

}

namespace cuda_utils {

void* Malloc(std::size_t bytes)
{
    void* ptr = nullptr;

    Check(cudaMalloc(&ptr, bytes));

    return ptr;
}

}

namespace cuda_utils {

void Free(void* ptr)
{
    if (ptr != nullptr)
        Check(cudaFree(ptr));
}

}

namespace cuda_utils {

void CopyHostToDevice(
    void* device,
    const void* host,
    std::size_t bytes)
{
    Check(cudaMemcpy(
        device,
        host,
        bytes,
        cudaMemcpyHostToDevice));
}

}

namespace cuda_utils {

void CopyDeviceToHost(
    void* host,
    const void* device,
    std::size_t bytes)
{
    Check(cudaMemcpy(
        host,
        device,
        bytes,
        cudaMemcpyDeviceToHost));
}

}
