#pragma once
#include <iostream>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline void printGpuDriverInfo()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}


#define GPUTIMER(...)                                                                                                           \
{                                                                                                                               \
    using namespace std;                                                                                                        \
    chrono::system_clock::time_point start, end;                                                                                \
                                                                                                                                \
    start = chrono::system_clock::now();                                                                                        \
                                                                                                                                \
    __VA_ARGS__;                                                                                                                \
    cudaDeviceSynchronize();                                                                                                    \
                                                                                                                                \
    end = chrono::system_clock::now();                                                                                          \
                                                                                                                                \
    double time = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);               \
    printf("time %lf[ms]\n", time);                                                                                             \
}

inline __global__ void testGPUfun(float* A, float* B, float* C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}