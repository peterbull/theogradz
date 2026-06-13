// fake cuda builtins for dev on non cuda machine
#pragma once

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};

extern __thread dim3 threadIdx;
extern __thread dim3 blockIdx;
extern __thread dim3 blockDim;
extern __thread dim3 gridDim;

#define __global__
#define __device__
#define __host__
#define __shared__

typedef int cudaError_t;
#define cudaSuccess 0

inline cudaError_t cudaDeviceSynchronize() { return 0; }
inline cudaError_t cudaConfigureCall(dim3, dim3, unsigned long = 0, void* = 0) { return 0; }

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
};

// templated so float**, int** etc. all match
template<typename T>
inline cudaError_t cudaMalloc(T** ptr, unsigned long size) { return 0; }
inline cudaError_t cudaFree(void* ptr) { return 0; }
template<typename T>
inline cudaError_t cudaMemcpy(T* dst, const T* src, unsigned long size, cudaMemcpyKind kind) { return 0; }


