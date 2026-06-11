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
inline cudaError_t cudaDeviceSynchronize() { return 0; }
inline cudaError_t cudaConfigureCall(dim3, dim3, unsigned long = 0, void* = 0) { return 0; }
