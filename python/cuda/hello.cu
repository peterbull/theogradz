#include "cuda_stubs.h"
#include <cstdio>
#include <stdio.h>

__global__ void helloKernel() {
  printf("hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

// general structure for a kernel:
// -> grid (1 per kernel)
//   -> blocks (65535 per dimension)
//   -> threads (1024 per block)
//   -> streaming multiprocessors(SM) (48)
//   -> warp size (32 threads per SM)

// implement a kernel that adds 10 to each position of vector
// `a` and stores it in vector `out`. 1 thread per position.
__device__ void printThread() { printf("thread: %d\n", threadIdx.x); }
__global__ void addTen(float *a, float *out, int n) {
  printThread();
  out[threadIdx.x] = a[threadIdx.x] + 10;
}

int main() {
  float *d_a;
  float *d_out;
  float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out[8];
  cudaMalloc(&d_a, 8 * sizeof(float));
  cudaMalloc(&d_out, 8 * sizeof(float));
  cudaMemcpy(d_a, a, 8 * sizeof(float), cudaMemcpyHostToDevice);
  addTen<<<1, 8>>>(d_a, d_out, 8);
  cudaMemcpy(out, d_out, 8 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 8; i++) {
    printf("array element %d: %f\n", i, out[i]);
  }
  // helloKernel<<<2, 4>>>();
  // helloKernel<<<1028, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}
