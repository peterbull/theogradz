#include "cuda_stubs.h"
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
float *d_a;

// Implement a "kernel" (GPU function) that adds 10 to each position of vector
// `a` and stores it in vector `out`. 1 thread per position.
__global__ void addTen(float *a, float *out, int n) {}
int main() {
  float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
  cudaMalloc(&d_a, 8 * sizeof(float));
  cudaMemcpy(d_a, a, sizeof(float), cudaMemcpyHostToDevice);
  // helloKernel<<<2, 4>>>();
  // helloKernel<<<1028, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}
