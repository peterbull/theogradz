#include <stdio.h>

__global__ void helloKernel() {
  printf("hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
  helloKernel<<<2, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}
