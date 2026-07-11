#include "cuda_stubs.h"
#include <stdio.h>

__global__ void helloKernel() {
  printf("hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

// general structure for a kernel:
// -> grid (1 per kernel, many dimensions)
//   -> blocks (65535 per dimension)
//   -> threads (1024 per block)
//   -> streaming multiprocessors(SM) (48)
//   -> warp size (32 threads per SM)

__device__ void printKernelInfo() {
  printf("thread idx x: %d\n", threadIdx.x);
  printf("thread idx y: %d\n", threadIdx.y);
  printf("block idx x: %d\n", blockIdx.x);
  printf("block idx y: %d\n", blockIdx.y);
  printf("block dim x: %d\n", blockDim.x);
  printf("block dim y: %d\n", blockDim.y);
}
void printFloatArray(float *arr, int n) {
  for (int i = 0; i < n; i++) {
    printf("array element %d: %f\n", i, arr[i]);
  }
}

// implement a kernel that adds 10 to each position of vector
// `a` and stores it in vector `out`. 1 thread per position.
__global__ void addTen(float *a, float *out) {
  out[threadIdx.x] = a[threadIdx.x] + 10;
}
void runAddTen() {
  float *d_a;
  float *d_out;
  float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out[8];
  cudaMalloc(&d_a, 8 * sizeof(float));
  cudaMalloc(&d_out, 8 * sizeof(float));
  cudaMemcpy(d_a, a, 8 * sizeof(float), cudaMemcpyHostToDevice);
  addTen<<<1, 8>>>(d_a, d_out);
  cudaMemcpy(out, d_out, 8 * sizeof(float), cudaMemcpyDeviceToHost);
  printf("back on device");
  printFloatArray(out, 8);
}

// a kernel that adds together each position of `a` and `b`
// and stores it in `out`. 1 thread per position.
__global__ void combineAB(float *a, float *b, float *out) {
  out[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
};

void runCombineAB() {
  float *d_a, *d_b;
  float *d_out;
  int len = 5;
  int size_f = len * sizeof(float);
  float a[] = {0, 1, 2, 3, 4};
  float b[] = {5, 6, 7, 8, 9};
  float out[len];
  cudaMalloc(&d_a, size_f);
  cudaMalloc(&d_b, size_f);
  cudaMalloc(&d_out, size_f);
  cudaMemcpy(d_a, a, size_f, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size_f, cudaMemcpyHostToDevice);
  combineAB<<<1, len>>>(d_a, d_b, d_out);
  cudaMemcpy(out, d_out, size_f, cudaMemcpyDeviceToHost);
  printFloatArray(out, len);
}

// guarding
// implement a kernel that adds 10 to each position of `a` and stores it in
// `out`. more threads than positions.
__global__ void addTenManyThreads(float *a, float *out, int length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    printf("adding in flat index:%d, thread index: %d\n", idx, threadIdx.x);
    out[idx] = a[idx] + 10;
  }
}

void runAddTenManyThreads() {
  float *d_a;
  float *d_out;
  float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out[8];
  cudaMalloc(&d_a, 8 * sizeof(float));
  cudaMalloc(&d_out, 8 * sizeof(float));
  cudaMemcpy(d_a, a, 8 * sizeof(float), cudaMemcpyHostToDevice);
  addTenManyThreads<<<1, 32>>>(d_a, d_out, 8);
  cudaMemcpy(out, d_out, 8 * sizeof(float), cudaMemcpyDeviceToHost);
  printf("back on device");
  printFloatArray(out, 8);
}

// 2d map
// implement a kernel that adds 10 to each position of `a` and stores it in
// `out`. Input `a` is 2D and square. more threads than positions.
__global__ void addTen2dSquare(float *a, float *out, int width, int height) {
  printKernelInfo();
  // (4,3)
  //
  // [1,2,3,4]
  // [5,6,7,8]
  // [9,10,11,12]
  int row = threadIdx.y;
  int col = threadIdx.x;
  if (col < width && row < height) { // guard each axis before flattening
    int i = row * width + col;
    out[i] = a[i] + 10;
  }
}

void runAddTen2dSquare() {
  float *d_a;
  float *d_out;
  int width = 4;
  int height = 3;
  int n = width * height;

  float a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float out[12];

  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
  dim3 blockDim(
      4, 4); // set to slightly bigger than the array to practice guarding
  dim3 gridDim(1, 1);
  addTen2dSquare<<<gridDim, blockDim>>>(d_a, d_out, width, height);
  cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  printFloatArray(out, n);
}

int main() {
  // float *d_a;
  // float *d_out;
  // float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
  // float out[8];
  // cudaMalloc(&d_a, 8 * sizeof(float));
  // cudaMalloc(&d_out, 8 * sizeof(float));
  // cudaMemcpy(d_a, a, 8 * sizeof(float), cudaMemcpyHostToDevice);
  // addTen<<<1, 8>>>(d_a, d_out, 8);
  // cudaMemcpy(out, d_out, 8 * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < 8; i++) {
  //   printf("array element %d: %f\n", i, out[i]);
  // }
  // helloKernel<<<2, 4>>>();
  // helloKernel<<<1028, 4>>>();
  // runAddTen();
  // runCombineAB();
  runAddTen2dSquare();
  cudaDeviceSynchronize();
  return 0;
}
