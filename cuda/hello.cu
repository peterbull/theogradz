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
  float out[n];

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

// broadcast
// Implement a kernel that adds `a` and `b` and stores it in `out`. Inputs `a`
// and `b` are vectors. more threads than positions.
__global__ void addAB(float *a, float *b, float *out, int L) {
  int row = threadIdx.x;
  int col = threadIdx.y;
  if (col < L && row < L) {
    int i = row * L + col;
    out[i] = a[col] + b[row];
  }
}

void runAddAB() {
  float *d_a;
  float *d_b;
  float *d_out;

  int L = 4;
  int n = L * L;

  float a[] = {1, 2, 3, 4};
  float b[] = {1, 2, 3, 4};

  float out[n];

  cudaMalloc(&d_a, L * sizeof(float));
  cudaMalloc(&d_b, L * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));
  cudaMemcpy(d_a, a, L * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, L * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(6, 6);
  dim3 gridDim(1, 1);

  addAB<<<gridDim, blockDim>>>(d_a, d_b, d_out, L);

  cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  printFloatArray(out, n);
}

// blocks(explicit tiles)
// implement a kernel that adds 10 to each position of `a` and stores it in out.
// fewer threads per block than the size of `a`
__global__ void addABlocks(float *a, float *out, int length) {
  extern __shared__ float buf[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;
  printf("local i: %d\n", local_i);
  if (idx < length) {
    buf[local_i] = a[idx];
  }
  // shared buffer is only shared to other threads on a block
  // no race cond. in this one, more to just get familiar with declaring/syncing
  __syncthreads();
  if (idx < length) {
    out[idx] = buf[local_i] + 10;
  }
}

void runAddABlocks() {
  float *d_a;
  float *d_out;
  int n = 8;
  int threadsPerBlock = 4;
  int numBlocks = 2;
  float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out[8];

  int sizeInBytes = n * sizeof(float);

  cudaMalloc(&d_a, sizeInBytes);
  cudaMalloc(&d_out, sizeInBytes);
  cudaMemcpy(d_a, a, sizeInBytes, cudaMemcpyHostToDevice);
  dim3 blockDim(threadsPerBlock);
  dim3 gridDim(numBlocks);
  addABlocks<<<gridDim, blockDim, sizeInBytes>>>(d_a, d_out, n);
  cudaMemcpy(out, d_out, sizeInBytes, cudaMemcpyDeviceToHost);
  printFloatArray(out, n);
}

// pooling
// a kernel that sums together the last 3 positions of `a` and stores it in
// `out` 1 thread per position.
__global__ void aPooling(float *a, float *out, int length) {
  __shared__ float buf[8];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;
  if (idx < length) {
    buf[local_i] = a[idx];
  }
  __syncthreads();

  if (idx < length) {
    float total = buf[local_i];
    if (idx > 0) {
      total += buf[local_i - 1];
    }
    if (idx > 1) {
      total += buf[local_i - 2];
    }
    out[idx] = total;
  }
}

void runAPooling() {
  float *d_a;
  float *d_out;
  int n = 8;
  int threadsPerBlock = n;
  int numBlocks = 1;
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float out[8];

  int sizeInBytes = n * sizeof(float);

  cudaMalloc(&d_a, sizeInBytes);
  cudaMalloc(&d_out, sizeInBytes);
  cudaMemcpy(d_a, a, sizeInBytes, cudaMemcpyHostToDevice);
  dim3 blockDim(threadsPerBlock);
  dim3 gridDim(numBlocks);
  aPooling<<<gridDim, blockDim, sizeInBytes>>>(d_a, d_out, n);
  cudaMemcpy(out, d_out, sizeInBytes, cudaMemcpyDeviceToHost);
  printFloatArray(out, n);
}

// dot product
// a kernel that computes the dot-product of `a` and `b` and stores it in `out`.
// 1 thread per position
__global__ void dotProduct(float *a, float *b, float *out, int length) {
  extern __shared__ float buf[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("idx: %d\n", idx);
  int local_i = threadIdx.x;

  // assign each elementwise result to a location in the shared buffer
  // wait for all threads to finish before proceeding
  buf[local_i] = (idx < length) ? a[idx] * b[idx] : 0.0f;
  __syncthreads();
  // reducer with a stride(s) to represent how far apart
  // op items are in the buffer. bitshift each loop to cut the space in half.
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (local_i < s) {
      printf("local i: %d :: op1: %f :: op2: %f :: s: %d :: blockdim: %d\n",
             local_i, buf[local_i], buf[local_i + s], local_i, blockDim.x);
      buf[local_i] += buf[local_i + s];
    }
    __syncthreads();
  }

  if (idx == 0) {
    atomicAdd(&out[0], buf[0]);
  }
}

void runDotProduct() {
  float *d_a;
  float *d_b;
  float *d_out;
  int n = 8;
  int threadsPerBlock = n;
  int numBlocks = 1;
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float b[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float out[1];
  int sizeInBytes = n * sizeof(float);
  cudaMalloc(&d_a, sizeInBytes);
  cudaMalloc(&d_b, sizeInBytes);
  cudaMalloc(&d_out, sizeof(float));
  cudaMemcpy(d_a, a, sizeInBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeInBytes, cudaMemcpyHostToDevice);
  dim3 blockDim(threadsPerBlock);
  dim3 gridDim(numBlocks);
  dotProduct<<<gridDim, blockDim, sizeInBytes>>>(d_a, d_b, d_out, n);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  printFloatArray(out, 1);
}
// 1d convolution
// kernel that computes a 1D convolution between `a` and `b` and stores it in
// `out`. handle the general case
__global__ void oneDConv(float *a, float *b, float *out, int length, int l2) {
  __shared__ float buf[12];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  if (idx < length) {
    // buf[tid] =
  }
}

void runOneDConv() {
  float *d_a;
  float *d_b;
  float *d_out;
  const int n = 8;
  int threadsPerBlock = n;
  const int numBlocks = 3;
  const int l1 = 15;
  const int l2 = 4;
  float a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  float b[] = {0, 1, 2, 3};
  const int outN = 18;
  float out[outN];
  int sizeA = l1 * sizeof(float);
  int sizeB = l2 * sizeof(float);
  int sizeOut = outN * sizeof(float);
  int sharedMemBytes = (threadsPerBlock + l2 - 1) * sizeof(float);
  cudaMalloc(&d_a, sizeA);
  cudaMalloc(&d_b, sizeB);
  cudaMalloc(&d_out, sizeOut);

  cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);
  dim3 blockDim(threadsPerBlock);
  dim3 gridDim(numBlocks);
  oneDConv<<<gridDim, blockDim, sharedMemBytes>>>(d_a, d_b, d_out, l1, l2);
  cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  printFloatArray(out, 1);
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
  // runAddTen2dSquare();
  // cudaDeviceSynchronize();
  // runAddABlocks();
  // runAPooling();
  // runDotProduct();
  runOneDConv();
  return 0;
}
