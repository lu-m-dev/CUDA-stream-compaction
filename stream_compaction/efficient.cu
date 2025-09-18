#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}

__global__ void kernUpSweep(const int N, const int numNodes, const int stepSize,
                            int *odata) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= numNodes) return;
  int k = stepSize * (index + 1) - 1;
  int left = k - (stepSize >> 1);
  odata[k] = odata[k] + odata[left];
}

__global__ void kernDownSweep(const int N, const int numNodes,
                              const int stepSize, int *odata) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numNodes) return;
  int k = stepSize * (index + 1) - 1;
  int left = k - (stepSize >> 1);
  int temp = odata[left];
  odata[left] = odata[k];
  odata[k] = temp + odata[k];
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  // TODO
  int N = (1 << ilog2ceil(n));
  int *dev_odata;
  cudaMalloc((void **)&dev_odata, N * sizeof(int));
  checkCUDAError("efficientScan cudaMalloc dev_odata failed!");
  cudaMemset(dev_odata, 0, N * sizeof(int));
  checkCUDAError("efficientScan cudaMemset dev_odata failed!");
  cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("efficientScan cudaMemcpy to device failed!");
  cudaDeviceSynchronize();

  timer().startGpuTimer();
  for (int d = 0; d < ilog2ceil(n); d++) {
    int numNodes = N >> (d + 1);
    dim3 fullBlocksPerGrid((numNodes + blockSize - 1) / blockSize);
    kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(N, numNodes, 1 << (d + 1),
                                                  dev_odata);
    checkCUDAError("efficientScan kernUpSweep failed!");
    cudaDeviceSynchronize();
  }
  cudaMemset(dev_odata + N - 1, 0, sizeof(int));  // set last element to 0
  checkCUDAError("efficientScan cudaMemset after UpSweep failed!");
  cudaDeviceSynchronize();
  for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
    int numNodes = N >> (d + 1);
    dim3 fullBlocksPerGrid((numNodes + blockSize - 1) / blockSize);
    kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(N, numNodes, 1 << (d + 1),
                                                    dev_odata);
    checkCUDAError("efficientScan kernDownSweep failed!");
    cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();
  timer().endGpuTimer();
  cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("efficientScan cudaMemcpy to host failed!");
  cudaFree(dev_odata);
}

/**
 * Performs scan on device array dev_odata in place.
 */
void dev_scan(int N, int *dev_odata) {
  for (int d = 0; d < ilog2ceil(N); d++) {
    int numNodes = N >> (d + 1);
    dim3 fullBlocksPerGrid((numNodes + blockSize - 1) / blockSize);
    kernUpSweep<<<fullBlocksPerGrid, blockSize>>>(N, numNodes, 1 << (d + 1),
                                                  dev_odata);
    checkCUDAError("efficientScan kernUpSweep failed!");
    cudaDeviceSynchronize();
  }
  cudaMemset(dev_odata + N - 1, 0, sizeof(int));  // set last element to 0
  checkCUDAError("efficientScan cudaMemset after UpSweep failed!");
  cudaDeviceSynchronize();
  for (int d = ilog2ceil(N) - 1; d >= 0; d--) {
    int numNodes = N >> (d + 1);
    dim3 fullBlocksPerGrid((numNodes + blockSize - 1) / blockSize);
    kernDownSweep<<<fullBlocksPerGrid, blockSize>>>(N, numNodes, 1 << (d + 1),
                                                    dev_odata);
    checkCUDAError("efficientScan kernDownSweep failed!");
    cudaDeviceSynchronize();
  }
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
  int *dev_idata;
  int *dev_odata;
  int *dev_bArr;
  int *dev_scanResult;
  int N = (1 << ilog2ceil(n));
  cudaMalloc((void **)&dev_idata, N * sizeof(int));
  checkCUDAError("efficientCompact cudaMalloc dev_idata failed!");
  cudaMalloc((void **)&dev_odata, N * sizeof(int));
  checkCUDAError("efficientCompact cudaMalloc dev_odata failed!");
  cudaMalloc((void **)&dev_bArr, N * sizeof(int));
  checkCUDAError("efficientCompact cudaMalloc dev_bArr failed!");
  cudaMalloc((void **)&dev_scanResult, N * sizeof(int));
  checkCUDAError("efficientCompact cudaMalloc dev_scanResult failed!");
  cudaMemset(dev_idata, 0, N * sizeof(int));
  checkCUDAError("efficientCompact cudaMemset dev_idata failed!");
  cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("efficientCompact cudaMemcpy dev_idata failed!");
  cudaMemcpy(dev_odata, idata, N * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("efficientCompact cudaMemcpy dev_odata failed!");
  cudaDeviceSynchronize();

  dim3 fullBlocksPerGrid = (N + blockSize - 1) / blockSize;

  timer().startGpuTimer();  // start timer
  // TODO
  StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(
      N, dev_bArr, dev_scanResult, dev_idata);
  checkCUDAError("efficientCompact kernMapToBoolean failed!");
  cudaDeviceSynchronize();
  dev_scan(N, dev_scanResult);
  cudaDeviceSynchronize();
  checkCUDAError("efficientCompact scan failed!");

  StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(
      N, dev_odata, dev_idata, dev_bArr, dev_scanResult);
  cudaDeviceSynchronize();
  checkCUDAError("efficientCompact kernScatter failed!");

  timer().endGpuTimer();  // end timer

  int lastElement, numElements;
  cudaMemcpy(&lastElement, dev_idata + n - 1, sizeof(int),
             cudaMemcpyDeviceToHost);
  checkCUDAError("efficientCompact cudaMemcpy lastElement failed!");
  cudaMemcpy(&numElements, dev_scanResult + n - 1, sizeof(int),
             cudaMemcpyDeviceToHost);
  checkCUDAError("efficientCompact cudaMemcpy numElements failed!");
  numElements += (lastElement == 0) ? 0 : 1;
  cudaMemcpy(odata, dev_odata, numElements * sizeof(int),
             cudaMemcpyDeviceToHost);
  checkCUDAError("efficientCompact cudaMemcpy to host failed!");
  cudaFree(dev_idata);
  cudaFree(dev_odata);
  cudaFree(dev_bArr);
  cudaFree(dev_scanResult);
  return numElements;
}

/**
 * Compute the
 * b array - bit value at bit position,
 * e array - negation of b array,
 * f array - index for false values,
 * f array is now a copy of e array for subsequent in-place scan input.
 */
__global__ void kernComputeBEF(const int n, int *b, int *e, int *f,
                               const int *idata, const int lsb) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    int bitmask = 1 << lsb;
    int bVal = (idata[index] & bitmask) >> lsb;
    int eVal = 1 - bVal;
    b[index] = bVal;
    e[index] = eVal;
    f[index] = eVal;
  }
}

/**
 * Compute the t array - index for true values.
 */
__global__ void kernComputeT(const int n, int *t, const int *f,
                             const int totalFalses) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    t[index] = index - f[index] + totalFalses;
  }
}

/**
 * Compute the d array - destination index.
 */
__global__ void kernComputeD(const int n, int *d, const int *b, const int *t,
                             int *f) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    d[index] = b[index] ? t[index] : f[index];
  }
}

__global__ void kernRadixScatter(const int n, int *odata, const int *idata,
                                 const int *d) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    odata[d[index]] = idata[index];
  }
}

/**
 * Performs radix sort on idata, storing the result into odata.
 */
void sort(int n, int *odata, const int *idata) {
  int *dev_idata, *dev_odata, *dev_b, *dev_e, *dev_f, *dev_t, *dev_d;
  int N = (1 << ilog2ceil(n));
  cudaMalloc((void **)&dev_idata, n * sizeof(int));
  checkCUDAError("efficientSort cudaMalloc dev_idata failed!");
  cudaMalloc((void **)&dev_odata, n * sizeof(int));
  checkCUDAError("efficientSort cudaMalloc dev_odata failed!");
  cudaMalloc((void **)&dev_b, n * sizeof(int));
  checkCUDAError("efficientSort cudaMalloc dev_b failed!");
  cudaMalloc((void **)&dev_e, n * sizeof(int));
  checkCUDAError("efficientSort cudaMalloc dev_e failed!");
  cudaMalloc((void **)&dev_f, N * sizeof(int));
  checkCUDAError("efficientSort cudaMalloc dev_f failed!");
  cudaMalloc((void **)&dev_t, n * sizeof(int));
  checkCUDAError("efficientSort cudaMalloc dev_t failed!");
  cudaMalloc((void **)&dev_d, n * sizeof(int));
  checkCUDAError("efficientSort cudaMalloc dev_d failed!");
  cudaMemset(dev_f, 0, N * sizeof(int));
  checkCUDAError("efficientSort cudaMemset dev_f failed!");
  cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("efficientSort cudaMemcpy dev_idata failed!");
  cudaDeviceSynchronize();
  dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
  timer().startGpuTimer();  // start timer
  for (int lsb = 0; lsb < 8 * sizeof(int); lsb++) {
    kernComputeBEF<<<fullBlocksPerGrid, blockSize>>>(n, dev_b, dev_e, dev_f,
                                                     dev_idata, lsb);
    checkCUDAError("efficientSort kernMapToBit failed!");
    cudaDeviceSynchronize();
    dev_scan(N, dev_f);
    cudaDeviceSynchronize();
    int totalFalses, lastFalse;
    cudaMemcpy(&totalFalses, dev_f + n - 1, sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastFalse, dev_e + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
    totalFalses += lastFalse;
    kernComputeT<<<fullBlocksPerGrid, blockSize>>>(n, dev_t, dev_f,
                                                   totalFalses);
    checkCUDAError("efficientSort kernComputeT failed!");
    cudaDeviceSynchronize();
    kernComputeD<<<fullBlocksPerGrid, blockSize>>>(n, dev_d, dev_b, dev_t,
                                                   dev_f);
    checkCUDAError("efficientSort kernComputeD failed!");
    cudaDeviceSynchronize();
    kernRadixScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata,
                                                       dev_d);
    std::swap(dev_idata, dev_odata);
  }
  cudaDeviceSynchronize();
  timer().endGpuTimer();  // end timer
  cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("efficientSort cudaMemcpy to host failed!");
  cudaFree(dev_idata);
  cudaFree(dev_odata);
  cudaFree(dev_b);
  cudaFree(dev_e);
  cudaFree(dev_f);
  cudaFree(dev_t);
  cudaFree(dev_d);
}
}  // namespace Efficient
}  // namespace StreamCompaction
