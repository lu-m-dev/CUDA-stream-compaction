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
void dev_scan(int n, int *dev_odata) {
  int N = (1 << ilog2ceil(n));

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
      N, dev_bArr, dev_idata);
  cudaDeviceSynchronize();
  checkCUDAError("efficientCompact kernMapToBoolean failed!");
  cudaMemcpy(dev_scanResult, dev_bArr, N * sizeof(int),
             cudaMemcpyDeviceToDevice);
  checkCUDAError("efficientCompact cudaMemcpy dev_scanResult failed!");
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
}  // namespace Efficient
}  // namespace StreamCompaction
