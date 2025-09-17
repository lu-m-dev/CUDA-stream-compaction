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
  checkCUDAError("efficientScan cudaMemset dev_idata failed!");
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
  cudaMemset(dev_odata + N - 1, 0, sizeof(int));
  checkCUDAError("efficientScan cudaMemset dev_odata after UpSweep failed!");
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
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
  timer().startGpuTimer();
  // TODO
  timer().endGpuTimer();
  return -1;
}
}  // namespace Efficient
}  // namespace StreamCompaction
