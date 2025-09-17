#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}
// TODO: __global__
__global__ void kernNaiveScan(int n, int d, int *odata, const int *idata) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    if (index >= (1 << (d - 1))) {
      odata[index] = idata[index - (1 << (d - 1))] + idata[index];
    }
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  int *dev_idata;
  int *dev_odata;
  cudaMalloc((void **)&dev_idata, n * sizeof(int));
  checkCUDAError("naiveScan cudaMalloc dev_idata failed!");
  cudaMalloc((void **)&dev_odata, n * sizeof(int));
  checkCUDAError("naiveScan cudaMalloc dev_odata failed!");
  cudaMemset(dev_idata, 0, n * sizeof(int));
  checkCUDAError("naiveScan cudaMemset dev_idata failed!");
  cudaMemcpy(dev_idata + 1, idata, (n - 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  checkCUDAError("naiveScan cudaMemcpy to device failed!");
  cudaDeviceSynchronize();

  dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

  timer().startGpuTimer();
  // TODO
  for (int d = 1; d <= ilog2ceil(n); d++) {
    cudaMemcpy(dev_odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
    kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(n, d, dev_odata, dev_idata);
    cudaDeviceSynchronize();
    std::swap(dev_odata, dev_idata);
  }
  cudaDeviceSynchronize();
  timer().endGpuTimer();

  cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("naiveScan cudaMemcpy to host failed!");
  cudaDeviceSynchronize();
  cudaFree(dev_odata);
  cudaFree(dev_idata);
}
}  // namespace Naive
}  // namespace StreamCompaction
