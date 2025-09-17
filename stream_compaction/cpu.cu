#include <cstdio>

#include "common.h"
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {
using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer() {
  static PerformanceTimer timer;
  return timer;
}

/**
 * CPU scan (prefix sum).
 * For performance analysis, this is supposed to be a simple for loop.
 * (Optional) For better understanding before starting moving to GPU, you can
 * simulate your GPU scan in this function first.
 */
void scan(int n, int *odata, const int *idata) {
  timer().startCpuTimer();
  // TODO
  odata[0] = 0;
  for (unsigned int i = 1; i < n; i++) {
    odata[i] = odata[i - 1] + idata[i - 1];
  }
  timer().endCpuTimer();
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
  timer().startCpuTimer();
  // TODO
  int count = 0;
  for (unsigned int i = 0; i < n; i++) {
    if (idata[i]) {
      odata[count] = idata[i];
      count++;
    }
  }
  timer().endCpuTimer();
  return count;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
  // TODO
  int *b_arr = new int[n];
  int *scan_result = new int[n];
  timer().startCpuTimer();
  int count = 0;
  for (unsigned int i = 0; i < n; i++) {
    b_arr[i] = (idata[i] == 0) ? 0 : 1;
  }
  scan(n, scan_result, b_arr);
  for (unsigned int i = 0; i < n; i++) {
    if (b_arr[i]) {
      odata[scan_result[i]] = idata[i];
      count++;
    }
  }
  timer().endCpuTimer();
  delete[] b_arr;
  delete[] scan_result;
  b_arr = nullptr;
  scan_result = nullptr;
  return count;
}
}  // namespace CPU
}  // namespace StreamCompaction
