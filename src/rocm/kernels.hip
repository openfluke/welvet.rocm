#include "welvet/rocm/kernels.hpp"

#include <hip/hip_runtime.h>

#include <string>

namespace {

__global__ void welvet_smoke_saxpy_kernel(const float* x, float* y, float a, float b,
                                          unsigned long long n) {
  const unsigned long long gid =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  y[gid] = a * x[gid] + b;
}

welvet::Status hip_status(hipError_t err, const char* what) {
  if (err == hipSuccess) return welvet::Status::Ok();
  return welvet::Status::Err(welvet::StatusCode::Internal,
                             std::string(what) + ": " + hipGetErrorString(err));
}

} // namespace

namespace welvet::rocm {

Status launch_smoke_saxpy(const float* x_dev, float* y_dev, float a, float b, std::size_t n) {
  if (!x_dev || !y_dev) return Status::Err(StatusCode::InvalidArgument, "null device ptr");
  if (n == 0) return Status::Err(StatusCode::InvalidArgument, "empty saxpy");
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  welvet_smoke_saxpy_kernel<<<blocks, kThreads>>>(x_dev, y_dev, a, b,
                                                  static_cast<unsigned long long>(n));
  auto st = hip_status(hipGetLastError(), "welvet_smoke_saxpy launch");
  if (!st) return st;
  return hip_status(hipDeviceSynchronize(), "welvet_smoke_saxpy sync");
}

Status launch_dense_matvec_f32(const float*, const float*, float*, std::uint32_t, std::uint32_t) {
  return Status::Gap("Dense MatVec HIP path — see docs/TRAINING_CHECKLIST.md (leaf FormatNone)");
}

} // namespace welvet::rocm
