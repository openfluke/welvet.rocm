#include "welvet/rocm/buffer.hpp"

#include "welvet/rocm/device.hpp"

#include <hip/hip_runtime.h>

namespace {

welvet::Status hip_status(hipError_t err, const char* what) {
  if (err == hipSuccess) return welvet::Status::Ok();
  return welvet::Status::Err(welvet::StatusCode::Internal,
                             std::string(what) + ": " + hipGetErrorString(err));
}

} // namespace

namespace welvet::rocm {

Status buffer_alloc(std::size_t bytes, Buffer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null Buffer");
  if (bytes == 0) return Status::Err(StatusCode::InvalidArgument, "zero-size buffer");

  auto st = init_default_device();
  if (!st) return st;

  void* ptr = nullptr;
  st = hip_status(hipMalloc(&ptr, bytes), "hipMalloc");
  if (!st) return st;

  out->device = ptr;
  out->bytes = bytes;
  out->device_owned = true;
  return Status::Ok();
}

Status buffer_upload(Buffer* buf, const void* host, std::size_t bytes) {
  if (!buf || !buf->device) return Status::Err(StatusCode::InvalidArgument, "null buffer");
  if (!host) return Status::Err(StatusCode::InvalidArgument, "null host");
  if (bytes > buf->bytes) return Status::Err(StatusCode::InvalidArgument, "upload overflow");
  return hip_status(hipMemcpy(buf->device, host, bytes, hipMemcpyHostToDevice), "hipMemcpy H2D");
}

Status buffer_download(const Buffer& buf, void* host, std::size_t bytes) {
  if (!buf.device) return Status::Err(StatusCode::InvalidArgument, "null buffer");
  if (!host) return Status::Err(StatusCode::InvalidArgument, "null host");
  if (bytes > buf.bytes) return Status::Err(StatusCode::InvalidArgument, "download overflow");
  return hip_status(hipMemcpy(host, buf.device, bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
}

void buffer_release(Buffer* buf) {
  if (!buf || !buf->device) return;
  if (buf->device_owned) {
    (void)hipFree(buf->device);
  }
  buf->device = nullptr;
  buf->bytes = 0;
}

} // namespace welvet::rocm
