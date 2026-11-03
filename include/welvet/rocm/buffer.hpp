#pragma once

#include <cstddef>

#include "welvet/status.hpp"

namespace welvet::rocm {

// Opaque device buffer owned by the engine.
struct Buffer {
  void* device = nullptr; // device pointer from hipMalloc
  std::size_t bytes = 0;
  bool device_owned = true;
};

Status buffer_alloc(std::size_t bytes, Buffer* out);
Status buffer_upload(Buffer* buf, const void* host, std::size_t bytes);
Status buffer_download(const Buffer& buf, void* host, std::size_t bytes);
void buffer_release(Buffer* buf);

} // namespace welvet::rocm
