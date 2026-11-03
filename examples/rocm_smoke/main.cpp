#include "welvet/gaps.hpp"
#include "welvet/rocm/buffer.hpp"
#include "welvet/rocm/device.hpp"
#include "welvet/rocm/kernels.hpp"
#include "welvet_c.h"

#include <cstdio>
#include <iostream>
#include <vector>

int main() {
  std::printf("welvet.rocm %s\n", welvet_version());

  auto st = welvet::rocm::init_default_device();
  if (!st) {
    std::fprintf(stderr, "init: %s\n", st.message.c_str());
    return 1;
  }
  std::printf("runtime: %s\n", welvet::rocm::runtime_identity().c_str());

  std::cout << "devices:\n";
  for (const auto& d : welvet::rocm::list_devices()) {
    std::cout << "  [" << d.index << "] " << d.name << " gcn=" << d.gcn_arch_major << "."
              << d.gcn_arch_minor << " mem=" << d.total_global_mem
              << " cus=" << d.multi_processor_count << "\n";
  }

  constexpr std::size_t n = 4;
  welvet::rocm::Buffer x_buf;
  welvet::rocm::Buffer y_buf;
  st = welvet::rocm::buffer_alloc(n * sizeof(float), &x_buf);
  if (!st) {
    std::fprintf(stderr, "buffer x: %s\n", st.message.c_str());
    return 1;
  }
  st = welvet::rocm::buffer_alloc(n * sizeof(float), &y_buf);
  if (!st) {
    welvet::rocm::buffer_release(&x_buf);
    std::fprintf(stderr, "buffer y: %s\n", st.message.c_str());
    return 1;
  }

  const float host_in[n] = {1, 2, 3, 4};
  float host_out[n] = {};
  st = welvet::rocm::buffer_upload(&x_buf, host_in, sizeof(host_in));
  if (!st) {
    std::fprintf(stderr, "upload: %s\n", st.message.c_str());
    return 1;
  }

  // y = 2*x + 1 → [3, 5, 7, 9]
  st = welvet::rocm::launch_smoke_saxpy(static_cast<const float*>(x_buf.device),
                                        static_cast<float*>(y_buf.device), 2.0f, 1.0f, n);
  if (!st) {
    std::fprintf(stderr, "saxpy: %s\n", st.message.c_str());
    welvet::rocm::buffer_release(&x_buf);
    welvet::rocm::buffer_release(&y_buf);
    return 1;
  }

  st = welvet::rocm::buffer_download(y_buf, host_out, sizeof(host_out));
  welvet::rocm::buffer_release(&x_buf);
  welvet::rocm::buffer_release(&y_buf);
  if (!st) {
    std::fprintf(stderr, "download: %s\n", st.message.c_str());
    return 1;
  }
  std::printf("smoke saxpy: [%g, %g, %g, %g] (expect 3,5,7,9)\n", host_out[0], host_out[1],
              host_out[2], host_out[3]);

  std::cout << "\n";
  welvet::gaps::print_census(std::cout);
  const bool ok =
      host_out[0] == 3.f && host_out[1] == 5.f && host_out[2] == 7.f && host_out[3] == 9.f;
  return ok ? 0 : 1;
}
