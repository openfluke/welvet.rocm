#include "welvet/rocm/device.hpp"

#include <hip/hip_runtime.h>

#include <mutex>
#include <sstream>

namespace {

std::mutex g_device_mu;
int g_device = -1;
std::string g_identity;

welvet::Status hip_status(hipError_t err, const char* what) {
  if (err == hipSuccess) return welvet::Status::Ok();
  return welvet::Status::Err(welvet::StatusCode::Internal,
                             std::string(what) + ": " + hipGetErrorString(err));
}

welvet::rocm::DeviceInfo info_from(int index) {
  welvet::rocm::DeviceInfo i;
  i.index = index;
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, index) != hipSuccess) {
    i.name = "(unavailable)";
    return i;
  }
  i.name = prop.name;
  i.gcn_arch_major = prop.major;
  i.gcn_arch_minor = prop.minor;
  i.total_global_mem = prop.totalGlobalMem;
  i.multi_processor_count = prop.multiProcessorCount;
  return i;
}

} // namespace

namespace welvet::rocm {

Status init_default_device() {
  std::lock_guard<std::mutex> lock(g_device_mu);
  if (g_device >= 0) return Status::Ok();

  int count = 0;
  auto st = hip_status(hipGetDeviceCount(&count), "hipGetDeviceCount");
  if (!st) return st;
  if (count <= 0) {
    return Status::Err(StatusCode::Unsupported, "no HIP/ROCm device available");
  }

  st = hip_status(hipSetDevice(0), "hipSetDevice");
  if (!st) return st;

  g_device = 0;
  auto info = info_from(0);
  std::ostringstream os;
  os << "rocm:" << info.name << ":idx=" << info.index << ":gcn=" << info.gcn_arch_major << "."
     << info.gcn_arch_minor << ":mem=" << info.total_global_mem;
  g_identity = os.str();
  return Status::Ok();
}

bool available() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) return false;
  return count > 0;
}

Status default_device_info(DeviceInfo* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null DeviceInfo");
  auto st = init_default_device();
  if (!st) return st;
  *out = info_from(g_device);
  return Status::Ok();
}

std::vector<DeviceInfo> list_devices() {
  std::vector<DeviceInfo> out;
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) return out;
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    out.push_back(info_from(i));
  }
  return out;
}

std::string runtime_identity() {
  (void)init_default_device();
  std::lock_guard<std::mutex> lock(g_device_mu);
  return g_identity.empty() ? std::string("rocm:unavailable") : g_identity;
}

} // namespace welvet::rocm
