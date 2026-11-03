#include "welvet_c.h"

#include "welvet/gaps.hpp"
#include "welvet/rocm/device.hpp"
#include "welvet/status.hpp"

#include <cstring>
#include <string>

namespace {

welvet_status to_c(const welvet::Status& st) {
  welvet_status out{};
  out.code = static_cast<int32_t>(st.code);
  std::strncpy(out.message, st.message.c_str(), sizeof(out.message) - 1);
  out.message[sizeof(out.message) - 1] = '\0';
  return out;
}

} // namespace

extern "C" {

const char* welvet_version(void) {
  // M0 ROCm/HIP host scaffold — engine board 0/100. Climb TRAINING_CHECKLIST → cuda 0.79.
  return "0.00";
}

const char* welvet_runtime_identity(void) {
  static thread_local std::string id;
  id = welvet::rocm::runtime_identity();
  return id.c_str();
}

welvet_status welvet_rocm_init(void) { return to_c(welvet::rocm::init_default_device()); }

int32_t welvet_rocm_available(void) { return welvet::rocm::available() ? 1 : 0; }

int32_t welvet_gaps_count(void) { return static_cast<int32_t>(welvet::gaps::count_gaps()); }

int32_t welvet_gaps_done(void) { return static_cast<int32_t>(welvet::gaps::count_done()); }

} // extern "C"
