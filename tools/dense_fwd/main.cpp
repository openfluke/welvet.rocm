#include "welvet/gaps.hpp"
#include "welvet/rocm/device.hpp"
#include "welvet_c.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

static void usage() {
  std::fprintf(stderr,
               "welvet_dense_fwd (welvet.rocm M0)\n"
               "  --version\n"
               "  --identity\n"
               "  --gaps\n"
               "  --help\n"
               "\n"
               "Leaf FormatNone / train wires land as TRAINING_CHECKLIST ticks "
               "(mirror welvet.cuda → 0.79 via w2a go_v_rocm).\n");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string arg = argv[1];
  if (arg == "--help" || arg == "-h") {
    usage();
    return 0;
  }
  if (arg == "--version") {
    std::printf("%s\n", welvet_version());
    return 0;
  }
  if (arg == "--identity") {
    auto st = welvet::rocm::init_default_device();
    if (!st) {
      std::fprintf(stderr, "init: %s\n", st.message.c_str());
      return 1;
    }
    std::printf("%s\n", welvet_runtime_identity());
    return 0;
  }
  if (arg == "--gaps") {
    welvet::gaps::print_census(std::cout);
    return 0;
  }
  usage();
  return 2;
}
