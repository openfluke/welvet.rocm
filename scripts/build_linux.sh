#!/usr/bin/env bash
# Linux AMD ROCm / HIP host build for welvet.rocm.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "welvet.rocm build_linux.sh expects Linux (got $(uname -s))" >&2
  exit 1
fi

# Prefer system ROCm (Fedora packages) or an explicit tree.
if [[ -z "${ROCM_PATH:-}" ]]; then
  for cand in /usr /opt/rocm /opt/rocm-7.1.0; do
    if [[ -x "${cand}/bin/hipcc" ]] || [[ -x /usr/bin/hipcc && "$cand" == /usr ]]; then
      export ROCM_PATH="$cand"
      break
    fi
  done
fi

if ! command -v hipcc >/dev/null 2>&1; then
  echo "hipcc not found. Install AMD ROCm (hipcc / rocm-hip-devel), or set ROCM_PATH." >&2
  exit 1
fi

export PATH="${ROCM_PATH:-/usr}/bin:${PATH}"
# hip-clang lives under /usr/lib64/rocm/llvm on Fedora.
if [[ -d /usr/lib64/rocm/llvm/bin ]]; then
  export PATH="/usr/lib64/rocm/llvm/bin:${PATH}"
fi

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/usr};${CMAKE_PREFIX_PATH:-}"
)

# Optional arch override: gfx1100, gfx1030, gfx906, …
if [[ -n "${CMAKE_HIP_ARCHITECTURES:-}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES}")
fi

# CMake HIP language wants Clang directly (not the hipcc wrapper).
HIP_CLANG=""
for cand in \
  /usr/lib64/rocm/llvm/bin/clang++ \
  "${ROCM_PATH:-/usr}/lib/llvm/bin/clang++" \
  "${ROCM_PATH:-/usr}/llvm/bin/clang++"
do
  if [[ -x "$cand" ]]; then
    HIP_CLANG="$cand"
    break
  fi
done
if [[ -z "$HIP_CLANG" ]] && command -v clang++ >/dev/null 2>&1; then
  HIP_CLANG="$(command -v clang++)"
fi
if [[ -n "$HIP_CLANG" ]]; then
  CMAKE_ARGS+=(-DCMAKE_HIP_COMPILER="$HIP_CLANG")
  echo "HIP compiler: $HIP_CLANG"
else
  echo "warning: no hip-clang found; letting CMake pick a HIP compiler" >&2
fi

cmake -S "$ROOT" -B "$BUILD" "${CMAKE_ARGS[@]}" "$@"
cmake --build "$BUILD" -j"$(nproc 2>/dev/null || echo 4)"

echo "built: $BUILD"
echo "  lib:  $BUILD/libwelvet_rocm.so"
echo "  tool: $BUILD/welvet_dense_fwd"
echo "  ex:   $BUILD/example_rocm_smoke"
echo
echo "smoke:  $BUILD/example_rocm_smoke"
echo "gaps:   $BUILD/welvet_dense_fwd --gaps"
