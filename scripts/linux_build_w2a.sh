#!/usr/bin/env bash
# Build welvet.rocm then run sibling w2a go_v_rocm gates.
# Determinism / exact_bit evidence lives in w2a — same pattern as go_v_cuda.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W2A="${W2A_DIR:-$ROOT/../w2a}"

"$ROOT/scripts/build_linux.sh"

if [[ ! -d "$W2A" ]]; then
  echo "w2a not found at $W2A (set W2A_DIR). Engine build succeeded; skip tests." >&2
  exit 0
fi

export WELVET_ROCM_LIB="${WELVET_ROCM_LIB:-$ROOT/build/libwelvet_rocm.so}"
export WELVET_DENSE_FWD="${WELVET_DENSE_FWD:-$ROOT/build/welvet_dense_fwd}"

MODE="${1:---smoke}"
cd "$W2A"

case "$MODE" in
  --smoke)
    echo "== welvet.rocm smoke =="
    "$ROOT/build/example_rocm_smoke"
    "$ROOT/build/welvet_dense_fwd" --version
    "$ROOT/build/welvet_dense_fwd" --identity || true
    "$ROOT/build/welvet_dense_fwd" --gaps
    if [[ -d "$W2A/tests/go_v_rocm" ]]; then
      go test ./tests/go_v_rocm/ -count=1 -run 'Smoke|Available|Gaps' -v
    else
      echo "note: w2a/tests/go_v_rocm not present yet — engine smoke only"
    fi
    ;;
  --leaf-fwd-all|--dense-train|--normalbp-all|--training-all|--help|-h)
    if [[ ! -d "$W2A/tests/go_v_rocm" ]]; then
      echo "go_v_rocm suites not landed yet in w2a. Scaffold engine is M0 (0.00)." >&2
      echo "Add suites under w2a/suites/go_v_rocm + tests/go_v_rocm as gates tick" >&2
      echo "TRAINING_CHECKLIST.md (mirror welvet.cuda → 0.79)." >&2
      exit 1
    fi
    echo "run: go test ./tests/go_v_rocm/ -v  (mode $MODE — wire flags as suites grow)"
    go test ./tests/go_v_rocm/ -count=1 -v
    ;;
  *)
    echo "usage: $0 [--smoke|--leaf-fwd-all|--dense-train|--normalbp-all|--training-all]" >&2
    exit 2
    ;;
esac
