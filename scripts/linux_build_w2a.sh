#!/usr/bin/env bash
# Build welvet.rocm then run sibling w2a Go↔ROCm suites.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W2A="${W2A_DIR:-$ROOT/../w2a}"
BUILD="${BUILD_DIR:-$ROOT/build}"

usage() {
  cat <<EOF
usage: $0 [flags]

  (default)           build + TestGoVRocmDenseForward
  --smoke             build + example_rocm_smoke
  --gaps              build + welvet_dense_fwd --gaps
  --dense-fwd         Dense FormatNone×34 forward
  --rmsnorm-fwd       RMSNorm FormatNone×34 forward
  --layernorm-fwd     LayerNorm FormatNone×34 forward
  --softmax-fwd       Softmax forward
  --swiglu-fwd        SwiGLU forward
  --embedding-fwd     Embedding forward
  --mha-fwd           MHA forward
  --cnn-fwd           CNN1/2/3 forward
  --convt-fwd         ConvT1/2/3 forward
  --rnn-fwd           RNN forward
  --lstm-fwd          LSTM forward
  --sequential-fwd    Sequential forward
  --residual-fwd      Residual forward
  --mamba-fwd         Mamba forward
  --metacognition-fwd Metacognition forward
  --gdn-fwd           GDN forward
  --kmeans-fwd        KMeans forward
  --parallel-fwd      Parallel forward
  --seqmix-fwd        seqmix smoke
  --leaf-fwd-all      all leaf FormatNone forward suites
  --dense-train       Dense NormalBP leaf + volume (7 cases)
  --rmsnorm-train     RMSNorm NormalBP leaf + volume (7 cases)
  --layernorm-train   LayerNorm NormalBP leaf + volume (7 cases)
  --softmax-train     Softmax NormalBP leaf + volume (9 cases)
  --swiglu-train      SwiGLU NormalBP leaf + volume (7 cases)
  --embedding-train   Embedding NormalBP leaf + volume (7 cases)
  --mha-train         MHA NormalBP leaf + volume (8 cases)
  --cnn-train         CNN1/2/3 NormalBP leaf + volume
  --convt-train       ConvT1/2/3 NormalBP leaf + volume
  --rnn-train         RNN NormalBP leaf + volume
  --lstm-train        LSTM NormalBP leaf + volume
  --sequential-train  Sequential NormalBP leaf + volume
  --residual-train    Residual NormalBP leaf + volume
  --stack-train       Stack NestedStack-of-Dense NormalBP leaf + volume
  --mamba-train       Mamba NormalBP leaf + volume
  --metacognition-train Metacognition NormalBP leaf + volume
  --gdn-train         GDN NormalBP leaf + volume (f32)
  --kmeans-train      KMeans NormalBP leaf + volume
  --help
EOF
}

MODE=dense-fwd
if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --smoke) MODE=smoke; shift ;;
    --gaps) MODE=gaps; shift ;;
    --dense-fwd|--rmsnorm-fwd|--layernorm-fwd|--softmax-fwd|--swiglu-fwd|--embedding-fwd|--mha-fwd|--cnn-fwd|--convt-fwd|--rnn-fwd|--lstm-fwd|--sequential-fwd|--residual-fwd|--mamba-fwd|--metacognition-fwd|--gdn-fwd|--kmeans-fwd|--parallel-fwd|--seqmix-fwd|--leaf-fwd-all|--dense-train|--rmsnorm-train|--layernorm-train|--softmax-train|--swiglu-train|--embedding-train|--mha-train|--cnn-train|--convt-train|--rnn-train|--lstm-train|--sequential-train|--residual-train|--stack-train|--mamba-train|--metacognition-train|--gdn-train|--kmeans-train)
      MODE="${1#--}"; shift ;;
    *)
      echo "unknown flag '$1'" >&2
      usage
      exit 2
      ;;
  esac
fi

"$ROOT/scripts/build_linux.sh"

export WELVET_ROCM_DENSE_FWD="$BUILD/welvet_dense_fwd"
if [[ -z "${WELVET_ROCM_LIBRARY:-}" ]]; then
  if [[ -f "$BUILD/libwelvet_rocm.so" ]]; then
    export WELVET_ROCM_LIBRARY="$BUILD/libwelvet_rocm.so"
  fi
fi
if [[ -z "${WELVET_DENSE_FWD:-}" ]]; then
  for cand in "$ROOT/../welvet.cpp/build/welvet_dense_fwd" "$ROOT/../welvet.cpp/build/Release/welvet_dense_fwd"; do
    if [[ -x "$cand" ]]; then
      export WELVET_DENSE_FWD="$cand"
      break
    fi
  done
fi

run_go_test() {
  local pattern="$1"
  if [[ ! -d "$W2A" ]]; then
    echo "w2a sibling not found at $W2A" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="${BUILD}:${LD_LIBRARY_PATH:-}"
  export WELVET_ROCM_BENCH_ITERS="${WELVET_ROCM_BENCH_ITERS:-1}"
  # Keep the caller's `go` first — putting /usr/bin ahead breaks mismatched toolchains.
  local go_bin
  go_bin="$(command -v go)"
  local go_dir
  go_dir="$(dirname "$go_bin")"
  (
    cd "$W2A"
    env -u CC -u CXX -u CMAKE_CXX_COMPILER -u CMAKE_C_COMPILER \
      PATH="${go_dir}:/usr/local/go/bin:/usr/bin:${PATH}" \
      WELVET_ROCM_DENSE_FWD="$WELVET_ROCM_DENSE_FWD" \
      WELVET_ROCM_LIBRARY="${WELVET_ROCM_LIBRARY:-}" \
      WELVET_DENSE_FWD="${WELVET_DENSE_FWD:-}" \
      WELVET_ROCM_BENCH_ITERS="$WELVET_ROCM_BENCH_ITERS" \
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      go test ./tests/go_v_rocm -v -count=1 -timeout "${WELVET_ROCM_TEST_TIMEOUT:-90m}" -run "$pattern"
  )
}

case "$MODE" in
  smoke) "$BUILD/example_rocm_smoke" ;;
  gaps) "$BUILD/welvet_dense_fwd" --gaps ;;
  dense-fwd) run_go_test '^TestGoVRocmDenseForward$' ;;
  rmsnorm-fwd) run_go_test '^TestGoVRocmRMSNorm$' ;;
  layernorm-fwd) run_go_test '^TestGoVRocmLayerNorm$' ;;
  softmax-fwd) run_go_test '^TestGoVRocmSoftmax$' ;;
  swiglu-fwd) run_go_test '^TestGoVRocmSwiGLU$' ;;
  embedding-fwd) run_go_test '^TestGoVRocmEmbedding$' ;;
  mha-fwd) run_go_test '^TestGoVRocmMHA$' ;;
  cnn-fwd) run_go_test '^TestGoVRocmCNN$' ;;
  convt-fwd) run_go_test '^TestGoVRocmConvT$' ;;
  rnn-fwd) run_go_test '^TestGoVRocmRNN$' ;;
  lstm-fwd) run_go_test '^TestGoVRocmLSTM$' ;;
  sequential-fwd) run_go_test '^TestGoVRocmSequential$' ;;
  residual-fwd) run_go_test '^TestGoVRocmResidual$' ;;
  mamba-fwd) run_go_test '^TestGoVRocmMamba$' ;;
  metacognition-fwd) run_go_test '^TestGoVRocmMetacognition$' ;;
  gdn-fwd) run_go_test '^TestGoVRocmGDN$' ;;
  kmeans-fwd) run_go_test '^TestGoVRocmKMeans$' ;;
  parallel-fwd) run_go_test '^TestGoVRocmParallel$' ;;
  seqmix-fwd) run_go_test '^TestGoVRocmSeqmix$' ;;
  leaf-fwd-all) run_go_test '^TestGoVRocm(DenseForward|RMSNorm|LayerNorm|Softmax|SwiGLU|Embedding|MHA|CNN|ConvT|RNN|LSTM|Sequential|Residual|Mamba|Metacognition|GDN|KMeans|Parallel|Seqmix)$' ;;
  dense-train) run_go_test '^TestGoVRocmDenseTrain$' ;;
  rmsnorm-train) run_go_test '^TestGoVRocmRMSNormTrain$' ;;
  layernorm-train) run_go_test '^TestGoVRocmLayerNormTrain$' ;;
  softmax-train) run_go_test '^TestGoVRocmSoftmaxTrain$' ;;
  swiglu-train) run_go_test '^TestGoVRocmSwiGLUTrain$' ;;
  embedding-train) run_go_test '^TestGoVRocmEmbeddingTrain$' ;;
  mha-train) run_go_test '^TestGoVRocmMHATrain$' ;;
  cnn-train) run_go_test '^TestGoVRocmCNNTrain$' ;;
  convt-train) run_go_test '^TestGoVRocmConvTTrain$' ;;
  rnn-train) run_go_test '^TestGoVRocmRNNTrain$' ;;
  lstm-train) run_go_test '^TestGoVRocmLSTMTrain$' ;;
  sequential-train) run_go_test '^TestGoVRocmSequentialTrain$' ;;
  residual-train) run_go_test '^TestGoVRocmResidualTrain$' ;;
  stack-train) run_go_test '^TestGoVRocmStackTrain$' ;;
  mamba-train) run_go_test '^TestGoVRocmMambaTrain$' ;;
  metacognition-train) run_go_test '^TestGoVRocmMetacognitionTrain$' ;;
  gdn-train) run_go_test '^TestGoVRocmGDNTrain$' ;;
  kmeans-train) run_go_test '^TestGoVRocmKMeansTrain$' ;;
  *)
    echo "unhandled mode: $MODE" >&2
    exit 2
    ;;
esac
