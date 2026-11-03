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
  --parallel-train    Parallel + ParallelPoly + NormalBPEdges
  --normalbp-all      Full TestGoVRocmNormalBP aggregator
  --stepbp-train      StepBP Line STB1 + Dense MeshBP MBP1 (+ ModeMeshBP vols)
  --meshbp-train      Heterogeneous MeshBP MGB1 (+ StepBP/volume + NormalBP + ROCm-only)
  --tween-train       ModeTween TWB1/TWV1 + per-kind volumes (TestGoVRocmTweenTrain)
  --tweenchain-train  ModeTweenChain TCB1/TCV1 + per-kind volumes (TestGoVRocmTweenChainTrain)
  --tweensplit-train  ModeTweenSplit TSB1/TSP1 + per-kind volumes (TestGoVRocmTweenSplitTrain)
  --steptweensplit-train  ModeStepTweenSplit STS1/STSO + TweenSplit volumes (TestGoVRocmStepTweenSplitTrain)
  --tweenalt-train    ModeTweenAlt TAB1/TAV1 + per-kind volumes (TestGoVRocmTweenAltTrain)
  --steptweenalt-train  ModeStepTweenAlt STA1/STAO + TweenAlt volumes (TestGoVRocmStepTweenAltTrain)
  --meshtween-train   MeshTween MTW1/MGW1 + ModeTween volumes (TestGoVRocmMeshTweenTrain)
  --meshtweenchain-train  MeshTweenChain MTC1/MGC1 + TweenChain volumes (TestGoVRocmMeshTweenChainTrain)
  --steptween-train  ModeStepTween STW1/STWO + Tween volumes (TestGoVRocmStepTweenTrain)
  --steptweenchain-train  ModeStepTweenChain STC1/STCO + TweenChain volumes (TestGoVRocmStepTweenChainTrain)
  --proxy-train       TweenSplit proxy modes GPB1/PGB1 (TestGoVRocmProxyTrain + ProxyVolumeTrain)
  --mesh-split-modes  MeshTweenSplit/Alt/FastProxy/Sparse clock + shared volumes (TestGoVRocmMeshSplitModes)
  --training-25-29-all  StepTweenSplit Linear…Sparse Line proxies (TestGoVRocmStepProxyTrain)
  --cameral-all         Freeze/Shadow/Adversarial/Memory + CamSync (TestGoVRocmCameralTrain|CamSync|…)
  --go-routing          Mixed Go/C++/ROCm JSON routing (requires install + C++ artifact)
  --help
EOF
}

MODE=dense-fwd
if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --smoke) MODE=smoke; shift ;;
    --gaps) MODE=gaps; shift ;;
    --cameral-all) MODE=cameral-all; shift ;;
    --go-routing) MODE=go-routing; shift ;;
    --dense-fwd|--rmsnorm-fwd|--layernorm-fwd|--softmax-fwd|--swiglu-fwd|--embedding-fwd|--mha-fwd|--cnn-fwd|--convt-fwd|--rnn-fwd|--lstm-fwd|--sequential-fwd|--residual-fwd|--mamba-fwd|--metacognition-fwd|--gdn-fwd|--kmeans-fwd|--parallel-fwd|--seqmix-fwd|--leaf-fwd-all|--dense-train|--rmsnorm-train|--layernorm-train|--softmax-train|--swiglu-train|--embedding-train|--mha-train|--cnn-train|--convt-train|--rnn-train|--lstm-train|--sequential-train|--residual-train|--stack-train|--mamba-train|--metacognition-train|--gdn-train|--kmeans-train|--parallel-train|--normalbp-all|--stepbp-train|--meshbp-train|--tween-train|--tweenchain-train|--tweensplit-train|--steptweensplit-train|--tweenalt-train|--steptweenalt-train|--meshtween-train|--meshtweenchain-train|--steptween-train|--steptweenchain-train|--proxy-train|--mesh-split-modes|--training-25-29-all)
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
  if [[ -f "$ROOT/../welvet/native/rocm/linux_x86_64/libwelvet_rocm.so" ]]; then
    export WELVET_ROCM_LIBRARY="$ROOT/../welvet/native/rocm/linux_x86_64/libwelvet_rocm.so"
  elif [[ -f "$BUILD/libwelvet_rocm.so" ]]; then
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
  # Default 1 measured bench iter (match metal); set WELVET_ROCM_BENCH_ITERS=2000 for long perf.
  export WELVET_ROCM_BENCH_ITERS="${WELVET_ROCM_BENCH_ITERS:-1}"
  (
    cd "$W2A"
    env -u CC -u CXX -u CMAKE_CXX_COMPILER -u CMAKE_C_COMPILER \
      PATH="/usr/bin:/usr/local/go/bin:${PATH}" \
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
  parallel-train) run_go_test '^TestGoVRocm(ParallelTrain|ParallelPolyTrain|NormalBPEdges)$' ;;
  normalbp-all) run_go_test '^TestGoVRocmNormalBP$' ;;
  stepbp-train) run_go_test '^TestGoVRocmStepBPTrain$' ;;
  meshbp-train) run_go_test '^(TestGoVRocmMeshBPTrain|TestGoVRocmStepBPTrain|TestGoVRocmNormalBP)$' ;;
  tween-train) run_go_test '^TestGoVRocmTweenTrain$' ;;
  tweenchain-train) run_go_test '^TestGoVRocmTweenChainTrain$' ;;
  tweensplit-train) run_go_test '^TestGoVRocmTweenSplitTrain$' ;;
  steptweensplit-train) run_go_test '^TestGoVRocmStepTweenSplitTrain$' ;;
  tweenalt-train) run_go_test '^TestGoVRocmTweenAltTrain$' ;;
  steptweenalt-train) run_go_test '^TestGoVRocmStepTweenAltTrain$' ;;
  meshtween-train) run_go_test '^(TestGoVRocmMeshTweenTrain|TestGoVRocmMeshBPTrain|TestGoVRocmTweenTrain)$' ;;
  meshtweenchain-train) run_go_test '^(TestGoVRocmMeshTweenChainTrain|TestGoVRocmMeshTweenTrain|TestGoVRocmTweenChainTrain)$' ;;
  steptween-train) run_go_test '^(TestGoVRocmStepTweenTrain|TestGoVRocmTweenTrain)$' ;;
  steptweenchain-train) run_go_test '^(TestGoVRocmStepTweenChainTrain|TestGoVRocmTweenChainTrain)$' ;;
  proxy-train) run_go_test '^(TestGoVRocmProxyTrain|TestGoVRocmProxyVolumeTrain)$' ;;
  mesh-split-modes) run_go_test '^TestGoVRocmMeshSplitModes$' ;;
  training-25-29-all) run_go_test '^TestGoVRocmStepProxyTrain$' ;;
  cameral-all) WELVET_ROCM_TEST_TIMEOUT="${WELVET_ROCM_TEST_TIMEOUT:-180m}" run_go_test '^(TestGoVRocmCameralTrain|TestGoVRocmCamSync|TestGoStoreReplaceInvalidatesWires|TestGoVRocmOnly)$' ;;
  go-routing)
    "$ROOT/scripts/linux_install_go.sh"
    export WELVET_REQUIRE_CPP_ROUTING=1
    export WELVET_ROCM_BENCH_ITERS="${WELVET_ROCM_BENCH_ITERS:-1}"
    export WELVET_ROCM_LIBRARY="${WELVET_ROCM_LIBRARY:-$ROOT/../welvet/native/rocm/linux_x86_64/libwelvet_rocm.so}"
    run_go_test '^(TestGoNativeRocm.*|TestGoVRocmOnly)$'
    ;;
esac
