#include "welvet/gaps.hpp"

#include <iostream>

namespace welvet::gaps {
namespace {

const std::vector<Entry>& build() {
  static const std::vector<Entry> k = {
      // --- M0 ROCm / HIP host (this scaffold) ---
      {"rocm.device.init", "rocm", "hipDevice probe + runtime identity", true},
      {"rocm.kernels.compile", "rocm", "hipcc compile smoke + matvec stub", true},
      {"rocm.buffer.device", "rocm", "hipMalloc alloc/upload/download", true},
      {"rocm.kernel.smoke", "rocm", "Launch welvet_smoke_saxpy", true},
      {"abi.version", "interop", "welvet_version + welvet_rocm_init C ABI", true},
      {"platform.linux", "platform", "Linux x86_64 ROCm/HIP host build", true},

      // --- Core / weights (port from welvet.cuda / welvet.cpp semantics) ---
      {"core.dtype.all34", "core", "all 34 DType names + FormatNone Dense forward gates", false},
      {"core.activate", "core", "Activate/ActivateDeriv host helpers (GPU act in Dense kernel)", false},
      {"core.fp_codecs", "core", "FP8/FP4/NF4/FP6 pack-unpack for FormatNone storage", false},
      {"weights.formatnone.x34", "weights", "FormatNone pack/decode ×34 for Dense forward", false},
      {"weights.matvec.rocm", "weights", "resident f64 Dense MatVec + bias/act on ROCm", false},
      {"weights.resident", "weights", "upload-once W/bias; warm execute reuses device buffers", false},

      // --- Layers (same 23 kinds — climb TRAINING_CHECKLIST toward cuda 0.79) ---
      {"dense.forward.rocm", "layers", "Dense ROCm FormatNone×34 forward (exact_bit vs Go)", false},
      {"dense.normalbp.rocm", "layers", "Dense NormalBP dX/dW + FormatNone SGD ROCm (leaf+vol)",
       false},
      {"mha.forward.rocm", "layers", "MHA ModeSelf FormatNone×34 ROCm forward", false},
      {"mha.normalbp.rocm", "layers", "MHA NormalBP ModeSelf + RoPEᵀ ROCm", false},
      {"swiglu.forward.rocm", "layers", "SwiGLU FormatNone×34 ROCm forward", false},
      {"swiglu.normalbp.rocm", "layers", "SwiGLU NormalBP ROCm", false},
      {"rmsnorm.rocm", "layers", "RMSNorm FormatNone×34 ROCm forward", false},
      {"rmsnorm.normalbp.rocm", "layers", "RMSNorm NormalBP ROCm", false},
      {"layernorm.rocm", "layers", "LayerNorm FormatNone×34 ROCm forward", false},
      {"layernorm.normalbp.rocm", "layers", "LayerNorm NormalBP ROCm", false},
      {"softmax.rocm", "layers", "Softmax Kind FormatNone×34 ROCm forward", false},
      {"softmax.normalbp.rocm", "layers", "Softmax NormalBP dX-only ROCm", false},
      {"embedding.forward.rocm", "layers", "Embedding FormatNone×34 ROCm gather", false},
      {"embedding.normalbp.rocm", "layers", "Embedding scatter NormalBP ROCm", false},
      {"cnn.rocm", "layers", "CNN1/2/3 FormatNone×34 ROCm forward", false},
      {"cnn.normalbp.rocm", "layers", "CNN1/2/3 NormalBP ROCm", false},
      {"convt.rocm", "layers", "ConvT1/2/3 FormatNone×34 ROCm forward", false},
      {"convt.normalbp.rocm", "layers", "ConvT1/2/3 NormalBP ROCm", false},
      {"rnn.rocm", "layers", "RNN FormatNone×34 ROCm forward", false},
      {"rnn.normalbp.rocm", "layers", "RNN BPTT ROCm", false},
      {"lstm.rocm", "layers", "LSTM FormatNone×34 ROCm forward", false},
      {"lstm.normalbp.rocm", "layers", "LSTM BPTT ROCm", false},
      {"sequential.rocm", "layers", "Sequential Dense chain FormatNone×34 ROCm", false},
      {"sequential.normalbp.rocm", "layers", "Sequential NormalBP ROCm", false},
      {"residual.rocm", "layers", "Residual F(x)+x FormatNone×34 ROCm", false},
      {"residual.normalbp.rocm", "layers", "Residual NormalBP ROCm", false},
      {"kmeans.rocm", "layers", "KMeans soft assignment FormatNone×34 ROCm", false},
      {"kmeans.normalbp.rocm", "layers", "KMeans NormalBP ROCm", false},
      {"mamba.rocm", "layers", "Mamba Dense+scan FormatNone×34 ROCm", false},
      {"mamba.normalbp.rocm", "layers", "Mamba NormalBP ROCm", false},
      {"metacognition.rocm", "layers", "Metacognition FormatNone×34 ROCm", false},
      {"metacognition.normalbp.rocm", "layers", "Metacognition NormalBP ROCm", false},
      {"gdn.rocm", "layers", "GDN Float32-only ROCm forward", false},
      {"gdn.normalbp.rocm", "layers", "GDN truncated-BPTT NormalBP f32 ROCm", false},
      {"stack.rocm", "layers", "Stack NestedStack-of-Dense ROCm", false},
      {"stack.normalbp.rocm", "layers", "Stack NormalBP ROCm", false},
      {"parallel.rocm", "layers", "Parallel Dense branches + combine modes ROCm", false},
      {"parallel.normalbp.rocm", "layers", "Parallel NormalBP ROCm", false},
      {"seqmix.rocm", "layers", "seqmix Kind name smoke", false},

      // --- Train modes 1–29 + cameral (same ladder as welvet.cuda) ---
      {"train.normalbp", "runtime", "ModeNormalBP all 23 families — TestGoVRocmNormalBP", false},
      {"train.stepbp", "runtime", "StepBP Line + Dense MeshBP", false},
      {"train.meshbp", "runtime", "Heterogeneous MeshBP", false},
      {"train.tween", "runtime", "ModeTween…MeshTweenSplitSparse + Step proxy ladder", false},
      {"train.cameral", "runtime", "Freeze/Shadow/Adv/Memory/CamSync", false},

      // --- Systems / model / peak ---
      {"systems.tween", "systems", "Tween* systems on ROCm", false},
      {"systems.camsync", "systems", "CamSync BlendStores on ROCm", false},
      {"model.io", "model", "Model mount/forward/train C ABI", false},
      {"peak.fused", "perf", "Resident graph + fused training", false},
  };
  return k;
}

} // namespace

const std::vector<Entry>& all() { return build(); }

std::size_t count_gaps() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (!e.implemented) ++n;
  }
  return n;
}

std::size_t count_done() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (e.implemented) ++n;
  }
  return n;
}

void print_census(std::ostream& os) {
  const auto& entries = all();
  os << "welvet.rocm gaps: done=" << count_done() << " open=" << count_gaps()
     << " total=" << entries.size() << "\n";
  for (const auto& e : entries) {
    os << (e.implemented ? "[x] " : "[ ] ") << e.id << "  (" << e.area << ")  " << e.detail << "\n";
  }
}

std::string mark(std::string_view what) {
  return std::string("gap? ") + std::string(what);
}

} // namespace welvet::gaps
