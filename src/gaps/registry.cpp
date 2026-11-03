#include "welvet/gaps.hpp"

#include <iostream>

namespace welvet::gaps {
namespace {

const std::vector<Entry>& build() {
  static const std::vector<Entry> k = {
      // --- M0 ROCm host (this scaffold) ---
      {"rocm.device.init", "rocm", "hipDevice probe + runtime identity", true},
      {"rocm.kernels.compile", "rocm", "hipcc compile src/rocm/kernels.hip (smoke + matvec stub)", true},
      {"rocm.buffer.device", "rocm", "hipMalloc alloc/upload/download", true},
      {"rocm.kernel.smoke", "rocm", "Launch welvet_smoke_saxpy", true},
      {"abi.version", "interop", "welvet_version + welvet_rocm_init C ABI", true},
      {"platform.linux", "platform", "Linux x86_64 ROCm host build", true},

      // --- Core / weights (port from welvet.cpp / welvet.metal semantics) ---
      {"core.dtype.all34", "core", "all 34 DType names + FormatNone Dense forward gates", true},
      {"core.activate", "core", "Activate/ActivateDeriv host helpers (GPU act in Dense kernel)", true},
      {"core.fp_codecs", "core", "FP8/FP4/NF4/FP6 pack-unpack for FormatNone storage", true},
      {"weights.formatnone.x34", "weights", "FormatNone pack/decode ×34 for Dense forward", true},
      {"weights.matvec.rocm", "weights", "resident f64 Dense MatVec + bias/act on ROCm", true},
      {"weights.resident", "weights", "upload-once W/bias; warm execute reuses device buffers", true},

      // --- Layers (same 23 kinds as welvet.cpp / Go / welvet.metal) ---
      {"dense.forward.rocm", "layers", "Dense ROCm FormatNone×34 forward (exact_bit vs Go)", true},
      {"dense.normalbp.rocm", "layers", "Dense NormalBP dX/dW + FormatNone SGD ROCm (leaf+vol)",
       true},
      {"mha.forward.rocm", "layers", "MHA ModeSelf FormatNone×34 ROCm forward (exact_bit vs Go)", true},
      {"mha.normalbp.rocm", "layers", "MHA NormalBP ModeSelf + RoPEᵀ ROCm (leaf+vol)", true},
      {"swiglu.forward.rocm", "layers", "SwiGLU FormatNone×34 ROCm forward (exact_bit vs Go)", true},
      {"swiglu.normalbp.rocm", "layers", "SwiGLU NormalBP projs+SiLU′ ROCm (leaf+vol)", true},
      {"rmsnorm.rocm", "layers", "RMSNorm FormatNone×34 ROCm forward (exact_bit vs Go)", true},
      {"rmsnorm.normalbp.rocm", "layers", "RMSNorm NormalBP dX/dγ + SGD ROCm (leaf+vol)", true},
      {"layernorm.rocm", "layers", "LayerNorm FormatNone×34 ROCm forward (exact_bit vs Go)", true},
      {"layernorm.normalbp.rocm", "layers", "LayerNorm NormalBP dX/dγβ + SGD ROCm (leaf+vol)", true},
      {"softmax.rocm", "layers", "Softmax all Kind FormatNone×34 ROCm forward (exact_bit vs Go)", true},
      {"softmax.normalbp.rocm", "layers", "Softmax NormalBP dX-only ROCm (leaf+vol, weightless)",
       true},
      {"embedding.forward.rocm", "layers", "Embedding FormatNone×34 ROCm gather (exact_bit vs Go)", true},
      {"embedding.normalbp.rocm", "layers", "Embedding scatter NormalBP ROCm (leaf+vol)", true},
      {"cnn.rocm", "layers", "CNN1/2/3 FormatNone×34 ROCm forward (im2col→Dense exact_bit)", true},
      {"cnn.normalbp.rocm", "layers", "CNN1/2/3 im2col→Dense NormalBP ROCm (leaf+vol)", true},
      {"convt.rocm", "layers", "ConvT1/2/3 FormatNone×34 ROCm forward (f64 scatter exact_bit)", true},
      {"convt.normalbp.rocm", "layers", "ConvT1/2/3 scatter adjoint NormalBP ROCm (leaf+vol)", true},
      {"rnn.rocm", "layers", "RNN FormatNone×34 ROCm forward (Dense IH/HH exact_bit)", true},
      {"rnn.normalbp.rocm", "layers", "RNN BPTT Dense IH/HH + tanh′ + SGD FormatNone×34; RNB1/RNV1",
       true},
      {"lstm.rocm", "layers", "LSTM FormatNone×34 ROCm forward (4-gate Dense exact_bit)", true},
      {"lstm.normalbp.rocm", "layers", "LSTM BPTT 4-gate Dense + SGD FormatNone×34; LSB1/LSV1",
       true},
      {"sequential.rocm", "layers", "Sequential Dense chain FormatNone×34 ROCm forward (exact_bit)",
       true},
      {"sequential.normalbp.rocm", "layers",
       "Sequential NormalBP reverse Dense chain + ApplyGradSGD FormatNone×34; SQB1/SQV1", true},
      {"residual.rocm", "layers", "Residual F(x)+x FormatNone×34 ROCm forward (exact_bit)", true},
      {"residual.normalbp.rocm", "layers",
       "Residual NormalBP F reverse Dense + skip grad + ApplyGradSGD FormatNone×34; REB1/REV1",
       true},
      {"kmeans.rocm", "layers", "KMeans soft assignment FormatNone×34 ROCm (exact_bit)", true},
      {"kmeans.normalbp.rocm", "layers",
       "KMeans NormalBP soft-assignment + SGD FormatNone×34; KMB1/KMV1", true},
      {"mamba.rocm", "layers", "Mamba Dense+scan FormatNone×34 ROCm forward (exact_bit)", true},
      {"mamba.normalbp.rocm", "layers",
       "Mamba NormalBP scan BPTT + In/Out Dense FormatNone×34; MBB1/MBV1", true},
      {"metacognition.rocm", "layers", "Metacognition Observed+rules FormatNone×34 ROCm (exact_bit)",
       true},
      {"metacognition.normalbp.rocm", "layers",
       "Metacognition NormalBP lastGate + Observed Dense FormatNone×34; MCB1/MCV1", true},
      {"gdn.rocm", "layers", "GDN Float32-only f32 forward (exact_bit vs Go)", true},
      {"gdn.normalbp.rocm", "layers",
       "GDN truncated-BPTT NormalBP f32-only; GDB1/GDV1", true},
      {"stack.rocm", "layers", "Stack NestedStack-of-Dense ROCm (NormalBP STKB/STKV)", true},
      {"stack.normalbp.rocm", "layers",
       "C ABI welvet_stack_normalbp_f32 + STKB/STKV tool", true},
      {"parallel.rocm", "layers", "Parallel Dense branches + combine modes ROCm (exact_bit)", true},
      {"parallel.normalbp.rocm", "layers",
       "Parallel NormalBP Dense/poly + edges + volume; PNB1/PNV1/PPB1/NBL1", true},
      {"seqmix.rocm", "layers", "seqmix Kind name smoke WSX1 (metadata, no tensors)", true},

      // --- Runtime / architecture / training ---
      {"arch.grid", "architecture",
       "Grid topology + binds Dense…Stack/Mamba/Meta/GDN/KMeans/Parallel", true},
      {"runtime.forward", "runtime",
       "volume forward Dense…Stack/Mamba/Meta/GDN/KMeans/Parallel ROCm", true},
      {"runtime.backward", "runtime",
       "volume backward Dense…Stack/Mamba/Meta/GDN/KMeans/Parallel ROCm", true},
      {"runtime.step", "runtime", "Line StepBP FamilyBP (STB1) + runtime/line", true},
      {"runtime.mesh", "runtime",
       "Heterogeneous MeshBP FamilyBP (MGB1) + Dense MBP1 + runtime/step", true},
      {"train.modes.x29", "training", "gap? all 29 TrainModes + Freeze/Shadow/Adv/Memory", false},
      {"train.advanced", "training",
       "ROCm Freeze/Shadow/Adversarial/Memory, mixed branches and Stack overrides (CMB1)", true},
      {"train.camsync", "training",
       "ROCm CamSync/BlendStores, native repacking, groups/cross links (CMB1)", true},
      {"abi.model_session", "abi",
       "welvet_model_* Mount/Forward/TrainMSE/Export + welvet_rocm_blob_free", true},

      // --- Systems / model ---
      {"systems.tween", "systems", "ModeTween projected credit ROCm", true},
      {"systems.tween_chain", "systems",
       "ModeTweenChain FamilyBP @ full lr; TCB1/TCBO + TCV1/TCVO + TRC1…MBTC", true},
      {"runtime.tween_chain_train", "runtime",
       "TrainStackMSE ModeTweenChain (FamilyBP @ full lr)", true},
      {"systems.tween_split", "systems",
       "ModeTweenSplit 1/N projected gap @ full lr; TSB1/TSP1 + TRP1…", true},
      {"runtime.tween_split_train", "runtime",
       "TrainStackMSE ModeTweenSplit (OpenSplitTape 1/N)", true},
      {"runtime.line.family_tween", "runtime",
       "Line FamilyTween on pop (STW1/STWO; apply_projected @ lr×0.5)", true},
      {"runtime.line.family_tween_chain", "runtime",
       "Line StepTweenChain FamilyBP alias (STC1/STCO ≡ StepBP on pop @ full lr)", true},
      {"systems.tween_split_head_proxy", "systems",
       "ModeTweenSplitHeadProxy head Backward+SGD @ full lr; hidden 1/(N−1) GradWOnly; GPB1/PGB1 variant 1",
       true},
      {"systems.tween_split_linear", "systems",
       "ModeTweenSplitLinear reverse walk 1/N + LinearGradIn; GPB1/PGB1 variant 2", true},
      {"systems.tween_split_fast_proxy", "systems",
       "ModeTweenSplitFastProxy linear head proxy then hidden 1/(N−1) GradWOnly; GPB1/PGB1 variant 3",
       true},
      {"systems.tween_split_linear_cache", "systems",
       "ModeTweenSplitLinearCache refresh every 20; rescale cached P(gy); GPB1/PGB1 variant 4",
       true},
      {"systems.tween_split_head_proxy_async", "systems",
       "ModeTweenSplitHeadProxyAsync stale Wᵀgy hidden then head GradWOnly; store post-update proxy; GPB1/PGB1 variant 5",
       true},
      {"systems.tween_split_sparse", "systems",
       "ModeTweenSplitSparse pre-update head proxy; rotate one hidden leaf; GPB1/PGB1 variant 6",
       true},
      {"runtime.line.family_tween_split", "runtime",
       "Line FamilyTweenSplit on pop (STS1/STSO; apply_split_child 1/N @ full lr)", true},
      {"systems.tween_alt", "systems",
       "ModeTweenAlt Split@lr → Tween@lr×0.5; TAB1/TAV1 + TRA1…KMAL/PNAL", true},
      {"runtime.tween_alt_train", "runtime",
       "TrainStackMSE ModeTweenAlt (AltTimes × Split then Tween soft lr)", true},
      {"runtime.line.family_tween_alt", "runtime",
       "Line FamilyTweenAlt on pop (STA1/STAO; apply_alt_child Split→Tween frozen gap)", true},
      {"runtime.step.family_tween", "runtime",
       "MeshTween last-tick local credit (MTW1/MTWO, MGW1/MGWO; apply_cell_projected @ lr×0.5)",
       true},
      {"runtime.step.meshtweenchain", "runtime",
       "MeshTweenChain ≡ MeshBP FamilyBP alias (MTC1/MTCO, MGC1/MGCO)", true},
      {"runtime.step.meshtweensplit", "runtime",
       "MeshTweenSplit last-tick FamilyTweenSplit 1/N @ full lr (MGS1/MGSO)", true},
      {"runtime.step.meshtweenalt", "runtime",
       "MeshTweenAlt last-tick Split→Tween@lr×0.5 (MGA1/MGAO)", true},
      {"runtime.step.meshtweensplit_fast_proxy", "runtime",
       "MeshTweenSplitFastProxy last-tick FastProxy (MGF1/MGFO)", true},
      {"runtime.step.meshtweensplit_sparse", "runtime",
       "MeshTweenSplitSparse last-tick Sparse rotate (MGQ1/MGQO)", true},
      {"runtime.line.step_tween_split_head_proxy", "runtime",
       "Line StepTweenSplitHeadProxy on pop (STH1/STHO; FamilyTweenSplitHeadProxy)", true},
      {"runtime.line.step_tween_split_linear", "runtime",
       "Line StepTweenSplitLinear on pop (STL1/STLO; apply_variant_child 2)", true},
      {"runtime.line.step_tween_split_fast_proxy", "runtime",
       "Line StepTweenSplitFastProxy on pop (STF1/STFO; apply_variant_child 3)", true},
      {"runtime.line.step_tween_split_linear_cache", "runtime",
       "Line StepTweenSplitLinearCache on pop (SLC1/SLCO; apply_variant_child 4)", true},
      {"runtime.line.step_tween_split_head_proxy_async", "runtime",
       "Line StepTweenSplitHeadProxyAsync on pop (SHA1/SHAO; apply_variant_child 5)", true},
      {"runtime.line.step_tween_split_sparse", "runtime",
       "Line StepTweenSplitSparse on pop (SSP1/SSPO; apply_variant_child 6)", true},
      {"systems.dna", "systems", "gap? DNA", false},
      {"systems.evolution", "systems", "gap? evolution", false},
      {"systems.tanhi", "systems", "gap? TANHI", false},
      {"systems.telemetry", "systems", "gap? telemetry", false},
      {"model.tokenizer", "model", "gap? tokenizer", false},
      {"model.entity", "model", "gap? ENTITY", false},
      {"model.transformer", "model", "gap? transformer", false},
      {"model.sampling", "model", "gap? sampling", false},
      {"model.hf", "model", "gap? HF load", false},
      {"lucy.lpd", "lucy", "gap? Lucy/LPD", false},

      // --- Delivery / validation ---
      {"platform.windows", "platform", "Windows amd64 ROCm host build (libwelvet_rocm.dll)", true},
      {"w2a.go_v_rocm", "interop", "w2a Go_v_Rocm Dense forward parity + speed", true},
      {"nac.go_delegate", "interop", "gap? Go Welvet NAC → ROCm dispatch", false},
      {"determinism.policy", "rocm", "gap? documented reduction order + exact_bit bands", false},
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
  os << "welvet.rocm gaps: done=" << count_done() << " open=" << count_gaps() << "\n";
  for (const auto& e : all()) {
    os << (e.implemented ? "  [done] " : "  [gap?] ") << e.id << " — " << e.detail << "\n";
  }
}

std::string mark(std::string_view what) {
  return std::string("gap? ") + std::string(what);
}

} // namespace welvet::gaps
