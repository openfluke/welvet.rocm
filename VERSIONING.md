# welvet.rocm versioning

**One scoreboard:** same ten categories / 100 points as Go Welvet.  
**Formula:** `version = 0.{round(earned)}` until **100 → 1.0.0**.

| Tag | Meaning |
|-----|---------|
| **0.00** | Bedrock — ROCm/HIP host scaffold; **0 / 100** on the engine board |
| **0.13** | M1 Dense FormatNone×34 forward signed (Foundation 9 + Dense fwd 4) |
| **0.27** | Leaf FormatNone forwards ×22 layers + seqmix (Stack→NormalBP) |
| **0.51** | NormalBP Mode 1 signed for all 23 layers (`TestGoVRocmNormalBP`) |
| **0.52** | StepBP Line + Dense MeshBP (`TestGoVRocmStepBPTrain`) |
| **0.53** | Heterogeneous MeshBP MGB1 all 23 families (`TestGoVRocmMeshBPTrain`, 103 groups) |
| **0.54** | ModeTween projected local @ lr×0.5 (`TestGoVRocmTweenTrain`, 130 groups) |
| **0.55** | ModeTweenChain FamilyBP @ full lr (`TestGoVRocmTweenChainTrain`, 130 groups) |
| **0.56** | ModeTweenSplit OpenSplitTape 1/N @ full lr (`TestGoVRocmTweenSplitTrain`, 131 groups) |
| **0.57** | ModeStepTweenSplit Line FamilyTweenSplit (`TestGoVRocmStepTweenSplitTrain`, 130 groups) |
| **0.58** | ModeTweenAlt Split→Tween (`TestGoVRocmTweenAltTrain`, 130 groups) |
| **0.59** | ModeStepTweenAlt Line FamilyTweenAlt (`TestGoVRocmStepTweenAltTrain`, 130 groups) |
| **0.60** | MeshTween mesh-clock FamilyTween (`TestGoVRocmMeshTweenTrain`, 159 groups) |
| **0.61** | MeshTweenChain ≡ MeshBP FamilyBP (`TestGoVRocmMeshTweenChainTrain`, 160 groups) |
| **0.62** | ModeStepTween Line FamilyTween (`TestGoVRocmStepTweenTrain`, 130 groups) |
| **0.63** | ModeStepTweenChain Line FamilyBP alias (`TestGoVRocmStepTweenChainTrain`, 130 groups) |
| **0.64** | ModeTweenSplitHeadProxy (`TestGoVRocmProxyTrain` 65 + `TestGoVRocmProxyVolumeTrain` 69) |
| **0.65** | ModeTweenSplitLinear (`TestGoVRocmProxyTrain` 65 + `TestGoVRocmProxyVolumeTrain` 69) |
| **0.66** | ModeTweenSplitFastProxy (`TestGoVRocmProxyTrain` 65 + `TestGoVRocmProxyVolumeTrain` 69) |
| **0.67** | ModeTweenSplitLinearCache (`TestGoVRocmProxyTrain` 88 + `TestGoVRocmProxyVolumeTrain` 92) |
| **0.68** | ModeTweenSplitHeadProxyAsync (`TestGoVRocmProxyTrain` 65 + `TestGoVRocmProxyVolumeTrain` 69) |
| **0.69** | ModeTweenSplitSparse (`TestGoVRocmProxyTrain` 65 + `TestGoVRocmProxyVolumeTrain` 69) |
| **0.70** | MeshTweenSplit (`TestGoVRocmMeshSplitModes` 154; MGS1/MGSO FamilyTweenSplit @ full lr) |
| **0.71** | MeshTweenAlt (`TestGoVRocmMeshSplitModes` 154; MGA1/MGAO Split→Tween@lr×0.5) |
| **0.72** | MeshTweenSplitFastProxy (`TestGoVRocmMeshSplitModes` 172; MGF1/MGFO) |
| **0.73** | MeshTweenSplitSparse (`TestGoVRocmMeshSplitModes` 172; MGQ1/MGQO) |
| **0.74** | StepTweenSplitHeadProxy (`TestGoVRocmProxyTrain/StepTweenSplitHeadProxy` 65; STH1/STHO) |
| **0.75** | StepTweenSplitLinear (`TestGoVRocmStepProxyTrain/StepTweenSplitLinear` 69; STL1/STLO) |
| **0.76** | StepTweenSplitFastProxy (`TestGoVRocmStepProxyTrain/StepTweenSplitFastProxy` 69; STF1/STFO) |
| **0.77** | StepTweenSplitLinearCache (`TestGoVRocmStepProxyTrain/StepTweenSplitLinearCache` 76; SLC1/SLCO) |
| **0.78** | StepTweenSplitHeadProxyAsync (`TestGoVRocmStepProxyTrain/StepTweenSplitHeadProxyAsync` 69; SHA1/SHAO) |
| **0.79** | StepTweenSplitSparse (`TestGoVRocmStepProxyTrain/StepTweenSplitSparse` 69; SSP1/SSPO) |
| **0.x** | `round(earned)` from the engine table in [README.md](README.md) |
| **1.0.0** | Engine board **100/100** — full Go parity on AMD GPU (FormatNone path + Model/IO + Systems + CamSync + peak fused, or explicit gate rewrite) |

| In (moves `0.x`) | Out |
|------------------|-----|
| Layers, FormatNone×34, quants, train modes, Grid/Line/Mesh, Tween*, systems, model/IO — **on ROCm/HIP** | Other vendors (CUDA/Metal/QNN) — sibling repos |
| Evidence via w2a `Go_v_Rocm` | welvet.cpp / welvet.cuda / welvet.metal / Go scores (not inherited) |
| Determinism + speed on named AMD GPU | Packaging polish alone |

**Sibling map:** welvet (Go ref) · welvet.cpp (CPU) · welvet.metal (Apple GPU) · welvet.cuda (NVIDIA) · **welvet.rocm** (this) · welvet.qnn · w2a.
