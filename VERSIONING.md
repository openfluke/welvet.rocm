# welvet.rocm versioning

**One scoreboard:** same ten categories / 100 points as Go Welvet.  
**Formula:** `version = 0.{round(earned)}` until **100 → 1.0.0**.

| Tag | Meaning |
|-----|---------|
| **0.00** | Bedrock — ROCm/HIP host scaffold; **0 / 100** on the engine board |
| **0.13** | M1 Dense FormatNone×34 forward signed (Foundation 9 + Dense fwd 4) |
| **0.27** | Leaf FormatNone forwards ×22 layers + seqmix (Stack→NormalBP) — **current** |
| **0.51** | NormalBP Mode 1 signed for all 23 layers (`TestGoVRocmNormalBP`) |
| **0.52…0.79** | Same train-mode ladder as [welvet.cuda/VERSIONING.md](../welvet.cuda/VERSIONING.md) (`TestGoVRocm*` evidence) |
| **0.x** | `round(earned)` from the engine table in [README.md](README.md) |
| **1.0.0** | Engine board **100/100** — full Go parity on AMD GPU (FormatNone path + Model/IO + Systems + CamSync + peak fused, or explicit gate rewrite) |

| In (moves `0.x`) | Out |
|------------------|-----|
| Layers, FormatNone×34, quants, train modes, Grid/Line/Mesh, Tween*, systems, model/IO — **on ROCm/HIP** | Other vendors (CUDA/Metal/QNN) — sibling repos |
| Evidence via w2a `Go_v_Rocm` | welvet.cpp / welvet.cuda / welvet.metal / Go scores (not inherited) |
| Determinism + speed on named AMD GPU | Packaging polish alone |

**Sibling map:** welvet (Go ref) · welvet.cpp (CPU) · welvet.metal (Apple GPU) · welvet.cuda (NVIDIA) · **welvet.rocm** (this) · welvet.qnn · w2a.

Climb the checklist toward cuda's **0.79** board; do not inherit cuda credit.
