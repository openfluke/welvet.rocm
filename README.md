# welvet.rocm — Welvet (3D volumetric AI engine) on AMD ROCm / HIP

**welvet.rocm is a full rebuild of Welvet — the 3D volumetric AI engine — in C++ / HIP.**
Same engine shape as Go Welvet and sibling `welvet.cuda`: numerical types, layers,
volumetric architecture, forward/backward, training modes, systems, model loading.
Execution is **ROCm compute + resident device buffers**. Host orchestration is C++.

| | |
|--|--|
| **Version** | **`0.27`** (leaf FormatNone ×22 + seqmix; Dense + RMSNorm NormalBP signed) |
| **Engine board** | **27 / 100** |
| **Compute** | AMD GPU (ROCm / HIP) — primary · CUDA / Metal / CPU siblings remain separate repos |
| **Parity reference** | [welvet](https://github.com/openfluke/welvet) Go |
| **CUDA sibling (target ladder)** | [welvet.cuda](https://github.com/openfluke/welvet.cuda) **0.79** — **score not inherited** |
| **Validation** | [w2a](https://github.com/openfluke/w2a) `Go_v_Rocm` leaf + Dense/RMSNorm NormalBP |

**This README is the runway.** Tick ⬜ → 🚧 → ✅ as gates land.
**Training port detail** lives in [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md)
(exact case/wire mirror of cuda). Pure-ROCm contract: [docs/ROCM_ONLY.md](docs/ROCM_ONLY.md).
Gap census: [GAPS.md](GAPS.md). Version formula: [VERSIONING.md](VERSIONING.md).

### What exists today

```bash
./scripts/build_linux.sh
./build/example_rocm_smoke
./build/welvet_dense_fwd --version   # 0.27
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --leaf-fwd-all
./scripts/linux_build_w2a.sh --dense-train
./scripts/linux_build_w2a.sh --rmsnorm-train
```

| Slice | Status |
|---|:---:|
| Linux x86_64 CMake → `libwelvet_rocm.so` | ✅ |
| HIP device / buffer / smoke saxpy | ✅ |
| Leaf FormatNone forward ×22 + seqmix (`go_v_rocm`) | ✅ signed **0.27** |
| Dense NormalBP leaf+vol (`WDB1`/`WVN1`, 7 cases) | ✅ signed **0.27** |
| RMSNorm NormalBP leaf+vol (`WRB1`/`WRV1`, 7 cases) | ✅ signed **0.27** |
| Other NormalBP … StepTweenSplitSparse (cuda 0.79 ladder) | ⬜ |
| CamSync / Freeze/Shadow / model | ⬜ |

Climb the checklist toward **cuda 0.79**. Determinism testing is done in sibling
**w2a** (`go_v_rocm`) as each engine gate is built — same pattern as `go_v_cuda`.

### Non-negotiable

- Numerical paths run on **HIP/ROCm** — no silent host math labeled ROCm.
- Evidence via w2a spectrum (`industry` → `exact_bit`); report the band.
- Sibling scores (cuda/metal/cpp/Go) are **not** inherited.

### Relationship

| Repo | Role |
|------|------|
| `welvet` | Go reference engine |
| `welvet.cpp` | CPU native |
| `welvet.cuda` | NVIDIA CUDA native (board **0.79**) |
| `welvet.metal` | Apple Metal native |
| **`welvet.rocm`** | **AMD ROCm / HIP native (this)** |
| `w2a` | Determinism / parity suites |

Requires Fedora/Ubuntu with ROCm (`hipcc`, `rocm-hip-devel`) and a **ROCm-supported**
AMD GPU for compute kernels. Graphics-only / unsupported GCN cards may probe via
`rocm-smi` but will not pass HSA compute — see AMD support matrices.
