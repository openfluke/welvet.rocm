# welvet.rocm docs

| Doc | Role |
|-----|------|
| **[../README.md](../README.md)** | **Authoritative runway** — full Go Welvet inventory as ROCm checkboxes (3D volumetric engine) |
| **[TRAINING_CHECKLIST.md](TRAINING_CHECKLIST.md)** | **Training port tracker** — NormalBP → all 29 TrainModes + Freeze/Shadow/Adv/Memory (Go↔ROCm evidence + per-mode layer tables). Mirror of cuda/metal/cpp checklists; ROCm status only; start unchecked at **0.00**, climb toward cuda **0.79**. |
| [ROCM_ONLY.md](ROCM_ONLY.md) | Pure-ROCm execution contract (no host numerical fallback) |
| [GO_INTEGRATION.md](GO_INTEGRATION.md) | C ABI / NAC notes (stub until model sessions land) |
| [../GAPS.md](../GAPS.md) | Machine gap dump (`--gaps`) |
| [../VERSIONING.md](../VERSIONING.md) | `0.x` board formula |
