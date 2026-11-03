# Go / NAC integration — welvet.rocm

M0 ships the C ABI surface used by sibling **w2a** smoke:

| Symbol | Role |
|--------|------|
| `welvet_version` | Engine board tag (`0.00` …) |
| `welvet_runtime_identity` | `rocm:<name>:idx=…:gcn=…:mem=…` |
| `welvet_rocm_init` / `welvet_rocm_available` | Device probe |
| `welvet_gaps_count` / `welvet_gaps_done` | Gap census |

Leaf FormatNone forwards, NormalBP, TrainModes and model sessions land as
[TRAINING_CHECKLIST.md](TRAINING_CHECKLIST.md) rows are signed via
`w2a` `go_v_rocm` (determinism spectrum: `industry` → `exact_bit`).

Wire names and descriptor layouts mirror `welvet.cuda`
([GO_INTEGRATION.md](../../welvet.cuda/docs/GO_INTEGRATION.md)) with
`welvet_rocm_*` / `libwelvet_rocm.so` substituted for the CUDA names.
