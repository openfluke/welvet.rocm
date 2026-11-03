# welvet.rocm gap census

**Authoritative checklist:** [README.md](README.md).  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.52`** — engine board **52 / 100**. Dump: `welvet_dense_fwd --gaps`.

```bash
./scripts/build_linux.sh
./build/example_rocm_smoke
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --smoke
./scripts/linux_build_w2a.sh --leaf-fwd-all
./scripts/linux_build_w2a.sh --normalbp-all
./scripts/linux_build_w2a.sh --stepbp-train
```

## Done

| Slice | Detail |
|---|---|
| M0 ROCm/HIP host | device, buffers, smoke saxpy, C ABI, Linux |
| FormatNone×34 + Dense MatVec | pack/decode + resident f64 HIP MatVec |
| Leaf FormatNone forward | all 22 layers + seqmix — `go_v_rocm` (`--leaf-fwd-all`) |
| NormalBP Mode 1 | 23 families leaf+vol — `TestGoVRocmNormalBP` / `--normalbp-all` |
| StepBP Line + Dense MeshBP | `STB1`/`MBP1` — `TestGoVRocmStepBPTrain` / `--stepbp-train` |

## Open

Heterogeneous MeshBP (0.53), Tween* / proxy / CamSync /
Freeze/Shadow / model / peak fused / NAC. See README / TRAINING_CHECKLIST.

Live counters: `./build/welvet_dense_fwd --gaps`.
