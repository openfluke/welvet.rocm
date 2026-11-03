# welvet.rocm gap census

**Authoritative checklist:** [README.md](README.md).  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.27`** — engine board **27 / 100**. Dump: `welvet_dense_fwd --gaps`.

```bash
./scripts/build_linux.sh
./build/example_rocm_smoke
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --smoke
./scripts/linux_build_w2a.sh --leaf-fwd-all
./scripts/linux_build_w2a.sh --dense-train
./scripts/linux_build_w2a.sh --rmsnorm-train
# as suites land (mirror welvet.cuda → 0.79):
# ./scripts/linux_build_w2a.sh --normalbp-all
# ./scripts/linux_build_w2a.sh --training-all
```

## Done

| Slice | Detail |
|---|---|
| M0 ROCm/HIP host | device, buffers, smoke saxpy, C ABI, Linux |
| FormatNone×34 + Dense MatVec | pack/decode + resident f64 HIP MatVec |
| Leaf FormatNone forward | all 22 layers + seqmix — `go_v_rocm` (`--leaf-fwd-all`) |
| Dense NormalBP | leaf+vol ×34 (`WDB1`/`WVN1`) — `TestGoVRocmDenseTrain` / `--dense-train` |
| RMSNorm NormalBP | leaf+vol ×34 (`WRB1`/`WRV1`) — `TestGoVRocmRMSNormTrain` / `--rmsnorm-train` |

## Open

LayerNorm…Stack NormalBP (full Mode 1 → 0.51), StepBP / MeshBP / Tween* / proxy / CamSync /
Freeze/Shadow / model / peak fused / NAC. See README / TRAINING_CHECKLIST.

Live counters: `./build/welvet_dense_fwd --gaps`.
