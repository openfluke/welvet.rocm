# welvet.rocm gap census

**Authoritative checklist:** [README.md](README.md).  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.00`** — engine board **0 / 100**. Dump: `welvet_dense_fwd --gaps`.

```bash
./scripts/build_linux.sh
./build/example_rocm_smoke
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --smoke
# as suites land (mirror welvet.rocm → 0.79):
# ./scripts/linux_build_w2a.sh --leaf-fwd-all
# ./scripts/linux_build_w2a.sh --normalbp-all
# ./scripts/linux_build_w2a.sh --training-all
```

## Done

| Slice | Detail |
|---|---|
| M0 ROCm/HIP host | device, buffers, smoke saxpy, C ABI, Linux |

## Open

Everything else on the cuda **0.79** ladder: FormatNone×34, leaf forwards, NormalBP,
StepBP / MeshBP / Tween* / proxy / CamSync / Freeze/Shadow / model / peak fused / NAC.

Live counters: `./build/welvet_dense_fwd --gaps`.
