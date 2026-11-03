# welvet.rocm gap census

**Authoritative checklist:** [README.md](README.md).  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.79`** — engine board **79 / 100**. Dump: `welvet_dense_fwd --gaps`.

```bash
./scripts/build_linux.sh
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --leaf-fwd-all
./scripts/linux_build_w2a.sh --normalbp-all
./scripts/linux_build_w2a.sh --stepbp-train
./scripts/linux_build_w2a.sh --meshbp-train
./scripts/linux_build_w2a.sh --tween-train
./scripts/linux_build_w2a.sh --tweenchain-train
./scripts/linux_build_w2a.sh --tweensplit-train
./scripts/linux_build_w2a.sh --steptweensplit-train
./scripts/linux_build_w2a.sh --tweenalt-train
./scripts/linux_build_w2a.sh --steptweenalt-train
./scripts/linux_build_w2a.sh --meshtween-train
./scripts/linux_build_w2a.sh --meshtweenchain-train
./scripts/linux_build_w2a.sh --steptween-train
./scripts/linux_build_w2a.sh --steptweenchain-train
./scripts/linux_build_w2a.sh --proxy-train
./scripts/linux_build_w2a.sh --mesh-split-modes
```

## Done

| Slice | Detail |
|---|---|
| M0 ROCm/HIP host | device, buffers, smoke, C ABI, Linux |
| FormatNone×34 | pack/decode + Dense MatVec/SGD |
| Leaf FormatNone forward | all 23 layers + seqmix (w2a exact_bit) |
| **NormalBP Mode 1** | leaf+volume for all 23 layers — `TestGoVRocmNormalBP` PASS |
| **StepBP Line + Dense MeshBP** | STB1 + MBP1 + ModeMeshBP vols — `TestGoVRocmStepBPTrain` PASS |
| **Heterogeneous MeshBP** | MGB1/MGBO all 23 families, 103 groups — `TestGoVRocmMeshBPTrain` PASS |
| **ModeTween** | TWB1/TWV1/per-kind/KMTV/PNTV — 130 groups — `TestGoVRocmTweenTrain` PASS |
| **ModeTweenChain** | TCB1/TCV1/per-kind/KMTC/PNTC — 130 groups — `TestGoVRocmTweenChainTrain` PASS |
| **ModeTweenSplit** | TSB1/TSP1/per-kind/KMSP/PNSP — 131 groups — `TestGoVRocmTweenSplitTrain` PASS |
| **ModeStepTweenSplit** | STS1/STSO + TweenSplit volumes — 130 groups — `TestGoVRocmStepTweenSplitTrain` PASS |
| **ModeTweenAlt** | TAB1/TAV1/per-kind/KMAL/PNAL — 130 groups — `TestGoVRocmTweenAltTrain` PASS |
| **ModeStepTweenAlt** | STA1/STAO + TweenAlt volumes — 130 groups — `TestGoVRocmStepTweenAltTrain` PASS |
| **MeshTween** | MTW1/MTWO + MGW1/MGWO + ModeTween volumes — 159 groups — `TestGoVRocmMeshTweenTrain` PASS |
| **MeshTweenChain** | MTC1/MTCO + MGC1/MGCO ≡ MeshBP FamilyBP — 160 groups — `TestGoVRocmMeshTweenChainTrain` PASS |
| **ModeStepTween** | STW1/STWO Line FamilyTween @ lr×0.5 + Tween volumes — 130 groups — `TestGoVRocmStepTweenTrain` PASS |
| **ModeStepTweenChain** | STC1/STCO Line FamilyBP alias + TweenChain volumes — 130 groups — `TestGoVRocmStepTweenChainTrain` PASS |
| **ModeTweenSplitHeadProxy** | GPB1/GPBO + PGB1/PGVO variant 1 — 65 leaf + 69 volume — `TestGoVRocmProxyTrain`/`ProxyVolumeTrain` PASS |
| **ModeTweenSplitLinear** | GPB1/GPBO + PGB1/PGVO variant 2 — 65 leaf + 69 volume — `TestGoVRocmProxyTrain`/`ProxyVolumeTrain` PASS |
| **ModeTweenSplitFastProxy** | GPB1/GPBO + PGB1/PGVO variant 3 — 65 leaf + 69 volume — `TestGoVRocmProxyTrain`/`ProxyVolumeTrain` PASS |
| **ModeTweenSplitLinearCache** | GPB1/GPBO + PGB1/PGVO variant 4 — 88 leaf + 92 volume (incl. epoch-22 refresh) — PASS |
| **ModeTweenSplitHeadProxyAsync** | GPB1/GPBO + PGB1/PGVO variant 5 — 65 leaf + 69 volume (4-epoch stale reuse) — PASS |
| **ModeTweenSplitSparse** | GPB1/GPBO + PGB1/PGVO variant 6 — 65 leaf + 69 volume (4-epoch rotate) — PASS |
| **MeshTweenSplit** | MGS1/MGSO FamilyTweenSplit 1/N @ full lr — 154 groups — `TestGoVRocmMeshSplitModes` PASS |
| **MeshTweenAlt** | MGA1/MGAO Split→Tween@lr×0.5 — 154 groups — `TestGoVRocmMeshSplitModes` PASS |
| **MeshTweenSplitFastProxy** | MGF1/MGFO last-tick FastProxy — 172 groups — `TestGoVRocmMeshSplitModes` PASS |
| **MeshTweenSplitSparse** | MGQ1/MGQO last-tick Sparse rotate — 172 groups — `TestGoVRocmMeshSplitModes` PASS |
| **StepTweenSplitHeadProxy** | STH1/STHO Line HeadProxy — 65 groups — `TestGoVRocmProxyTrain/StepTweenSplitHeadProxy` PASS |
| **StepTweenSplitLinear** | STL1/STLO variant 2 — 69 groups — `TestGoVRocmStepProxyTrain/StepTweenSplitLinear` PASS |
| **StepTweenSplitFastProxy** | STF1/STFO variant 3 — 69 groups — `TestGoVRocmStepProxyTrain/StepTweenSplitFastProxy` PASS |
| **StepTweenSplitLinearCache** | SLC1/SLCO variant 4 — 76 groups — `TestGoVRocmStepProxyTrain/StepTweenSplitLinearCache` PASS |
| **StepTweenSplitHeadProxyAsync** | SHA1/SHAO variant 5 — 69 groups — `TestGoVRocmStepProxyTrain/StepTweenSplitHeadProxyAsync` PASS |
| **StepTweenSplitSparse** | SSP1/SSPO variant 6 — 69 groups — `TestGoVRocmStepProxyTrain/StepTweenSplitSparse` PASS |

## Open

CamSync / Freeze/Shadow / other systems / model / lucy / peak fused / NAC / packed quants. Windows ROCm artifact open (Linux primary). See README.

Live counters: `./build/welvet_dense_fwd --gaps`.
