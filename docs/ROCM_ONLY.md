# ROCm execution contract

`welvet.rocm` is a standalone native AMD GPU backend (HIP / ROCm). `welvet.cpp`
is the separate CPU backend; `welvet.rocm` is the AMD GPU sibling;
`welvet.metal` is the Apple GPU sibling. Each ships through its own native
artifact and C ABI. There is no dependency from one backend to the other, and
**no host numerical fallback** labeled as ROCm.

## Supported computation

**M0** (host scaffolding — not engine-board credit alone):

- HIP device probe + runtime identity string
- Device buffer alloc / H2D / D2H
- Compiled smoke kernel `welvet_smoke_saxpy`
- C ABI: version / init / available / gaps

**M1+** (engine credit — leaf FormatNone forwards):

- FormatNone×34 decode → resident **HW float64** MatVec / layer kernels on HIP
- Dense + 21 other layers + seqmix leaf forward (Stack deferred to NormalBP)
- Wire tool `welvet_dense_fwd` multiplexes leaf magics — w2a `go_v_rocm` exact_bit×34

Everything else (NormalBP, volume, train modes, …) is gap until ticked in
[TRAINING_CHECKLIST.md](TRAINING_CHECKLIST.md) and the [README runway](../README.md).
Climb the **same mode ladder** as `welvet.rocm` (currently **0.79**).

## Non-negotiable

- Supported numerical forward/backward/training paths run on **ROCm / HIP**.
- Host code may schedule, validate, allocate and stage storage.
- Unsupported dtype/layer/mode → hard error (no silent CPU math).
- FormatNone + DType (or packed quant) is storage truth — not an undeclared f32 master.
- Parity vs Go is best-effort **60–100%** on w2a's spectrum; report the band.

## Build and delivery

```sh
./scripts/build_linux.sh
./build/example_rocm_smoke
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --smoke
# as suites land:
# ./scripts/linux_build_w2a.sh --leaf-fwd-all
```

Tests and CPU parity references live in sibling `w2a`, not this repository.
Shared artifact: `libwelvet_rocm.so`. Requires CMake + AMD ROCm (`hipcc` / HIP runtime)
and a supported GCN/RDNA target (see AMD ROCm GPU support matrix).

Determinism evidence is collected in sibling **w2a** (`go_v_rocm`) as each
engine gate lands — same pattern as `go_v_rocm` / `go_v_metal` / `go_v_cpp`.
