// Welvet ROCm kernels — M0 scaffold reference copy.
// The build compiles src/rocm/kernels.hip into the library.
// Dense MatVec / activations / backward kernels land as gates flip in GAPS.md.
//
// Identity saxpy-style smoke kernel: y[i] = a * x[i] + b
// See src/rocm/kernels.hip for the compiled definitions.
