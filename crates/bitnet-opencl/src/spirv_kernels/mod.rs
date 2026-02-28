//! SPIR-V kernel registry.
//!
//! Maps kernel names to their source representation — either an embedded
//! `.cl` source string (for runtime / JIT compilation) or pre-compiled
//! SPIR-V bytes.

use std::collections::HashMap;

// ── Kernel source representation ─────────────────────────────────────────────

/// How a kernel's code is stored in the registry.
#[derive(Debug, Clone)]
pub enum KernelSource {
    /// Raw `OpenCL` C source (`.cl`), used when no offline compiler was
    /// available at build time.
    ClSource(String),
    /// Pre-compiled SPIR-V binary.
    SpirV(Vec<u8>),
}

// ── Registry ─────────────────────────────────────────────────────────────────

/// Registry mapping kernel names to their [`KernelSource`].
///
/// Populated at startup from either embedded `.cl` sources or, when a
/// `build_spirv` step has run, from the resulting `.spv` blobs.
#[derive(Debug, Clone)]
pub struct SpirvKernelRegistry {
    kernels: HashMap<String, KernelSource>,
}

impl SpirvKernelRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { kernels: HashMap::new() }
    }

    /// Register a kernel under `name`.
    pub fn register(&mut self, name: impl Into<String>, source: KernelSource) {
        self.kernels.insert(name.into(), source);
    }

    /// Look up a kernel by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&KernelSource> {
        self.kernels.get(name)
    }

    /// Number of kernels in the registry.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Returns `true` if the registry contains no kernels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Iterate over all registered kernel names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.kernels.keys().map(String::as_str)
    }

    /// Build a default registry pre-loaded with the built-in `.cl`
    /// kernel sources shipped with this crate.
    #[must_use]
    pub fn with_builtin_kernels() -> Self {
        let mut reg = Self::new();
        // Placeholder: real kernels would be included here via
        // `include_str!()` once the `.cl` files exist.
        reg.register("matmul_i2", KernelSource::ClSource(MATMUL_I2_CL.into()));
        reg.register("dequant_i2s", KernelSource::ClSource(DEQUANT_I2S_CL.into()));
        reg
    }
}

impl Default for SpirvKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Embedded kernel sources (stubs) ──────────────────────────────────────────

/// Stub `.cl` source for the `matmul_i2` kernel.
const MATMUL_I2_CL: &str = "\
__kernel void matmul_i2(
    __global const uchar* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += (float)A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
";

/// Stub `.cl` source for the `dequant_i2s` kernel.
const DEQUANT_I2S_CL: &str = "\
__kernel void dequant_i2s(
    __global const uchar* packed,
    __global float* output,
    const float scale,
    const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        int byte_idx = gid / 4;
        int bit_off  = (gid % 4) * 2;
        int val = (packed[byte_idx] >> bit_off) & 0x3;
        float dequant = (val == 0) ? -1.0f :
                        (val == 1) ?  0.0f :
                                      1.0f;
        output[gid] = dequant * scale;
    }
}
";
