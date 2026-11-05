//! Memory bandwidth optimization techniques for Intel Arc A770 GPUs.
//!
//! Provides CPU reference implementations and analysis tools for optimizing
//! global memory throughput on Xe-HPG hardware. All components work without
//! an OpenCL runtime — the embedded kernel source strings demonstrate the
//! techniques for eventual GPU dispatch.
//!
//! # Techniques
//!
//! | Technique | A770 target |
//! |-----------|-------------|
//! | Coalesced access | 128-byte transaction alignment |
//! | Vectorized loads | `float2`/`float4`/`float8`/`float16` |
//! | SLM tiling | 64 KB shared local memory per sub-slice |
//! | Async prefetch | Hide DRAM latency behind compute |
//! | Cache-line alignment | 64 B / 128 B boundary alignment |

use std::fmt;

// ───────────────────────────────────────────────────────────────────────────
// Constants — Intel Arc A770 (Xe-HPG) hardware parameters
// ───────────────────────────────────────────────────────────────────────────

/// Cache-line size in bytes for the A770 L1/L2 (64 B).
pub const CACHE_LINE_64: usize = 64;

/// Wider cache-line / transaction size used by the memory controller (128 B).
pub const CACHE_LINE_128: usize = 128;

/// Maximum shared local memory per sub-slice on A770 (64 KB).
pub const A770_SLM_SIZE: usize = 64 * 1024;

/// Theoretical peak memory bandwidth of the A770 in GB/s.
pub const A770_PEAK_BANDWIDTH_GBPS: f64 = 560.0;

/// SIMD width (subgroup size) for Xe-HPG.
pub const XE_SIMD_WIDTH: usize = 16;

/// Bytes per `float` element.
const FLOAT_SIZE: usize = 4;

// ───────────────────────────────────────────────────────────────────────────
// MemoryAccessPattern
// ───────────────────────────────────────────────────────────────────────────

/// Describes a memory access pattern for bandwidth analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryAccessPattern {
    /// Contiguous addresses — ideal for coalescing.
    Sequential,
    /// Fixed-stride access with the given stride in elements.
    Strided(usize),
    /// Unpredictable / pointer-chasing access.
    Random,
}

impl fmt::Display for MemoryAccessPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequential => write!(f, "sequential"),
            Self::Strided(s) => write!(f, "strided({})", s),
            Self::Random => write!(f, "random"),
        }
    }
}

impl MemoryAccessPattern {
    /// Estimated efficiency factor in `[0.0, 1.0]` relative to peak bandwidth.
    ///
    /// Sequential access achieves near-peak utilization; large strides and
    /// random patterns waste cache-line fetches.
    pub fn efficiency(&self) -> f64 {
        match self {
            Self::Sequential => 0.95,
            Self::Strided(stride) => {
                if *stride == 0 || *stride == 1 {
                    return 0.95;
                }
                // Larger strides waste more of each cache-line fetch.
                let useful_fraction = FLOAT_SIZE as f64
                    / (*stride as f64 * FLOAT_SIZE as f64).min(CACHE_LINE_64 as f64);
                useful_fraction.clamp(0.05, 0.95)
            }
            Self::Random => 0.05,
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// VectorWidth
// ───────────────────────────────────────────────────────────────────────────

/// OpenCL vector load width (number of f32 elements per load).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorWidth {
    /// Scalar `float` — 4 bytes.
    Float1,
    /// `float2` — 8 bytes.
    Float2,
    /// `float4` — 16 bytes.
    Float4,
    /// `float8` — 32 bytes.
    Float8,
    /// `float16` — 64 bytes.
    Float16,
}

impl VectorWidth {
    /// Number of `f32` elements in this vector.
    pub fn elements(self) -> usize {
        match self {
            Self::Float1 => 1,
            Self::Float2 => 2,
            Self::Float4 => 4,
            Self::Float8 => 8,
            Self::Float16 => 16,
        }
    }

    /// Size of the vector in bytes.
    pub fn byte_size(self) -> usize {
        self.elements() * FLOAT_SIZE
    }

    /// All supported widths in ascending order.
    pub fn all() -> &'static [VectorWidth] {
        &[Self::Float1, Self::Float2, Self::Float4, Self::Float8, Self::Float16]
    }

    /// Select the widest vector width that evenly divides `element_count`.
    pub fn best_for(element_count: usize) -> Self {
        for &w in Self::all().iter().rev() {
            if element_count.is_multiple_of(w.elements()) {
                return w;
            }
        }
        Self::Float1
    }
}

impl fmt::Display for VectorWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float1 => write!(f, "float"),
            Self::Float2 => write!(f, "float2"),
            Self::Float4 => write!(f, "float4"),
            Self::Float8 => write!(f, "float8"),
            Self::Float16 => write!(f, "float16"),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// CoalescedAccess
// ───────────────────────────────────────────────────────────────────────────

/// Analyzes and transforms memory accesses for coalesced (contiguous) patterns.
///
/// On Xe-HPG the memory controller issues 64-byte or 128-byte transactions;
/// coalesced accesses within a subgroup fetch exactly one transaction instead
/// of up to `SIMD_WIDTH` individual ones.
#[derive(Debug, Clone)]
pub struct CoalescedAccess {
    /// SIMD width of the target device.
    pub simd_width: usize,
    /// Transaction size in bytes.
    pub transaction_bytes: usize,
}

impl Default for CoalescedAccess {
    fn default() -> Self {
        Self { simd_width: XE_SIMD_WIDTH, transaction_bytes: CACHE_LINE_64 }
    }
}

impl CoalescedAccess {
    /// Create with custom hardware parameters.
    pub fn new(simd_width: usize, transaction_bytes: usize) -> Self {
        Self { simd_width, transaction_bytes }
    }

    /// Check whether `n` sequential float accesses starting at `base_offset`
    /// are fully coalesced within one transaction.
    pub fn is_coalesced(&self, base_offset: usize, n: usize) -> bool {
        if n == 0 {
            return true;
        }
        let start_byte = base_offset * FLOAT_SIZE;
        let end_byte = start_byte + n * FLOAT_SIZE;
        let start_line = start_byte / self.transaction_bytes;
        let end_line = (end_byte - 1) / self.transaction_bytes;
        start_line == end_line
    }

    /// Number of cache-line transactions needed for `n` contiguous floats
    /// starting at byte offset `base_byte_offset`.
    pub fn transactions_needed(&self, base_byte_offset: usize, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let end = base_byte_offset + n * FLOAT_SIZE;
        let first_line = base_byte_offset / self.transaction_bytes;
        let last_line = (end - 1) / self.transaction_bytes;
        last_line - first_line + 1
    }

    /// Number of transactions for a strided access pattern over `n` elements.
    pub fn strided_transactions(&self, base_byte_offset: usize, n: usize, stride: usize) -> usize {
        if n == 0 || stride == 0 {
            return 0;
        }
        let mut lines = std::collections::BTreeSet::new();
        for i in 0..n {
            let addr = base_byte_offset + i * stride * FLOAT_SIZE;
            lines.insert(addr / self.transaction_bytes);
        }
        lines.len()
    }

    /// CPU reference: read `src` sequentially into `dst`, simulating coalesced
    /// access within each `simd_width`-element group.
    pub fn coalesced_read(&self, src: &[f32], dst: &mut [f32]) {
        let n = src.len().min(dst.len());
        dst[..n].copy_from_slice(&src[..n]);
    }

    /// CPU reference: read `src` with stride, simulating strided access.
    pub fn strided_read(&self, src: &[f32], dst: &mut [f32], stride: usize) {
        if stride == 0 {
            return;
        }
        for (i, d) in dst.iter_mut().enumerate() {
            let idx = i * stride;
            *d = if idx < src.len() { src[idx] } else { 0.0 };
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// VectorizedLoad
// ───────────────────────────────────────────────────────────────────────────

/// Performs vectorized load/store operations at configurable widths.
///
/// Wider loads reduce instruction count and improve bus utilization on A770.
#[derive(Debug, Clone)]
pub struct VectorizedLoad {
    pub width: VectorWidth,
}

impl VectorizedLoad {
    pub fn new(width: VectorWidth) -> Self {
        Self { width }
    }

    /// Load `src` into `dst` in groups of `width` elements.
    ///
    /// Tail elements that do not fill a full vector are loaded one-by-one.
    pub fn load(&self, src: &[f32], dst: &mut [f32]) {
        let n = src.len().min(dst.len());
        let vec_elems = self.width.elements();
        let full_vecs = n / vec_elems;
        let tail = n % vec_elems;

        // Vectorized portion — copy in chunks.
        for v in 0..full_vecs {
            let start = v * vec_elems;
            dst[start..start + vec_elems].copy_from_slice(&src[start..start + vec_elems]);
        }

        // Scalar tail.
        let tail_start = full_vecs * vec_elems;
        dst[tail_start..tail_start + tail].copy_from_slice(&src[tail_start..tail_start + tail]);
    }

    /// Scale every element by `factor` using vectorized groups.
    pub fn scale(&self, data: &mut [f32], factor: f32) {
        let vec_elems = self.width.elements();
        let n = data.len();
        let full = n / vec_elems;
        let tail = n % vec_elems;

        for v in 0..full {
            let start = v * vec_elems;
            for e in &mut data[start..start + vec_elems] {
                *e *= factor;
            }
        }

        let tail_start = full * vec_elems;
        for e in &mut data[tail_start..tail_start + tail] {
            *e *= factor;
        }
    }

    /// Fused multiply-add: `dst[i] = a[i] * b[i] + c[i]`, vectorized.
    pub fn fma(&self, a: &[f32], b: &[f32], c: &[f32], dst: &mut [f32]) {
        let n = a.len().min(b.len()).min(c.len()).min(dst.len());
        let vec_elems = self.width.elements();
        let full = n / vec_elems;
        let tail = n % vec_elems;

        for v in 0..full {
            let s = v * vec_elems;
            for i in s..s + vec_elems {
                dst[i] = a[i] * b[i] + c[i];
            }
        }

        let tail_start = full * vec_elems;
        for i in tail_start..tail_start + tail {
            dst[i] = a[i] * b[i] + c[i];
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// SharedMemoryTiler
// ───────────────────────────────────────────────────────────────────────────

/// Tile-based computation that caches strips of global data in shared local
/// memory (SLM) to exploit reuse.
///
/// On the A770, each sub-slice has 64 KB of SLM. Tiling converts many
/// high-latency global loads into fast SLM reads.
#[derive(Debug, Clone)]
pub struct SharedMemoryTiler {
    /// Tile dimension (number of f32 elements per tile side).
    pub tile_size: usize,
    /// Available SLM in bytes.
    pub slm_bytes: usize,
}

impl SharedMemoryTiler {
    pub fn new(tile_size: usize, slm_bytes: usize) -> Self {
        Self { tile_size, slm_bytes }
    }

    /// Default A770 tiler: 16×16 tiles, 64 KB SLM.
    pub fn a770_default() -> Self {
        Self { tile_size: 16, slm_bytes: A770_SLM_SIZE }
    }

    /// Maximum tile side that fits two tiles (A, B) in SLM.
    pub fn max_tile_size(&self) -> usize {
        // Two square tiles of f32: 2 * tile^2 * 4 ≤ slm_bytes
        let max_elements = self.slm_bytes / (2 * FLOAT_SIZE);
        // tile^2 ≤ max_elements ⟹ tile ≤ sqrt(max_elements)
        (max_elements as f64).sqrt() as usize
    }

    /// Number of tiles needed to cover an `m × n` matrix.
    pub fn tile_count(&self, m: usize, n: usize) -> (usize, usize) {
        (m.div_ceil(self.tile_size), n.div_ceil(self.tile_size))
    }

    /// Total global-memory loads saved by tiling an `(m, k) × (k, n)` matmul.
    ///
    /// Without tiling each output element reads an entire row of A and column
    /// of B from global memory. With tiling, each tile strip is loaded once
    /// into SLM and reused `tile_size` times.
    ///
    /// Returns `(tiled_loads, naive_loads)`.
    pub fn global_load_comparison(&self, m: usize, k: usize, n: usize) -> (usize, usize) {
        let t = self.tile_size;
        let tiles_m = m.div_ceil(t);
        let tiles_n = n.div_ceil(t);
        let tiles_k = k.div_ceil(t);

        // Tiled: for each (tm, tn) pair we iterate over tk tiles,
        // loading one A-tile (t×t) and one B-tile (t×t) each iteration.
        let tiled_loads = tiles_m * tiles_n * tiles_k * 2 * t * t;

        // Naïve: each of (m*n) output elements reads k values from A and
        // k values from B.
        let naive_loads = m * n * k * 2;

        (tiled_loads, naive_loads)
    }

    /// CPU reference: tiled matrix multiply `C = A × B`.
    ///
    /// A is `(m, k)` row-major, B is `(k, n)` row-major, C is `(m, n)`.
    pub fn tiled_matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        assert!(a.len() >= m * k, "A too small");
        assert!(b.len() >= k * n, "B too small");
        assert!(c.len() >= m * n, "C too small");

        let t = self.tile_size;

        // Zero output.
        for v in c[..m * n].iter_mut() {
            *v = 0.0;
        }

        for tm in (0..m).step_by(t) {
            for tn in (0..n).step_by(t) {
                for tk in (0..k).step_by(t) {
                    // Simulated SLM tile loads.
                    let tm_end = (tm + t).min(m);
                    let tn_end = (tn + t).min(n);
                    let tk_end = (tk + t).min(k);
                    for i in tm..tm_end {
                        for j in tn..tn_end {
                            let mut sum = 0.0_f32;
                            for p in tk..tk_end {
                                sum += a[i * k + p] * b[p * n + j];
                            }
                            c[i * n + j] += sum;
                        }
                    }
                }
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// PrefetchScheduler
// ───────────────────────────────────────────────────────────────────────────

/// Prefetch hint distance in elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchDistance {
    /// Prefetch the next cache line.
    Near,
    /// Prefetch 2–4 cache lines ahead.
    Medium,
    /// Prefetch 8+ cache lines ahead (hide DRAM latency).
    Far,
}

impl PrefetchDistance {
    /// Distance in bytes.
    pub fn bytes(self) -> usize {
        match self {
            Self::Near => CACHE_LINE_64,
            Self::Medium => CACHE_LINE_64 * 4,
            Self::Far => CACHE_LINE_64 * 8,
        }
    }

    /// Distance in f32 elements.
    pub fn elements(self) -> usize {
        self.bytes() / FLOAT_SIZE
    }
}

/// Schedules asynchronous prefetches to overlap memory latency with compute.
#[derive(Debug, Clone)]
pub struct PrefetchScheduler {
    /// Prefetch distance.
    pub distance: PrefetchDistance,
    /// Number of elements to process per iteration before issuing next prefetch.
    pub compute_chunk: usize,
}

impl PrefetchScheduler {
    pub fn new(distance: PrefetchDistance, compute_chunk: usize) -> Self {
        let compute_chunk = if compute_chunk == 0 { 1 } else { compute_chunk };
        Self { distance, compute_chunk }
    }

    /// Default for A770: far prefetch, 256-element compute chunk.
    pub fn a770_default() -> Self {
        Self { distance: PrefetchDistance::Far, compute_chunk: 256 }
    }

    /// Number of prefetch instructions issued for a buffer of `n` elements.
    pub fn prefetch_count(&self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        n.div_ceil(self.compute_chunk)
    }

    /// CPU reference: process `data` in chunks, "prefetching" the next chunk
    /// while accumulating the current one.
    ///
    /// Returns the dot-product `Σ data[i] * weights[i]`.
    pub fn prefetched_dot(&self, data: &[f32], weights: &[f32]) -> f32 {
        let n = data.len().min(weights.len());
        let mut acc = 0.0_f32;

        for chunk_start in (0..n).step_by(self.compute_chunk) {
            let chunk_end = (chunk_start + self.compute_chunk).min(n);
            // In a real GPU kernel we would issue an async prefetch here for
            // the *next* chunk at (chunk_start + distance.elements()).
            for i in chunk_start..chunk_end {
                acc += data[i] * weights[i];
            }
        }
        acc
    }

    /// Generate an OpenCL prefetch call string for documentation / source gen.
    pub fn opencl_prefetch_call(&self, ptr_name: &str, offset_expr: &str) -> String {
        let dist_elements = self.distance.elements();
        format!("prefetch({ptr_name} + {offset_expr}, {dist_elements});")
    }
}

// ───────────────────────────────────────────────────────────────────────────
// CacheLineAligner
// ───────────────────────────────────────────────────────────────────────────

/// Aligns buffer sizes and offsets to cache-line boundaries.
#[derive(Debug, Clone, Copy)]
pub struct CacheLineAligner {
    /// Alignment in bytes (must be a power of two).
    pub alignment: usize,
}

impl CacheLineAligner {
    /// Create an aligner for the given byte boundary.
    ///
    /// # Panics
    ///
    /// Panics if `alignment` is zero or not a power of two.
    pub fn new(alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "alignment must be a power of two");
        Self { alignment }
    }

    /// 64-byte aligned (A770 L1 cache line).
    pub fn cl64() -> Self {
        Self { alignment: CACHE_LINE_64 }
    }

    /// 128-byte aligned (A770 memory transaction).
    pub fn cl128() -> Self {
        Self { alignment: CACHE_LINE_128 }
    }

    /// Round `size_bytes` up to the next aligned boundary.
    pub fn align_size(&self, size_bytes: usize) -> usize {
        let mask = self.alignment - 1;
        (size_bytes + mask) & !mask
    }

    /// Number of padding bytes needed to align `size_bytes`.
    pub fn padding(&self, size_bytes: usize) -> usize {
        self.align_size(size_bytes) - size_bytes
    }

    /// Check whether `addr` is aligned.
    pub fn is_aligned(&self, addr: usize) -> bool {
        addr.is_multiple_of(self.alignment)
    }

    /// Align an f32 element count so the resulting byte size is aligned.
    pub fn align_element_count(&self, element_count: usize) -> usize {
        let byte_size = element_count * FLOAT_SIZE;
        self.align_size(byte_size) / FLOAT_SIZE
    }

    /// Create an aligned buffer of at least `min_elements` f32 values,
    /// zero-initialized, whose byte length is a multiple of `alignment`.
    pub fn alloc_aligned(&self, min_elements: usize) -> Vec<f32> {
        let aligned = self.align_element_count(min_elements);
        vec![0.0; aligned]
    }
}

// ───────────────────────────────────────────────────────────────────────────
// BandwidthEstimator
// ───────────────────────────────────────────────────────────────────────────

/// Estimates effective memory bandwidth for a given configuration.
#[derive(Debug, Clone)]
pub struct BandwidthEstimator {
    /// Theoretical peak bandwidth in GB/s.
    pub peak_gbps: f64,
}

impl Default for BandwidthEstimator {
    fn default() -> Self {
        Self { peak_gbps: A770_PEAK_BANDWIDTH_GBPS }
    }
}

impl BandwidthEstimator {
    pub fn new(peak_gbps: f64) -> Self {
        Self { peak_gbps }
    }

    /// Estimate effective bandwidth for an access pattern.
    pub fn estimate(&self, pattern: MemoryAccessPattern) -> f64 {
        self.peak_gbps * pattern.efficiency()
    }

    /// Estimate transfer time in nanoseconds for `bytes` at the given pattern.
    pub fn estimate_time_ns(&self, bytes: usize, pattern: MemoryAccessPattern) -> f64 {
        let effective_gbps = self.estimate(pattern);
        if effective_gbps <= 0.0 {
            return f64::INFINITY;
        }
        // GB/s == bytes/ns, so time_ns = bytes / effective_gbps
        bytes as f64 / effective_gbps
    }

    /// Estimate bandwidth given a vector width and access pattern.
    pub fn estimate_vectorized(&self, pattern: MemoryAccessPattern, width: VectorWidth) -> f64 {
        let base = self.estimate(pattern);
        // Wider vectors reduce instruction overhead; model a small uplift
        // (diminishing returns past float4 on Xe-HPG).
        let vector_factor = match width {
            VectorWidth::Float1 => 1.0,
            VectorWidth::Float2 => 1.05,
            VectorWidth::Float4 => 1.10,
            VectorWidth::Float8 => 1.12,
            VectorWidth::Float16 => 1.13,
        };
        (base * vector_factor).min(self.peak_gbps)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// BandwidthStats
// ───────────────────────────────────────────────────────────────────────────

/// Bottleneck type identified by bandwidth analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bottleneck {
    /// Memory bandwidth is the limiting factor.
    MemoryBound,
    /// Compute throughput is the limiting factor.
    ComputeBound,
    /// Kernel launch / synchronization overhead dominates.
    LatencyBound,
    /// No clear bottleneck (balanced).
    Balanced,
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryBound => write!(f, "memory-bound"),
            Self::ComputeBound => write!(f, "compute-bound"),
            Self::LatencyBound => write!(f, "latency-bound"),
            Self::Balanced => write!(f, "balanced"),
        }
    }
}

/// Measured / estimated bandwidth statistics for a kernel or transfer.
#[derive(Debug, Clone)]
pub struct BandwidthStats {
    /// Measured bandwidth in GB/s.
    pub measured_gbps: f64,
    /// Theoretical peak bandwidth in GB/s.
    pub peak_gbps: f64,
    /// Total bytes transferred.
    pub bytes_transferred: usize,
    /// Elapsed time in nanoseconds.
    pub elapsed_ns: u64,
    /// Identified bottleneck.
    pub bottleneck: Bottleneck,
    /// Optimization suggestions (human-readable).
    pub suggestions: Vec<String>,
}

impl BandwidthStats {
    /// Utilization as a fraction in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        if self.peak_gbps <= 0.0 {
            return 0.0;
        }
        (self.measured_gbps / self.peak_gbps).clamp(0.0, 1.0)
    }

    /// Utilization as a percentage.
    pub fn utilization_pct(&self) -> f64 {
        self.utilization() * 100.0
    }
}

impl fmt::Display for BandwidthStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1} GB/s ({:.1}% of {:.0} GB/s peak) — {}",
            self.measured_gbps,
            self.utilization_pct(),
            self.peak_gbps,
            self.bottleneck,
        )
    }
}

/// Build `BandwidthStats` from a CPU-side measurement.
pub fn measure_bandwidth(bytes: usize, elapsed_ns: u64, peak_gbps: f64) -> BandwidthStats {
    let measured_gbps = if elapsed_ns == 0 {
        0.0
    } else {
        bytes as f64 / elapsed_ns as f64 // bytes/ns == GB/s
    };

    let utilization = if peak_gbps > 0.0 { measured_gbps / peak_gbps } else { 0.0 };

    let bottleneck = if elapsed_ns == 0 {
        Bottleneck::Balanced
    } else if utilization > 0.6 {
        Bottleneck::MemoryBound
    } else if utilization < 0.1 && bytes < 4096 {
        Bottleneck::LatencyBound
    } else if utilization < 0.3 {
        Bottleneck::ComputeBound
    } else {
        Bottleneck::Balanced
    };

    let mut suggestions = Vec::new();
    if utilization < 0.3 && bytes >= 4096 {
        suggestions.push("Consider wider vector loads (float4/float8)".into());
    }
    if utilization < 0.1 {
        suggestions.push("Access pattern may be strided or random — try coalescing".into());
    }
    if bytes < CACHE_LINE_64 {
        suggestions.push("Transfer smaller than a cache line — batch with neighbors".into());
    }

    BandwidthStats {
        measured_gbps,
        peak_gbps,
        bytes_transferred: bytes,
        elapsed_ns,
        bottleneck,
        suggestions,
    }
}

// ───────────────────────────────────────────────────────────────────────────
// AccessPatternDetector
// ───────────────────────────────────────────────────────────────────────────

/// Detects the memory access pattern from a sequence of byte addresses.
pub struct AccessPatternDetector;

impl AccessPatternDetector {
    /// Analyze a sequence of accessed byte addresses and return the dominant
    /// access pattern.
    pub fn detect(addresses: &[usize]) -> MemoryAccessPattern {
        if addresses.len() < 2 {
            return MemoryAccessPattern::Sequential;
        }

        // Compute strides between consecutive accesses.
        let mut strides = Vec::with_capacity(addresses.len() - 1);
        for pair in addresses.windows(2) {
            // Use wrapping subtraction to handle any order.
            strides.push(pair[1].wrapping_sub(pair[0]));
        }

        // Check if all strides are equal.
        let first = strides[0];
        let uniform = strides.iter().all(|&s| s == first);

        if !uniform {
            // Check whether strides vary wildly → random.
            let min = *strides.iter().min().unwrap();
            let max = *strides.iter().max().unwrap();
            if max.saturating_sub(min) > CACHE_LINE_128 {
                return MemoryAccessPattern::Random;
            }
            // Non-uniform but small variation → treat as strided with average.
            let avg = strides.iter().sum::<usize>() / strides.len();
            let stride_elements = avg / FLOAT_SIZE;
            return if stride_elements <= 1 {
                MemoryAccessPattern::Sequential
            } else {
                MemoryAccessPattern::Strided(stride_elements)
            };
        }

        let stride_elements = first / FLOAT_SIZE;
        if stride_elements <= 1 {
            MemoryAccessPattern::Sequential
        } else {
            MemoryAccessPattern::Strided(stride_elements)
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// OpenCL kernel source — bandwidth-optimized patterns
// ───────────────────────────────────────────────────────────────────────────

/// Embedded OpenCL C kernel demonstrating bandwidth-optimization techniques.
pub const BANDWIDTH_OPT_CL: &str = r#"
// ──────────────────────────────────────────────────────────────────────
// bandwidth_opt.cl — A770 memory bandwidth optimization patterns
// ──────────────────────────────────────────────────────────────────────

// 1. Coalesced vector copy using float4 loads/stores.
__kernel void coalesced_copy_float4(
    __global const float4* restrict src,
    __global       float4* restrict dst,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        dst[gid] = src[gid];
    }
}

// 2. SLM-tiled matrix multiply (TILE×TILE tiles in local memory).
#ifndef TILE
#define TILE 16
#endif

__kernel void matmul_slm_tiled(
    __global const float* restrict A,
    __global const float* restrict B,
    __global       float* restrict C,
    const uint M, const uint K, const uint N)
{
    __local float tileA[TILE][TILE];
    __local float tileB[TILE][TILE];

    uint row = get_local_id(1) + get_group_id(1) * TILE;
    uint col = get_local_id(0) + get_group_id(0) * TILE;
    float acc = 0.0f;

    for (uint tk = 0; tk < K; tk += TILE) {
        // Cooperative load into SLM.
        uint lx = get_local_id(0);
        uint ly = get_local_id(1);
        tileA[ly][lx] = (row < M && (tk + lx) < K) ? A[row * K + tk + lx] : 0.0f;
        tileB[ly][lx] = ((tk + ly) < K && col < N) ? B[(tk + ly) * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint p = 0; p < TILE; ++p)
            acc += tileA[ly][p] * tileB[p][lx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// 3. Prefetch-assisted reduction.
__kernel void prefetch_reduce(
    __global const float* restrict data,
    __global       float* restrict partial_sums,
    const uint count)
{
    uint gid = get_global_id(0);
    uint stride = get_global_size(0);
    float acc = 0.0f;

    // Issue first prefetch.
    if (gid + stride < count)
        prefetch(&data[gid + stride], 16);

    for (uint i = gid; i < count; i += stride) {
        acc += data[i];
        // Prefetch next chunk.
        if (i + stride < count)
            prefetch(&data[i + stride], 16);
    }
    partial_sums[get_global_id(0)] = acc;
}

// 4. Vectorized scale (float8).
__kernel void scale_float8(
    __global float8* restrict buf,
    const float factor,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        buf[gid] *= factor;
    }
}

// 5. Coalesced strided gather (subgroup-cooperative).
__kernel void coalesced_gather(
    __global const float* restrict src,
    __global       float* restrict dst,
    __global const uint*  restrict indices,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        dst[gid] = src[indices[gid]];
    }
}
"#;

/// Return the embedded bandwidth-optimization OpenCL kernel source.
pub fn kernel_source() -> &'static str {
    BANDWIDTH_OPT_CL
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // ── MemoryAccessPattern ────────────────────────────────────────────

    #[test]
    fn test_sequential_efficiency() {
        let e = MemoryAccessPattern::Sequential.efficiency();
        assert!((e - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_random_efficiency() {
        let e = MemoryAccessPattern::Random.efficiency();
        assert!((e - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_stride1_is_sequential_efficiency() {
        let e = MemoryAccessPattern::Strided(1).efficiency();
        assert!((e - 0.95).abs() < 1e-9, "stride-1 should equal sequential");
    }

    #[test]
    fn test_stride0_is_sequential_efficiency() {
        let e = MemoryAccessPattern::Strided(0).efficiency();
        assert!((e - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_large_stride_low_efficiency() {
        let e = MemoryAccessPattern::Strided(16).efficiency();
        assert!(e < 0.5, "stride-16 should have low efficiency, got {e}");
    }

    #[test]
    fn test_efficiency_monotonically_decreases_with_stride() {
        let mut prev = 1.0;
        for stride in [1, 2, 4, 8, 16] {
            let e = MemoryAccessPattern::Strided(stride).efficiency();
            assert!(e <= prev + 1e-9, "efficiency should not increase: stride={stride}");
            prev = e;
        }
    }

    #[test]
    fn test_pattern_display() {
        assert_eq!(format!("{}", MemoryAccessPattern::Sequential), "sequential");
        assert_eq!(format!("{}", MemoryAccessPattern::Strided(4)), "strided(4)");
        assert_eq!(format!("{}", MemoryAccessPattern::Random), "random");
    }

    // ── VectorWidth ────────────────────────────────────────────────────

    #[test]
    fn test_vector_width_elements() {
        assert_eq!(VectorWidth::Float1.elements(), 1);
        assert_eq!(VectorWidth::Float2.elements(), 2);
        assert_eq!(VectorWidth::Float4.elements(), 4);
        assert_eq!(VectorWidth::Float8.elements(), 8);
        assert_eq!(VectorWidth::Float16.elements(), 16);
    }

    #[test]
    fn test_vector_width_byte_size() {
        assert_eq!(VectorWidth::Float4.byte_size(), 16);
        assert_eq!(VectorWidth::Float16.byte_size(), 64);
    }

    #[test]
    fn test_vector_width_all_ordered() {
        let all = VectorWidth::all();
        assert_eq!(all.len(), 5);
        for pair in all.windows(2) {
            assert!(pair[0].elements() < pair[1].elements());
        }
    }

    #[test]
    fn test_best_for_selects_widest_divisor() {
        assert_eq!(VectorWidth::best_for(16), VectorWidth::Float16);
        assert_eq!(VectorWidth::best_for(8), VectorWidth::Float8);
        assert_eq!(VectorWidth::best_for(12), VectorWidth::Float4);
        assert_eq!(VectorWidth::best_for(6), VectorWidth::Float2);
        assert_eq!(VectorWidth::best_for(3), VectorWidth::Float1);
    }

    #[test]
    fn test_best_for_one_element() {
        assert_eq!(VectorWidth::best_for(1), VectorWidth::Float1);
    }

    #[test]
    fn test_vector_width_display() {
        assert_eq!(format!("{}", VectorWidth::Float4), "float4");
        assert_eq!(format!("{}", VectorWidth::Float1), "float");
    }

    // ── CoalescedAccess ────────────────────────────────────────────────

    #[test]
    fn test_coalesced_within_one_line() {
        let ca = CoalescedAccess::default();
        // 16 floats = 64 bytes = exactly one 64-byte cache line.
        assert!(ca.is_coalesced(0, 16));
    }

    #[test]
    fn test_not_coalesced_across_lines() {
        let ca = CoalescedAccess::default();
        // 17 floats starting at 0 → spans 2 cache lines.
        assert!(!ca.is_coalesced(0, 17));
    }

    #[test]
    fn test_coalesced_empty() {
        let ca = CoalescedAccess::default();
        assert!(ca.is_coalesced(999, 0));
    }

    #[test]
    fn test_transactions_needed_aligned() {
        let ca = CoalescedAccess::default();
        assert_eq!(ca.transactions_needed(0, 16), 1); // 64 bytes, aligned
        assert_eq!(ca.transactions_needed(0, 32), 2); // 128 bytes
    }

    #[test]
    fn test_transactions_needed_unaligned() {
        let ca = CoalescedAccess::default();
        // 1 float at offset 60 bytes: spans bytes [60..64) — still in line 0.
        assert_eq!(ca.transactions_needed(60, 1), 1);
        // 2 floats at offset 60: bytes [60..68) → lines 0 and 1.
        assert_eq!(ca.transactions_needed(60, 2), 2);
    }

    #[test]
    fn test_transactions_zero_elements() {
        let ca = CoalescedAccess::default();
        assert_eq!(ca.transactions_needed(0, 0), 0);
    }

    #[test]
    fn test_strided_transactions_stride1() {
        let ca = CoalescedAccess::default();
        // stride=1 is sequential.
        let t_strided = ca.strided_transactions(0, 16, 1);
        let t_seq = ca.transactions_needed(0, 16);
        assert_eq!(t_strided, t_seq);
    }

    #[test]
    fn test_strided_transactions_large_stride() {
        let ca = CoalescedAccess::default();
        // stride=16 elements = 64 bytes = 1 cache line per access.
        let t = ca.strided_transactions(0, 4, 16);
        assert_eq!(t, 4, "each access hits a different line");
    }

    #[test]
    fn test_strided_transactions_zero() {
        let ca = CoalescedAccess::default();
        assert_eq!(ca.strided_transactions(0, 0, 4), 0);
        assert_eq!(ca.strided_transactions(0, 4, 0), 0);
    }

    #[test]
    fn test_coalesced_read_copies_data() {
        let ca = CoalescedAccess::default();
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0; 4];
        ca.coalesced_read(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_strided_read_basic() {
        let ca = CoalescedAccess::default();
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0; 4];
        ca.strided_read(&src, &mut dst, 4);
        assert_eq!(dst, vec![0.0, 4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_strided_read_out_of_bounds_zero_fills() {
        let ca = CoalescedAccess::default();
        let src = vec![1.0, 2.0];
        let mut dst = vec![-1.0; 4];
        ca.strided_read(&src, &mut dst, 1);
        assert_eq!(dst, vec![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_strided_read_zero_stride_noop() {
        let ca = CoalescedAccess::default();
        let src = vec![1.0, 2.0];
        let mut dst = vec![-1.0; 2];
        ca.strided_read(&src, &mut dst, 0);
        assert_eq!(dst, vec![-1.0, -1.0]);
    }

    // ── VectorizedLoad ─────────────────────────────────────────────────

    #[test]
    fn test_vectorized_load_float4_exact() {
        let vl = VectorizedLoad::new(VectorWidth::Float4);
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut dst = vec![0.0; 8];
        vl.load(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_vectorized_load_float2_with_tail() {
        let vl = VectorizedLoad::new(VectorWidth::Float2);
        let src = vec![1.0, 2.0, 3.0]; // 1 full float2 + 1 tail
        let mut dst = vec![0.0; 3];
        vl.load(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_vectorized_load_float8() {
        let vl = VectorizedLoad::new(VectorWidth::Float8);
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0; 16];
        vl.load(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_vectorized_load_float16() {
        let vl = VectorizedLoad::new(VectorWidth::Float16);
        let src: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut dst = vec![0.0; 32];
        vl.load(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_vectorized_load_float1_is_scalar() {
        let vl = VectorizedLoad::new(VectorWidth::Float1);
        let src = vec![42.0, 99.0];
        let mut dst = vec![0.0; 2];
        vl.load(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_vectorized_load_empty() {
        let vl = VectorizedLoad::new(VectorWidth::Float4);
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        vl.load(&src, &mut dst);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_vectorized_load_dst_smaller_than_src() {
        let vl = VectorizedLoad::new(VectorWidth::Float4);
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut dst = vec![0.0; 5]; // only 5 elements, not aligned to float4
        vl.load(&src, &mut dst);
        assert_eq!(dst, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vectorized_scale() {
        let vl = VectorizedLoad::new(VectorWidth::Float4);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        vl.scale(&mut data, 2.0);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_vectorized_scale_empty() {
        let vl = VectorizedLoad::new(VectorWidth::Float4);
        let mut data: Vec<f32> = vec![];
        vl.scale(&mut data, 3.0);
        assert!(data.is_empty());
    }

    #[test]
    fn test_vectorized_fma() {
        let vl = VectorizedLoad::new(VectorWidth::Float4);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let c = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let mut dst = vec![0.0; 5];
        vl.fma(&a, &b, &c, &mut dst);
        assert_eq!(dst, vec![2.5, 6.5, 12.5, 20.5, 30.5]);
    }

    #[test]
    fn test_vectorized_fma_empty() {
        let vl = VectorizedLoad::new(VectorWidth::Float2);
        let mut dst: Vec<f32> = vec![];
        vl.fma(&[], &[], &[], &mut dst);
        assert!(dst.is_empty());
    }

    // ── SharedMemoryTiler ──────────────────────────────────────────────

    #[test]
    fn test_max_tile_size_default() {
        let t = SharedMemoryTiler::a770_default();
        let max = t.max_tile_size();
        // 64KB / (2 * 4) = 8192 elements per tile; sqrt(8192) ≈ 90
        assert!(max >= 64 && max <= 128, "max_tile={max}");
    }

    #[test]
    fn test_tile_count_exact() {
        let t = SharedMemoryTiler::new(16, A770_SLM_SIZE);
        assert_eq!(t.tile_count(32, 48), (2, 3));
    }

    #[test]
    fn test_tile_count_non_exact() {
        let t = SharedMemoryTiler::new(16, A770_SLM_SIZE);
        assert_eq!(t.tile_count(17, 33), (2, 3));
    }

    #[test]
    fn test_global_load_comparison_tiled_fewer() {
        let t = SharedMemoryTiler::new(16, A770_SLM_SIZE);
        let (tiled, naive) = t.global_load_comparison(64, 64, 64);
        assert!(tiled < naive, "tiled={tiled} should be < naive={naive}");
    }

    #[test]
    fn test_global_load_comparison_small_matrix() {
        let t = SharedMemoryTiler::new(16, A770_SLM_SIZE);
        let (tiled, naive) = t.global_load_comparison(4, 4, 4);
        // Small matrix may not benefit, but should still compute.
        assert!(tiled > 0 && naive > 0);
    }

    #[test]
    fn test_tiled_matmul_identity() {
        let t = SharedMemoryTiler::new(2, A770_SLM_SIZE);
        // 2×2 identity multiply.
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        t.tiled_matmul(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_tiled_matmul_3x3() {
        let t = SharedMemoryTiler::new(2, A770_SLM_SIZE);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ];
        let mut c = vec![0.0; 9];
        t.tiled_matmul(&a, &b, &mut c, 3, 3, 3);

        // Expected: standard matmul result.
        let expected = vec![30.0, 24.0, 18.0, 84.0, 69.0, 54.0, 138.0, 114.0, 90.0];
        for (i, (&got, &exp)) in c.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-4, "c[{i}]: got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_tiled_matmul_non_square() {
        let t = SharedMemoryTiler::new(2, A770_SLM_SIZE);
        // A: 2×3, B: 3×2 → C: 2×2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0; 4];
        t.tiled_matmul(&a, &b, &mut c, 2, 3, 2);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_tiled_matmul_tile_larger_than_matrix() {
        let t = SharedMemoryTiler::new(32, A770_SLM_SIZE);
        let a = vec![2.0, 0.0, 0.0, 3.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let mut c = vec![0.0; 4];
        t.tiled_matmul(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![2.0, 2.0, 3.0, 3.0]);
    }

    // ── PrefetchScheduler ──────────────────────────────────────────────

    #[test]
    fn test_prefetch_count() {
        let ps = PrefetchScheduler::new(PrefetchDistance::Far, 256);
        assert_eq!(ps.prefetch_count(1024), 4);
        assert_eq!(ps.prefetch_count(0), 0);
        assert_eq!(ps.prefetch_count(1), 1);
        assert_eq!(ps.prefetch_count(257), 2);
    }

    #[test]
    fn test_prefetch_distance_bytes() {
        assert_eq!(PrefetchDistance::Near.bytes(), 64);
        assert_eq!(PrefetchDistance::Medium.bytes(), 256);
        assert_eq!(PrefetchDistance::Far.bytes(), 512);
    }

    #[test]
    fn test_prefetch_distance_elements() {
        assert_eq!(PrefetchDistance::Near.elements(), 16);
        assert_eq!(PrefetchDistance::Far.elements(), 128);
    }

    #[test]
    fn test_prefetched_dot_correctness() {
        let ps = PrefetchScheduler::new(PrefetchDistance::Far, 4);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let result = ps.prefetched_dot(&data, &weights);
        assert!((result - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_prefetched_dot_empty() {
        let ps = PrefetchScheduler::a770_default();
        assert!((ps.prefetched_dot(&[], &[]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_prefetched_dot_mismatched_lengths() {
        let ps = PrefetchScheduler::new(PrefetchDistance::Near, 2);
        let data = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0];
        let result = ps.prefetched_dot(&data, &weights);
        assert!((result - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_prefetch_zero_compute_chunk() {
        let ps = PrefetchScheduler::new(PrefetchDistance::Near, 0);
        assert_eq!(ps.compute_chunk, 1, "zero chunk should be clamped to 1");
    }

    #[test]
    fn test_opencl_prefetch_call() {
        let ps = PrefetchScheduler::a770_default();
        let call = ps.opencl_prefetch_call("data", "i + 128");
        assert!(call.contains("prefetch(data + i + 128,"), "got: {call}");
    }

    // ── CacheLineAligner ───────────────────────────────────────────────

    #[test]
    fn test_align_size_64() {
        let a = CacheLineAligner::cl64();
        assert_eq!(a.align_size(0), 0);
        assert_eq!(a.align_size(1), 64);
        assert_eq!(a.align_size(64), 64);
        assert_eq!(a.align_size(65), 128);
    }

    #[test]
    fn test_align_size_128() {
        let a = CacheLineAligner::cl128();
        assert_eq!(a.align_size(100), 128);
        assert_eq!(a.align_size(128), 128);
        assert_eq!(a.align_size(129), 256);
    }

    #[test]
    fn test_padding() {
        let a = CacheLineAligner::cl64();
        assert_eq!(a.padding(64), 0);
        assert_eq!(a.padding(60), 4);
        assert_eq!(a.padding(1), 63);
    }

    #[test]
    fn test_is_aligned() {
        let a = CacheLineAligner::cl64();
        assert!(a.is_aligned(0));
        assert!(a.is_aligned(64));
        assert!(a.is_aligned(128));
        assert!(!a.is_aligned(1));
        assert!(!a.is_aligned(63));
    }

    #[test]
    fn test_align_element_count() {
        let a = CacheLineAligner::cl64();
        // 16 f32 = 64 bytes → already aligned.
        assert_eq!(a.align_element_count(16), 16);
        // 15 f32 = 60 bytes → 64 bytes = 16 elements.
        assert_eq!(a.align_element_count(15), 16);
        // 17 f32 = 68 bytes → 128 bytes = 32 elements.
        assert_eq!(a.align_element_count(17), 32);
    }

    #[test]
    fn test_alloc_aligned() {
        let a = CacheLineAligner::cl64();
        let buf = a.alloc_aligned(10);
        assert!(buf.len() >= 10);
        assert_eq!((buf.len() * FLOAT_SIZE) % 64, 0);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn test_aligner_non_power_of_two_panics() {
        CacheLineAligner::new(48);
    }

    #[test]
    fn test_aligned_size_geq_original() {
        // Property: aligned size is always ≥ original.
        let a = CacheLineAligner::cl64();
        for size in 0..300 {
            assert!(a.align_size(size) >= size, "size={size}");
        }
    }

    #[test]
    fn test_aligned_size_is_multiple() {
        // Property: aligned size is a multiple of alignment.
        let a = CacheLineAligner::cl128();
        for size in 0..300 {
            let aligned = a.align_size(size);
            if aligned > 0 {
                assert_eq!(aligned % 128, 0, "size={size} aligned={aligned}");
            }
        }
    }

    // ── BandwidthEstimator ─────────────────────────────────────────────

    #[test]
    fn test_estimate_sequential() {
        let be = BandwidthEstimator::default();
        let bw = be.estimate(MemoryAccessPattern::Sequential);
        assert!((bw - 560.0 * 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_random() {
        let be = BandwidthEstimator::default();
        let bw = be.estimate(MemoryAccessPattern::Random);
        assert!((bw - 560.0 * 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_time_ns() {
        let be = BandwidthEstimator::new(100.0); // 100 GB/s
        let t = be.estimate_time_ns(1000, MemoryAccessPattern::Sequential);
        // effective = 100 * 0.95 = 95 GB/s = 95 bytes/ns
        // time = 1000 / 95 ≈ 10.526
        assert!((t - 1000.0 / 95.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_time_ns_zero_bandwidth() {
        let be = BandwidthEstimator::new(0.0);
        let t = be.estimate_time_ns(100, MemoryAccessPattern::Sequential);
        assert!(t.is_infinite());
    }

    #[test]
    fn test_estimate_vectorized_uplift() {
        let be = BandwidthEstimator::default();
        let bw1 = be.estimate_vectorized(MemoryAccessPattern::Sequential, VectorWidth::Float1);
        let bw4 = be.estimate_vectorized(MemoryAccessPattern::Sequential, VectorWidth::Float4);
        assert!(bw4 > bw1, "float4 should be faster than float1");
    }

    #[test]
    fn test_estimate_vectorized_capped_at_peak() {
        let be = BandwidthEstimator::new(10.0);
        let bw = be.estimate_vectorized(MemoryAccessPattern::Sequential, VectorWidth::Float16);
        assert!(bw <= 10.0, "should not exceed peak");
    }

    // ── BandwidthStats / measure_bandwidth ─────────────────────────────

    #[test]
    fn test_measure_bandwidth_basic() {
        // 1 GB transferred in 1 second = 1 GB/s
        let stats = measure_bandwidth(1_000_000_000, 1_000_000_000, 100.0);
        assert!((stats.measured_gbps - 1.0).abs() < 1e-6);
        assert_eq!(stats.bytes_transferred, 1_000_000_000);
    }

    #[test]
    fn test_measure_bandwidth_zero_time() {
        let stats = measure_bandwidth(1000, 0, 100.0);
        assert!((stats.measured_gbps - 0.0).abs() < 1e-9);
        assert_eq!(stats.bottleneck, Bottleneck::Balanced);
    }

    #[test]
    fn test_utilization() {
        let stats = measure_bandwidth(560_000_000, 1_000_000_000, 560.0);
        // 0.56 GB/s / 560 GB/s = 0.001 → very low
        assert!(stats.utilization() < 1.0);
        assert!(stats.utilization() >= 0.0);
    }

    #[test]
    fn test_utilization_pct() {
        let mut stats = measure_bandwidth(100, 1, 100.0);
        stats.measured_gbps = 50.0;
        stats.peak_gbps = 100.0;
        assert!((stats.utilization_pct() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_bottleneck_display() {
        assert_eq!(format!("{}", Bottleneck::MemoryBound), "memory-bound");
        assert_eq!(format!("{}", Bottleneck::ComputeBound), "compute-bound");
        assert_eq!(format!("{}", Bottleneck::LatencyBound), "latency-bound");
        assert_eq!(format!("{}", Bottleneck::Balanced), "balanced");
    }

    #[test]
    fn test_bandwidth_stats_display() {
        let stats = BandwidthStats {
            measured_gbps: 100.0,
            peak_gbps: 560.0,
            bytes_transferred: 1000,
            elapsed_ns: 10,
            bottleneck: Bottleneck::MemoryBound,
            suggestions: vec![],
        };
        let s = format!("{stats}");
        assert!(s.contains("100.0 GB/s"));
        assert!(s.contains("memory-bound"));
    }

    #[test]
    fn test_suggestions_for_low_utilization() {
        let stats = measure_bandwidth(100_000, 1_000_000_000, 560.0);
        // Very low bw → should produce suggestions.
        assert!(!stats.suggestions.is_empty());
    }

    #[test]
    fn test_suggestions_tiny_transfer() {
        let stats = measure_bandwidth(32, 100, 560.0);
        let has_tiny_msg = stats.suggestions.iter().any(|s| s.contains("cache line"));
        assert!(has_tiny_msg, "expected cache-line suggestion for 32-byte transfer");
    }

    #[test]
    fn test_latency_bound_small_transfer() {
        let stats = measure_bandwidth(100, 1_000_000_000, 560.0);
        assert_eq!(stats.bottleneck, Bottleneck::LatencyBound);
    }

    // ── AccessPatternDetector ──────────────────────────────────────────

    #[test]
    fn test_detect_sequential() {
        let addrs: Vec<usize> = (0..16).map(|i| i * 4).collect();
        assert_eq!(AccessPatternDetector::detect(&addrs), MemoryAccessPattern::Sequential);
    }

    #[test]
    fn test_detect_strided() {
        // stride = 8 elements = 32 bytes
        let addrs: Vec<usize> = (0..8).map(|i| i * 32).collect();
        assert_eq!(AccessPatternDetector::detect(&addrs), MemoryAccessPattern::Strided(8));
    }

    #[test]
    fn test_detect_random() {
        let addrs = vec![0, 10000, 200, 50000, 3, 99999];
        assert_eq!(AccessPatternDetector::detect(&addrs), MemoryAccessPattern::Random);
    }

    #[test]
    fn test_detect_single_address() {
        assert_eq!(AccessPatternDetector::detect(&[42]), MemoryAccessPattern::Sequential);
    }

    #[test]
    fn test_detect_empty() {
        assert_eq!(AccessPatternDetector::detect(&[]), MemoryAccessPattern::Sequential);
    }

    // ── Kernel source ──────────────────────────────────────────────────

    #[test]
    fn test_kernel_source_not_empty() {
        assert!(!kernel_source().is_empty());
    }

    #[test]
    fn test_kernel_source_contains_entry_points() {
        let src = kernel_source();
        assert!(src.contains("coalesced_copy_float4"));
        assert!(src.contains("matmul_slm_tiled"));
        assert!(src.contains("prefetch_reduce"));
        assert!(src.contains("scale_float8"));
        assert!(src.contains("coalesced_gather"));
    }

    #[test]
    fn test_kernel_source_has_slm_tile() {
        let src = kernel_source();
        assert!(src.contains("__local float tileA"));
        assert!(src.contains("__local float tileB"));
    }

    #[test]
    fn test_kernel_source_uses_prefetch() {
        assert!(kernel_source().contains("prefetch("));
    }

    #[test]
    fn test_kernel_source_uses_restrict() {
        assert!(kernel_source().contains("restrict"));
    }

    // ── Coalesced vs Strided comparison ────────────────────────────────

    #[test]
    fn test_coalesced_fewer_transactions_than_strided() {
        let ca = CoalescedAccess::default();
        let n = 16;
        let seq = ca.transactions_needed(0, n);
        let strided = ca.strided_transactions(0, n, 4);
        assert!(
            seq <= strided,
            "sequential ({seq}) should use ≤ transactions vs strided ({strided})"
        );
    }

    #[test]
    fn test_coalesced_timing_reference() {
        // Just verify the CPU reference paths run without error.
        let ca = CoalescedAccess::default();
        let n = 1024;
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let mut dst_seq = vec![0.0; n];
        let start = Instant::now();
        ca.coalesced_read(&src, &mut dst_seq);
        let _elapsed_seq = start.elapsed();

        let mut dst_stride = vec![0.0; n / 4];
        let start = Instant::now();
        ca.strided_read(&src, &mut dst_stride, 4);
        let _elapsed_stride = start.elapsed();

        // Sequential result should match source.
        assert_eq!(dst_seq, src);
        // Strided should pick every 4th element.
        for (i, &v) in dst_stride.iter().enumerate() {
            assert_eq!(v, (i * 4) as f32);
        }
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_unaligned_buffer_alignment() {
        let a = CacheLineAligner::cl64();
        // 3 bytes → should align to 64.
        assert_eq!(a.align_size(3), 64);
        // Very large value.
        let big = 1_000_000_001;
        let aligned = a.align_size(big);
        assert!(aligned >= big);
        assert_eq!(aligned % 64, 0);
    }

    #[test]
    fn test_tiny_transfer_bandwidth_stats() {
        let stats = measure_bandwidth(1, 1, A770_PEAK_BANDWIDTH_GBPS);
        assert!(stats.measured_gbps > 0.0);
        assert!(stats.utilization() < 0.01);
    }

    #[test]
    fn test_custom_coalesced_access() {
        let ca = CoalescedAccess::new(32, 128);
        assert!(ca.is_coalesced(0, 32)); // 32 floats = 128 bytes = 1 line
        assert!(!ca.is_coalesced(0, 33)); // 33 floats > 128 bytes
    }

    #[test]
    fn test_bandwidth_estimator_custom_peak() {
        let be = BandwidthEstimator::new(1000.0);
        let bw = be.estimate(MemoryAccessPattern::Sequential);
        assert!((bw - 950.0).abs() < 1e-6);
    }
}
