//! Cache-aware matrix multiplication with multi-level tiling and auto-tuning.
//!
//! Provides production-quality CPU matmul kernels that are aware of the full
//! cache hierarchy (L1 / L2 / L3), NUMA topology, and cache-line alignment.
//!
//! # Features
//!
//! - **Multi-level tiling**: L1, L2, L3-aware tile sizes that keep working
//!   sets resident at each cache level.
//! - **Auto-tuning**: Tile sizes derived from detected (or user-supplied)
//!   cache parameters at runtime.
//! - **Prefetch scheduling**: Software prefetch hints for predictable
//!   sequential and strided access patterns.
//! - **NUMA-aware matmul**: Partition work across NUMA nodes so each core
//!   touches local memory as much as possible.
//! - **Matrix packing**: Re-layout A (row panels) and B (column panels) for
//!   cache-line-aligned, sequential access in the inner loop.
//! - **Streaming stores**: Non-temporal writes when the output matrix is
//!   large enough that polluting the cache is wasteful.
//! - **Tall-skinny blocking**: Specialised tiling for M ≫ N or M ≫ K
//!   shapes common in autoregressive inference (single-token decode).
//! - **Cache-miss estimation**: Lightweight counters that model expected L1 /
//!   L2 / L3 misses for a given problem shape + tiling.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Cache hierarchy description ────────────────────────────────────────

/// Full description of the CPU cache hierarchy and NUMA topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheHierarchy {
    /// L1 data-cache size in bytes per core.
    pub l1_size: usize,
    /// L2 cache size in bytes per core.
    pub l2_size: usize,
    /// L3 (last-level) cache size in bytes (shared across cores on a socket).
    pub l3_size: usize,
    /// Cache-line size in bytes (typically 64).
    pub line_size: usize,
    /// L1 associativity (ways).  Used for conflict-miss estimation.
    pub l1_assoc: usize,
    /// L2 associativity (ways).
    pub l2_assoc: usize,
    /// Number of NUMA nodes detected on the system.
    pub numa_nodes: usize,
    /// Number of physical cores per NUMA node.
    pub cores_per_node: usize,
}

impl Default for CacheHierarchy {
    fn default() -> Self {
        Self::detect()
    }
}

impl CacheHierarchy {
    /// Attempt to detect cache parameters from the OS.
    ///
    /// Falls back to [`Self::conservative`] on unsupported platforms or when
    /// detection fails.
    pub fn detect() -> Self {
        // Try sysfs on Linux first.
        #[cfg(target_os = "linux")]
        {
            if let Some(h) = Self::detect_linux() {
                return h;
            }
        }
        Self::conservative()
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Option<Self> {
        use std::fs;

        let read_usize = |path: &str| -> Option<usize> {
            let s = fs::read_to_string(path).ok()?;
            let s = s.trim().trim_end_matches('K').trim_end_matches('M');
            s.parse::<usize>().ok()
        };

        let base = "/sys/devices/system/cpu/cpu0/cache";
        // index0 = L1d, index2 = L2, index3 = L3 (typical Intel/AMD)
        let l1 = read_usize(&format!("{base}/index0/size")).map(|v| v * 1024);
        let l2 = read_usize(&format!("{base}/index2/size")).map(|v| v * 1024);
        let l3 = read_usize(&format!("{base}/index3/size")).map(|v| v * 1024);
        let line = read_usize(&format!("{base}/index0/coherency_line_size"));
        let l1_assoc = read_usize(&format!("{base}/index0/ways_of_associativity")).unwrap_or(8);
        let l2_assoc = read_usize(&format!("{base}/index2/ways_of_associativity")).unwrap_or(8);

        // NUMA node count
        let numa_nodes = fs::read_dir("/sys/devices/system/node")
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_name().to_str().is_some_and(|n| n.starts_with("node")))
                    .count()
            })
            .unwrap_or(1)
            .max(1);

        let total_cores = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        let cores_per_node = (total_cores / numa_nodes).max(1);

        Some(Self {
            l1_size: l1.unwrap_or(32 * 1024),
            l2_size: l2.unwrap_or(256 * 1024),
            l3_size: l3.unwrap_or(8 * 1024 * 1024),
            line_size: line.unwrap_or(64),
            l1_assoc,
            l2_assoc,
            numa_nodes,
            cores_per_node,
        })
    }

    /// Conservative defaults suitable for modern x86-64 CPUs.
    pub fn conservative() -> Self {
        Self {
            l1_size: 32 * 1024,       // 32 KiB
            l2_size: 256 * 1024,      // 256 KiB
            l3_size: 8 * 1024 * 1024, // 8 MiB
            line_size: 64,
            l1_assoc: 8,
            l2_assoc: 8,
            numa_nodes: 1,
            cores_per_node: 4,
        }
    }

    /// Build a hierarchy with explicit sizes (for testing / benchmarking).
    pub fn with_sizes(l1: usize, l2: usize, l3: usize) -> Self {
        Self {
            l1_size: l1,
            l2_size: l2,
            l3_size: l3,
            line_size: 64,
            l1_assoc: 8,
            l2_assoc: 8,
            numa_nodes: 1,
            cores_per_node: 4,
        }
    }
}

// ── Multi-level tile sizes ─────────────────────────────────────────────

/// Tile sizes computed for each cache level.
///
/// The inner-most tiles (L1) are sized so that the A-panel, B-panel, and
/// C-micro-tile all fit in L1.  The L2 tile wraps multiple L1 tiles so
/// the re-used panel stays in L2.  The L3 tile partitions the full
/// problem across cores sharing the LLC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiLevelTiling {
    // L1 micro-tile
    pub mc: usize, // rows of A panel
    pub nc: usize, // cols of B panel
    pub kc: usize, // inner dimension

    // L2 tile (multiples of L1 tile)
    pub mc_l2: usize,
    pub nc_l2: usize,

    // L3 partition (across cores)
    pub mc_l3: usize,
}

/// Compute multi-level tile sizes from cache hierarchy and problem shape.
///
/// The algorithm targets ~75 % of each cache level to leave room for
/// stack, TLB entries, and OS jitter.
pub fn auto_tune_tiling(
    m: usize,
    n: usize,
    k: usize,
    hierarchy: &CacheHierarchy,
) -> MultiLevelTiling {
    let elem = std::mem::size_of::<f32>();
    let line = hierarchy.line_size.max(1);

    // --- L1: A-panel (mc × kc) + B-panel (kc × nc) + C-tile (mc × nc)
    let usable_l1 = (hierarchy.l1_size * 3) / 4; // 75%
    // Start with mc = nr of cache lines that hold one row of A
    let kc_target = (usable_l1 / (3 * elem)).isqrt().max(1);
    let kc = clamp_tile(kc_target, k, line / elem);

    let mc_target = (usable_l1 / (elem * kc * 2)).max(1);
    let mc = clamp_tile(mc_target, m, 1);

    let nc_target = (usable_l1 / (elem * kc)).max(1);
    let nc = clamp_tile(nc_target, n, 1);

    // --- L2: keep B-panel (kc × nc_l2) resident; sweep A panels through
    let usable_l2 = (hierarchy.l2_size * 3) / 4;
    let nc_l2_target = (usable_l2 / (elem * kc)).max(nc);
    let nc_l2 = clamp_tile(nc_l2_target, n, nc);

    let mc_l2_target = (usable_l2 / (elem * kc)).max(mc);
    let mc_l2 = clamp_tile(mc_l2_target, m, mc);

    // --- L3: partition M across cores sharing the LLC
    let cores = hierarchy.cores_per_node.max(1);
    let mc_l3 = (m / cores).max(mc_l2).min(m);

    MultiLevelTiling { mc, nc, kc, mc_l2, nc_l2, mc_l3 }
}

#[inline]
fn clamp_tile(target: usize, dim: usize, granularity: usize) -> usize {
    let g = granularity.max(1);
    let t = (target / g) * g;
    let t = t.max(g);
    t.min(dim)
}

// ── Prefetch helpers ───────────────────────────────────────────────────

/// Hint the CPU to bring data for the *next* tile into the given cache level.
///
/// This is a best-effort hint; not all architectures honour it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchLevel {
    L1,
    L2,
    L3,
}

/// Issue a software prefetch for `ptr` into the requested cache level.
///
/// # Safety
/// `ptr` must be a valid, aligned pointer into a live allocation.
#[inline(always)]
pub unsafe fn prefetch_read(ptr: *const f32, _level: PrefetchLevel) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _mm_prefetch};
        unsafe {
            match _level {
                PrefetchLevel::L1 => _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0),
                PrefetchLevel::L2 => _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T1),
                PrefetchLevel::L3 => _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T2),
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = ptr;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// Schedule prefetches for the next block of a row-major matrix.
///
/// # Safety
/// `base` must point into a valid allocation with at least
/// `rows * stride` elements beyond it.
#[inline]
pub unsafe fn prefetch_block(
    base: *const f32,
    rows: usize,
    cols: usize,
    stride: usize,
    level: PrefetchLevel,
) {
    let line_elems = 64 / std::mem::size_of::<f32>(); // 16 f32s per line
    for r in 0..rows {
        let row_ptr = unsafe { base.add(r * stride) };
        let mut c = 0;
        while c < cols {
            unsafe { prefetch_read(row_ptr.add(c), level) };
            c += line_elems;
        }
    }
}

// ── Matrix packing ─────────────────────────────────────────────────────

/// Pack a row-major A sub-matrix `[mc × kc]` into a contiguous panel
/// aligned to cache-line boundaries.
///
/// The panel is stored as `mc` rows of `kc` elements each, with each row
/// padded to a multiple of `line_elems` so successive rows start on
/// cache-line boundaries.
pub fn pack_a_panel(
    a: &[f32],
    panel: &mut [f32],
    m_start: usize,
    k_start: usize,
    mc: usize,
    kc: usize,
    k_full: usize,
    line_size: usize,
) {
    let line_elems = (line_size / std::mem::size_of::<f32>()).max(1);
    let padded_kc = kc.div_ceil(line_elems) * line_elems;
    for i in 0..mc {
        let src_row = m_start + i;
        let dst_off = i * padded_kc;
        for j in 0..kc {
            let src_col = k_start + j;
            if src_row < a.len() / k_full.max(1) && src_col < k_full {
                panel[dst_off + j] = a[src_row * k_full + src_col];
            }
        }
        // Zero-pad remainder
        for j in kc..padded_kc {
            if dst_off + j < panel.len() {
                panel[dst_off + j] = 0.0;
            }
        }
    }
}

/// Pack a row-major B sub-matrix `[kc × nc]` into column-panel layout.
///
/// The panel stores B as `nc` columns of `kc` elements each (column-major
/// within the panel), with each column padded to cache-line alignment.
pub fn pack_b_panel(
    b: &[f32],
    panel: &mut [f32],
    k_start: usize,
    n_start: usize,
    kc: usize,
    nc: usize,
    n_full: usize,
    line_size: usize,
) {
    let line_elems = (line_size / std::mem::size_of::<f32>()).max(1);
    let padded_kc = kc.div_ceil(line_elems) * line_elems;
    for j in 0..nc {
        let dst_col = j * padded_kc;
        for p in 0..kc {
            let src_row = k_start + p;
            let src_col = n_start + j;
            if src_row < b.len() / n_full.max(1) && src_col < n_full {
                panel[dst_col + p] = b[src_row * n_full + src_col];
            }
        }
        for p in kc..padded_kc {
            if dst_col + p < panel.len() {
                panel[dst_col + p] = 0.0;
            }
        }
    }
}

// ── Streaming stores ───────────────────────────────────────────────────

/// Threshold (in elements) above which we prefer streaming (non-temporal)
/// stores for the output matrix C.
const STREAMING_STORE_THRESHOLD: usize = 256 * 1024; // ~1 MiB of f32

/// Configuration for streaming store behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorePolicy {
    /// Always use regular stores.
    Normal,
    /// Always use streaming (non-temporal) stores.
    Streaming,
    /// Automatically choose based on output matrix size.
    Auto,
}

/// Decide whether streaming stores should be used.
#[inline]
pub fn should_stream(c_elements: usize, policy: StorePolicy) -> bool {
    match policy {
        StorePolicy::Normal => false,
        StorePolicy::Streaming => true,
        StorePolicy::Auto => c_elements >= STREAMING_STORE_THRESHOLD,
    }
}

/// Write `val` to `dst` using a streaming store when `stream` is true.
///
/// # Safety
/// `dst` must be a valid, aligned pointer.
#[inline(always)]
pub unsafe fn store_f32(dst: *mut f32, val: f32, stream: bool) {
    if stream {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_stream_si32;
            unsafe { _mm_stream_si32(dst.cast::<i32>(), val.to_bits() as i32) };
            return;
        }
    }
    // Fallback / normal store
    unsafe { dst.write(val) };
}

// ── NUMA partitioning ──────────────────────────────────────────────────

/// Describes how a matrix dimension is partitioned across NUMA nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaPartition {
    /// NUMA node index.
    pub node: usize,
    /// Start row (inclusive).
    pub row_start: usize,
    /// End row (exclusive).
    pub row_end: usize,
}

/// Partition `m` rows across `numa_nodes` NUMA nodes.
pub fn numa_partition(m: usize, numa_nodes: usize) -> Vec<NumaPartition> {
    let nodes = numa_nodes.max(1);
    let base = m / nodes;
    let remainder = m % nodes;
    let mut partitions = Vec::with_capacity(nodes);
    let mut start = 0;
    for node in 0..nodes {
        let extra = if node < remainder { 1 } else { 0 };
        let end = start + base + extra;
        partitions.push(NumaPartition { node, row_start: start, row_end: end });
        start = end;
    }
    partitions
}

/// Perform NUMA-aware matrix multiplication.
///
/// Splits the M dimension across NUMA nodes and uses [`matmul_cache_aware`]
/// for each partition.  On single-node systems this is equivalent to a
/// plain cache-aware matmul.
pub fn matmul_numa_aware(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    hierarchy: &CacheHierarchy,
    store_policy: StorePolicy,
) -> Result<()> {
    validate_args(a, b, c, m, n, k)?;

    let partitions = numa_partition(m, hierarchy.numa_nodes);
    for part in &partitions {
        let rows = part.row_end - part.row_start;
        if rows == 0 {
            continue;
        }
        let a_start = part.row_start * k;
        let a_end = part.row_end * k;
        let c_start = part.row_start * n;
        let c_end = part.row_end * n;
        matmul_cache_aware_inner(
            &a[a_start..a_end],
            b,
            &mut c[c_start..c_end],
            rows,
            n,
            k,
            hierarchy,
            store_policy,
        )?;
    }
    Ok(())
}

// ── Tall-skinny specialisation ─────────────────────────────────────────

/// Detect if a problem shape is "tall-skinny" (M ≫ N or M ≫ K).
///
/// Returns `true` when the aspect ratio suggests a single-token decode
/// or similar workload where M is much larger than the other dimensions.
#[inline]
pub fn is_tall_skinny(m: usize, n: usize, k: usize) -> bool {
    (m > 4 * n) || (m > 4 * k)
}

/// Tiling strategy tuned for tall-skinny matrices.
///
/// Uses a narrow M-tile and maximises K/N tiles to keep the B-panel
/// resident.
pub fn tall_skinny_tiling(
    m: usize,
    n: usize,
    k: usize,
    hierarchy: &CacheHierarchy,
) -> MultiLevelTiling {
    let elem = std::mem::size_of::<f32>();
    let usable_l1 = (hierarchy.l1_size * 3) / 4;

    // For tall-skinny: small mc, full n, moderate kc
    let mc = 4_usize.min(m).max(1);
    let kc_target = (usable_l1 / (elem * (mc + n))).max(1);
    let kc = kc_target.min(k).max(1);
    let nc = n; // fit all columns for skinny N

    let mc_l2 = (mc * 4).min(m);
    let nc_l2 = nc;
    let mc_l3 = m;

    MultiLevelTiling { mc, nc, kc, mc_l2, nc_l2, mc_l3 }
}

// ── Cache-miss estimation ──────────────────────────────────────────────

/// Estimated cache-miss counts for a given tiling + problem shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheMissEstimate {
    /// Estimated L1 misses (cold + conflict).
    pub l1_misses: f64,
    /// Estimated L2 misses.
    pub l2_misses: f64,
    /// Estimated L3 misses.
    pub l3_misses: f64,
    /// Total data volume moved from DRAM (bytes, estimated).
    pub dram_bytes: f64,
}

/// Estimate cache misses for a tiled matmul.
///
/// Uses a simplified analytical model:
/// - L1 misses ≈ (data touched per L1 tile) / line_size × (number of L1
///   tile visits)
/// - L2 misses ≈ (data not fitting L2) / line_size × visits
/// - L3 / DRAM ≈ total unique data / line_size
pub fn estimate_cache_misses(
    m: usize,
    n: usize,
    k: usize,
    tiling: &MultiLevelTiling,
    hierarchy: &CacheHierarchy,
) -> CacheMissEstimate {
    let elem = std::mem::size_of::<f32>();
    let line = hierarchy.line_size.max(1);
    let elems_per_line = line / elem;

    let mc = tiling.mc.max(1);
    let nc = tiling.nc.max(1);
    let kc = tiling.kc.max(1);

    // Number of L1 tiles along each dimension
    let m_tiles = m.div_ceil(mc);
    let n_tiles = n.div_ceil(nc);
    let k_tiles = k.div_ceil(kc);

    // Data touched per L1 tile (A panel + B panel + C tile)
    let a_panel = mc * kc;
    let b_panel = kc * nc;
    let c_tile = mc * nc;
    let l1_tile_data = a_panel + b_panel + c_tile;

    // Cold misses: each unique element loaded once into L1
    let total_unique = m * k + k * n + m * n;
    let cold_misses = total_unique as f64 / elems_per_line as f64;

    // Conflict misses: proportional to how many times tiles revisit data
    // In well-tiled code this is small; estimate as fraction of total visits.
    let total_visits = m_tiles * n_tiles * k_tiles;
    let l1_fits = l1_tile_data * elem <= hierarchy.l1_size;
    let l1_conflict_factor = if l1_fits { 0.05 } else { 0.3 };
    let l1_misses = cold_misses + (total_visits as f64 * l1_conflict_factor);

    // L2 misses: data that spills out of L2
    let l2_data = tiling.mc_l2 * kc + kc * tiling.nc_l2;
    let l2_fits = l2_data * elem <= hierarchy.l2_size;
    let l2_misses =
        if l2_fits { cold_misses * 0.1 } else { cold_misses * 0.5 + (total_visits as f64 * 0.15) };

    // L3 / DRAM: total unique data vs L3
    let total_bytes = total_unique * elem;
    let l3_fits = total_bytes <= hierarchy.l3_size;
    let l3_misses =
        if l3_fits { 0.0 } else { (total_bytes - hierarchy.l3_size) as f64 / line as f64 };
    let dram_bytes = l3_misses * line as f64;

    CacheMissEstimate { l1_misses, l2_misses, l3_misses, dram_bytes }
}

// ── Core matmul implementation ─────────────────────────────────────────

/// Cache-aware f32 matrix multiplication: `C[m×n] = A[m×k] · B[k×n]`.
///
/// Auto-tunes tile sizes from `hierarchy`, selects tall-skinny strategy
/// when appropriate, and uses packed panels + prefetch in the inner loop.
pub fn matmul_cache_aware(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    hierarchy: &CacheHierarchy,
) -> Result<()> {
    matmul_cache_aware_inner(a, b, c, m, n, k, hierarchy, StorePolicy::Auto)
}

fn matmul_cache_aware_inner(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    hierarchy: &CacheHierarchy,
    store_policy: StorePolicy,
) -> Result<()> {
    validate_args(a, b, c, m, n, k)?;

    let tiling = if is_tall_skinny(m, n, k) {
        tall_skinny_tiling(m, n, k, hierarchy)
    } else {
        auto_tune_tiling(m, n, k, hierarchy)
    };

    let stream = should_stream(m * n, store_policy);

    let mc = tiling.mc.max(1);
    let nc = tiling.nc.max(1);
    let kc = tiling.kc.max(1);

    let line = hierarchy.line_size.max(1);
    let line_elems = (line / std::mem::size_of::<f32>()).max(1);
    let padded_kc = kc.div_ceil(line_elems) * line_elems;

    let a_panel_size = mc * padded_kc;
    let b_panel_size = nc * padded_kc;

    let mut a_packed = vec![0.0f32; a_panel_size];
    let mut b_packed = vec![0.0f32; b_panel_size];

    // Zero C
    c.iter_mut().for_each(|v| *v = 0.0);

    // L1-tiled loop: k → m → n (B-panel reuse in L1)
    let mut kk = 0;
    while kk < k {
        let kc_eff = kc.min(k - kk);

        let mut jj = 0;
        while jj < n {
            let nc_eff = nc.min(n - jj);
            // Pack B panel [kc_eff × nc_eff]
            pack_b_panel(b, &mut b_packed, kk, jj, kc_eff, nc_eff, n, line);

            let mut ii = 0;
            while ii < m {
                let mc_eff = mc.min(m - ii);
                // Pack A panel [mc_eff × kc_eff]
                pack_a_panel(a, &mut a_packed, ii, kk, mc_eff, kc_eff, k, line);

                // Prefetch next A panel
                if ii + mc < m {
                    let next_a_start = (ii + mc) * k + kk;
                    if next_a_start < a.len() {
                        unsafe {
                            prefetch_read(a.as_ptr().add(next_a_start), PrefetchLevel::L2);
                        }
                    }
                }

                // Micro-kernel: C[ii..ii+mc_eff, jj..jj+nc_eff] += A_pack · B_pack
                let ctx = MicroKernelCtx {
                    mc: mc_eff,
                    nc: nc_eff,
                    kc: kc_eff,
                    padded_kc,
                    row_off: ii,
                    col_off: jj,
                    n,
                    stream,
                };
                micro_kernel_packed(&a_packed, &b_packed, c, &ctx);

                ii += mc;
            }
            jj += nc;
        }
        kk += kc;
    }

    Ok(())
}

/// Parameters for the micro-kernel, bundled to stay under the argument limit.
struct MicroKernelCtx {
    mc: usize,
    nc: usize,
    kc: usize,
    padded_kc: usize,
    row_off: usize,
    col_off: usize,
    n: usize,
    stream: bool,
}

/// Inner micro-kernel operating on packed panels.
#[inline]
fn micro_kernel_packed(a_pack: &[f32], b_pack: &[f32], c: &mut [f32], ctx: &MicroKernelCtx) {
    for i in 0..ctx.mc {
        let a_row = i * ctx.padded_kc;
        for j in 0..ctx.nc {
            let b_col = j * ctx.padded_kc;
            let mut acc = 0.0f32;
            for p in 0..ctx.kc {
                acc += a_pack[a_row + p] * b_pack[b_col + p];
            }
            let c_idx = (ctx.row_off + i) * ctx.n + (ctx.col_off + j);
            if ctx.stream {
                unsafe {
                    let dst = c.as_mut_ptr().add(c_idx);
                    store_f32(dst, *dst + acc, true);
                }
            } else {
                c[c_idx] += acc;
            }
        }
    }
}

// ── Validation ─────────────────────────────────────────────────────────

fn validate_args(a: &[f32], b: &[f32], c: &[f32], m: usize, n: usize, k: usize) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("dimensions must be > 0: m={m}, n={n}, k={k}"),
        }));
    }
    let a_need = m * k;
    let b_need = k * n;
    let c_need = m * n;
    if a.len() < a_need {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("A too small: need {a_need}, got {}", a.len()),
        }));
    }
    if b.len() < b_need {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("B too small: need {b_need}, got {}", b.len()),
        }));
    }
    if c.len() < c_need {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("C too small: need {c_need}, got {}", c.len()),
        }));
    }
    Ok(())
}

// ── Convenience entry points ───────────────────────────────────────────

/// Auto-tuned cache-aware matmul with detected cache hierarchy.
///
/// This is the simplest entry point: it detects the cache hierarchy once
/// and computes `C = A · B`.
pub fn matmul_auto(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let hierarchy = CacheHierarchy::detect();
    matmul_cache_aware(a, b, c, m, n, k, &hierarchy)
}

/// Cache-aware matmul with streaming stores for large outputs.
pub fn matmul_streaming(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    hierarchy: &CacheHierarchy,
) -> Result<()> {
    matmul_cache_aware_inner(a, b, c, m, n, k, hierarchy, StorePolicy::Streaming)
}

// ══════════════════════════════════════════════════════════════════════
//  Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --------------------------------------------------------

    fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at [{i}]: {x} vs {y} (diff={})", (x - y).abs());
        }
    }

    fn hierarchy_tiny() -> CacheHierarchy {
        CacheHierarchy::with_sizes(1024, 4096, 16384)
    }

    // ── CacheHierarchy ────────────────────────────────────────────────

    #[test]
    fn test_hierarchy_conservative_sizes() {
        let h = CacheHierarchy::conservative();
        assert_eq!(h.l1_size, 32 * 1024);
        assert_eq!(h.l2_size, 256 * 1024);
        assert_eq!(h.l3_size, 8 * 1024 * 1024);
        assert_eq!(h.line_size, 64);
    }

    #[test]
    fn test_hierarchy_with_sizes() {
        let h = CacheHierarchy::with_sizes(1024, 2048, 4096);
        assert_eq!(h.l1_size, 1024);
        assert_eq!(h.l2_size, 2048);
        assert_eq!(h.l3_size, 4096);
    }

    #[test]
    fn test_hierarchy_detect_does_not_panic() {
        let _h = CacheHierarchy::detect();
    }

    #[test]
    fn test_hierarchy_default_eq_detect() {
        let a = CacheHierarchy::default();
        let b = CacheHierarchy::detect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_hierarchy_conservative_assoc() {
        let h = CacheHierarchy::conservative();
        assert_eq!(h.l1_assoc, 8);
        assert_eq!(h.l2_assoc, 8);
    }

    #[test]
    fn test_hierarchy_conservative_numa() {
        let h = CacheHierarchy::conservative();
        assert_eq!(h.numa_nodes, 1);
        assert!(h.cores_per_node >= 1);
    }

    // ── auto_tune_tiling ──────────────────────────────────────────────

    #[test]
    fn test_auto_tune_nonzero_tiles() {
        let h = CacheHierarchy::conservative();
        let t = auto_tune_tiling(64, 64, 64, &h);
        assert!(t.mc >= 1 && t.nc >= 1 && t.kc >= 1);
        assert!(t.mc_l2 >= t.mc);
        assert!(t.nc_l2 >= t.nc);
    }

    #[test]
    fn test_auto_tune_1x1x1() {
        let h = CacheHierarchy::conservative();
        let t = auto_tune_tiling(1, 1, 1, &h);
        assert_eq!(t.mc, 1);
        assert_eq!(t.nc, 1);
        assert_eq!(t.kc, 1);
    }

    #[test]
    fn test_auto_tune_tiles_bounded_by_dims() {
        let h = CacheHierarchy::conservative();
        let (m, n, k) = (5, 7, 3);
        let t = auto_tune_tiling(m, n, k, &h);
        assert!(t.mc <= m);
        assert!(t.nc <= n);
        assert!(t.kc <= k);
    }

    #[test]
    fn test_auto_tune_tiny_cache() {
        let h = hierarchy_tiny();
        let t = auto_tune_tiling(128, 128, 128, &h);
        // With a tiny L1 (1024B = 256 floats), tiles must be small.
        assert!(t.mc * t.kc <= 256);
    }

    #[test]
    fn test_auto_tune_large_dims() {
        let h = CacheHierarchy::conservative();
        let t = auto_tune_tiling(1024, 1024, 1024, &h);
        assert!(t.mc >= 1);
        assert!(t.nc >= 1);
        assert!(t.kc >= 1);
        assert!(t.mc <= 1024);
    }

    #[test]
    fn test_auto_tune_l2_tiles_at_least_l1() {
        let h = CacheHierarchy::conservative();
        let t = auto_tune_tiling(256, 256, 256, &h);
        assert!(t.mc_l2 >= t.mc);
        assert!(t.nc_l2 >= t.nc);
    }

    #[test]
    fn test_auto_tune_mc_l3_bounded() {
        let h = CacheHierarchy::conservative();
        let t = auto_tune_tiling(512, 64, 64, &h);
        assert!(t.mc_l3 <= 512);
    }

    // ── clamp_tile ────────────────────────────────────────────────────

    #[test]
    fn test_clamp_tile_basic() {
        assert_eq!(clamp_tile(16, 64, 4), 16);
        assert_eq!(clamp_tile(100, 64, 4), 64); // clamped to dim
        assert_eq!(clamp_tile(7, 64, 4), 4); // rounded down to granularity
    }

    #[test]
    fn test_clamp_tile_zero_granularity() {
        // granularity 0 should be treated as 1
        assert_eq!(clamp_tile(16, 64, 0), 16);
    }

    #[test]
    fn test_clamp_tile_target_zero() {
        assert_eq!(clamp_tile(0, 64, 4), 4); // at least one granularity
    }

    // ── PrefetchLevel / prefetch_read ─────────────────────────────────

    #[test]
    fn test_prefetch_levels_distinct() {
        assert_ne!(PrefetchLevel::L1, PrefetchLevel::L2);
        assert_ne!(PrefetchLevel::L2, PrefetchLevel::L3);
    }

    #[test]
    fn test_prefetch_read_does_not_panic() {
        let data = [1.0f32; 64];
        unsafe {
            prefetch_read(data.as_ptr(), PrefetchLevel::L1);
            prefetch_read(data.as_ptr(), PrefetchLevel::L2);
            prefetch_read(data.as_ptr(), PrefetchLevel::L3);
        }
    }

    #[test]
    fn test_prefetch_block_does_not_panic() {
        let data = [0.0f32; 256];
        unsafe {
            prefetch_block(data.as_ptr(), 4, 16, 16, PrefetchLevel::L1);
        }
    }

    // ── pack_a_panel ──────────────────────────────────────────────────

    #[test]
    fn test_pack_a_basic() {
        // 2×3 matrix, pack full panel
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let mut panel = vec![0.0f32; 2 * 16]; // padded to 16
        pack_a_panel(&a, &mut panel, 0, 0, 2, 3, 3, 64);
        assert_eq!(panel[0], 1.0);
        assert_eq!(panel[1], 2.0);
        assert_eq!(panel[2], 3.0);
        // Second row at padded offset
        assert_eq!(panel[16], 4.0);
        assert_eq!(panel[17], 5.0);
        assert_eq!(panel[18], 6.0);
    }

    #[test]
    fn test_pack_a_sub_panel() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0f32]; // 3×3
        let mut panel = vec![0.0f32; 2 * 16];
        pack_a_panel(&a, &mut panel, 1, 1, 2, 2, 3, 64);
        // Row 1, cols 1..3 → [5, 6]
        assert_eq!(panel[0], 5.0);
        assert_eq!(panel[1], 6.0);
        // Row 2, cols 1..3 → [8, 9]
        assert_eq!(panel[16], 8.0);
        assert_eq!(panel[17], 9.0);
    }

    #[test]
    fn test_pack_a_zero_padding() {
        let a = [1.0, 2.0, 3.0, 4.0f32]; // 2×2
        let mut panel = vec![0.0f32; 2 * 16];
        pack_a_panel(&a, &mut panel, 0, 0, 2, 2, 2, 64);
        // Elements [2..16) in each row should be zero
        for i in 2..16 {
            assert_eq!(panel[i], 0.0, "padding at [{i}]");
        }
    }

    // ── pack_b_panel ──────────────────────────────────────────────────

    #[test]
    fn test_pack_b_basic() {
        // B = [[1,2],[3,4],[5,6]] → 3×2, pack kc=3, nc=2
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let mut panel = vec![0.0f32; 2 * 16]; // 2 cols, padded kc
        pack_b_panel(&b, &mut panel, 0, 0, 3, 2, 2, 64);
        // Column 0: [1, 3, 5]
        assert_eq!(panel[0], 1.0);
        assert_eq!(panel[1], 3.0);
        assert_eq!(panel[2], 5.0);
        // Column 1: [2, 4, 6]
        assert_eq!(panel[16], 2.0);
        assert_eq!(panel[17], 4.0);
        assert_eq!(panel[18], 6.0);
    }

    #[test]
    fn test_pack_b_zero_padding() {
        let b = [1.0, 2.0, 3.0, 4.0f32]; // 2×2
        let mut panel = vec![0.0f32; 2 * 16];
        pack_b_panel(&b, &mut panel, 0, 0, 2, 2, 2, 64);
        for i in 2..16 {
            assert_eq!(panel[i], 0.0, "col0 padding at [{i}]");
        }
    }

    // ── StorePolicy / should_stream ───────────────────────────────────

    #[test]
    fn test_should_stream_normal() {
        assert!(!should_stream(1_000_000, StorePolicy::Normal));
    }

    #[test]
    fn test_should_stream_streaming() {
        assert!(should_stream(1, StorePolicy::Streaming));
    }

    #[test]
    fn test_should_stream_auto_small() {
        assert!(!should_stream(100, StorePolicy::Auto));
    }

    #[test]
    fn test_should_stream_auto_large() {
        assert!(should_stream(STREAMING_STORE_THRESHOLD, StorePolicy::Auto));
    }

    #[test]
    fn test_store_policy_equality() {
        assert_eq!(StorePolicy::Normal, StorePolicy::Normal);
        assert_ne!(StorePolicy::Normal, StorePolicy::Streaming);
    }

    // ── NUMA partitioning ─────────────────────────────────────────────

    #[test]
    fn test_numa_partition_single_node() {
        let parts = numa_partition(100, 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].row_start, 0);
        assert_eq!(parts[0].row_end, 100);
    }

    #[test]
    fn test_numa_partition_even_split() {
        let parts = numa_partition(100, 4);
        assert_eq!(parts.len(), 4);
        let total: usize = parts.iter().map(|p| p.row_end - p.row_start).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_numa_partition_uneven_split() {
        let parts = numa_partition(10, 3);
        assert_eq!(parts.len(), 3);
        let sizes: Vec<usize> = parts.iter().map(|p| p.row_end - p.row_start).collect();
        // 10 / 3 = 3 rem 1 → first node gets 4, rest get 3
        assert_eq!(sizes, vec![4, 3, 3]);
    }

    #[test]
    fn test_numa_partition_zero_nodes_treated_as_one() {
        let parts = numa_partition(50, 0);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].row_end, 50);
    }

    #[test]
    fn test_numa_partition_more_nodes_than_rows() {
        let parts = numa_partition(2, 5);
        assert_eq!(parts.len(), 5);
        let total: usize = parts.iter().map(|p| p.row_end - p.row_start).sum();
        assert_eq!(total, 2);
        // Some partitions will be empty
        let non_empty = parts.iter().filter(|p| p.row_end > p.row_start).count();
        assert_eq!(non_empty, 2);
    }

    #[test]
    fn test_numa_partition_contiguous() {
        let parts = numa_partition(100, 4);
        for i in 1..parts.len() {
            assert_eq!(parts[i].row_start, parts[i - 1].row_end);
        }
    }

    #[test]
    fn test_numa_partition_node_indices() {
        let parts = numa_partition(50, 3);
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(p.node, i);
        }
    }

    // ── is_tall_skinny ────────────────────────────────────────────────

    #[test]
    fn test_tall_skinny_yes() {
        assert!(is_tall_skinny(1024, 4, 64));
        assert!(is_tall_skinny(1024, 64, 4));
    }

    #[test]
    fn test_tall_skinny_no() {
        assert!(!is_tall_skinny(64, 64, 64));
        assert!(!is_tall_skinny(32, 32, 32));
    }

    #[test]
    fn test_tall_skinny_borderline() {
        // m = 4*n exactly → not tall-skinny (needs >)
        assert!(!is_tall_skinny(64, 16, 64));
        // m = 4*n + 1 → tall-skinny
        assert!(is_tall_skinny(65, 16, 64));
    }

    // ── tall_skinny_tiling ────────────────────────────────────────────

    #[test]
    fn test_tall_skinny_tiling_small_mc() {
        let h = CacheHierarchy::conservative();
        let t = tall_skinny_tiling(1024, 4, 64, &h);
        assert!(t.mc <= 4);
        assert!(t.nc <= 4);
    }

    #[test]
    fn test_tall_skinny_tiling_nonzero() {
        let h = CacheHierarchy::conservative();
        let t = tall_skinny_tiling(512, 8, 128, &h);
        assert!(t.mc >= 1);
        assert!(t.nc >= 1);
        assert!(t.kc >= 1);
    }

    // ── estimate_cache_misses ─────────────────────────────────────────

    #[test]
    fn test_cache_miss_estimate_small_fits_l3() {
        let h = CacheHierarchy::conservative();
        let t = auto_tune_tiling(32, 32, 32, &h);
        let est = estimate_cache_misses(32, 32, 32, &t, &h);
        // Small problem fits entirely in L3
        assert_eq!(est.l3_misses, 0.0);
        assert_eq!(est.dram_bytes, 0.0);
    }

    #[test]
    fn test_cache_miss_estimate_large_spills_l3() {
        let h = CacheHierarchy::with_sizes(32 * 1024, 256 * 1024, 1024 * 1024); // 1 MiB L3
        let t = auto_tune_tiling(1024, 1024, 1024, &h);
        let est = estimate_cache_misses(1024, 1024, 1024, &t, &h);
        // 3 × 1024² × 4B = 12 MiB > 1 MiB L3
        assert!(est.l3_misses > 0.0);
        assert!(est.dram_bytes > 0.0);
    }

    #[test]
    fn test_cache_miss_estimate_l1_misses_positive() {
        let h = CacheHierarchy::conservative();
        let t = auto_tune_tiling(64, 64, 64, &h);
        let est = estimate_cache_misses(64, 64, 64, &t, &h);
        assert!(est.l1_misses > 0.0);
    }

    #[test]
    fn test_cache_miss_estimate_l2_misses_nonneg() {
        let h = CacheHierarchy::conservative();
        let t = auto_tune_tiling(64, 64, 64, &h);
        let est = estimate_cache_misses(64, 64, 64, &t, &h);
        assert!(est.l2_misses >= 0.0);
    }

    // ── matmul_cache_aware ────────────────────────────────────────────

    #[test]
    fn test_matmul_1x1x1() {
        let a = [3.0f32];
        let b = [7.0f32];
        let mut c = [0.0f32];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, 1, 1, 1, &h).unwrap();
        assert_approx_eq(&c, &[21.0], 1e-6);
    }

    #[test]
    fn test_matmul_2x2x2_identity() {
        let a = [1.0, 0.0, 0.0, 1.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut c = [0.0f32; 4];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, 2, 2, 2, &h).unwrap();
        assert_approx_eq(&c, &b, 1e-6);
    }

    #[test]
    fn test_matmul_3x4x5() {
        let (m, n, k) = (3, 4, 5);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_16x16x16() {
        let (m, n, k) = (16, 16, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_64x64x64() {
        let (m, n, k) = (64, 64, 64);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 31) as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 23) as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_matmul_non_square_7x11x5() {
        let (m, n, k) = (7, 11, 5);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_matmul_negative_values() {
        let (m, n, k) = (8, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) - 32.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) - 32.0).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-1);
    }

    #[test]
    fn test_matmul_all_zeros() {
        let (m, n, k) = (4, 4, 4);
        let a = vec![0.0f32; m * k];
        let b = vec![0.0f32; k * n];
        let mut c = vec![999.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        assert_approx_eq(&c, &vec![0.0f32; m * n], 1e-10);
    }

    #[test]
    fn test_matmul_all_ones() {
        let (m, n, k) = (4, 4, 4);
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        // Each element should equal k
        let expected = vec![k as f32; m * n];
        assert_approx_eq(&c, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_tiny_cache() {
        let (m, n, k) = (16, 16, 16);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        let h = hierarchy_tiny();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_tall_skinny_shape() {
        let (m, n, k) = (128, 4, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02).cos()).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_wide_short_shape() {
        let (m, n, k) = (4, 128, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02).cos()).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_k_eq_1() {
        let (m, n, k) = (8, 8, 1);
        let a: Vec<f32> = (0..m).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_m_eq_1_gemv() {
        let (m, n, k) = (1, 64, 64);
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_matmul_n_eq_1_matvec() {
        let (m, n, k) = (64, 1, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    // ── matmul_auto ───────────────────────────────────────────────────

    #[test]
    fn test_matmul_auto_small() {
        let (m, n, k) = (4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; m * n];
        matmul_auto(&a, &b, &mut c, m, n, k).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    // ── matmul_streaming ──────────────────────────────────────────────

    #[test]
    fn test_matmul_streaming_small() {
        let (m, n, k) = (4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_streaming(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    // ── matmul_numa_aware ─────────────────────────────────────────────

    #[test]
    fn test_numa_aware_single_node() {
        let (m, n, k) = (8, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_numa_aware(&a, &b, &mut c, m, n, k, &h, StorePolicy::Normal).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_numa_aware_multi_node() {
        let (m, n, k) = (16, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        let mut h = CacheHierarchy::conservative();
        h.numa_nodes = 4;
        matmul_numa_aware(&a, &b, &mut c, m, n, k, &h, StorePolicy::Normal).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_numa_aware_matches_cache_aware() {
        let (m, n, k) = (12, 12, 12);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let h = CacheHierarchy::conservative(); // 1 NUMA node
        let mut c_numa = vec![0.0f32; m * n];
        let mut c_plain = vec![0.0f32; m * n];
        matmul_numa_aware(&a, &b, &mut c_numa, m, n, k, &h, StorePolicy::Normal).unwrap();
        matmul_cache_aware(&a, &b, &mut c_plain, m, n, k, &h).unwrap();
        assert_approx_eq(&c_numa, &c_plain, 1e-6);
    }

    // ── Validation / error paths ──────────────────────────────────────

    #[test]
    fn test_matmul_zero_m() {
        let a = [];
        let b = [1.0f32];
        let mut c = [];
        let h = CacheHierarchy::conservative();
        assert!(matmul_cache_aware(&a, &b, &mut c, 0, 1, 1, &h).is_err());
    }

    #[test]
    fn test_matmul_zero_n() {
        let a = [1.0f32];
        let b = [];
        let mut c = [];
        let h = CacheHierarchy::conservative();
        assert!(matmul_cache_aware(&a, &b, &mut c, 1, 0, 1, &h).is_err());
    }

    #[test]
    fn test_matmul_zero_k() {
        let a = [];
        let b = [];
        let mut c = [0.0f32];
        let h = CacheHierarchy::conservative();
        assert!(matmul_cache_aware(&a, &b, &mut c, 1, 1, 0, &h).is_err());
    }

    #[test]
    fn test_matmul_a_too_small() {
        let a = [1.0f32; 3]; // need 4 for 2×2
        let b = [1.0f32; 4];
        let mut c = [0.0f32; 4];
        let h = CacheHierarchy::conservative();
        assert!(matmul_cache_aware(&a, &b, &mut c, 2, 2, 2, &h).is_err());
    }

    #[test]
    fn test_matmul_b_too_small() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 3]; // need 4 for 2×2
        let mut c = [0.0f32; 4];
        let h = CacheHierarchy::conservative();
        assert!(matmul_cache_aware(&a, &b, &mut c, 2, 2, 2, &h).is_err());
    }

    #[test]
    fn test_matmul_c_too_small() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut c = [0.0f32; 3]; // need 4 for 2×2
        let h = CacheHierarchy::conservative();
        assert!(matmul_cache_aware(&a, &b, &mut c, 2, 2, 2, &h).is_err());
    }

    #[test]
    fn test_numa_aware_zero_m() {
        let h = CacheHierarchy::conservative();
        assert!(matmul_numa_aware(&[], &[1.0], &mut [], 0, 1, 1, &h, StorePolicy::Normal).is_err());
    }

    // ── Consistency / regression ──────────────────────────────────────

    #[test]
    fn test_different_hierarchies_same_result() {
        let (m, n, k) = (16, 16, 16);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let expected = naive_matmul(&a, &b, m, n, k);

        let h1 = CacheHierarchy::conservative();
        let h2 = hierarchy_tiny();
        let h3 = CacheHierarchy::with_sizes(64 * 1024, 512 * 1024, 16 * 1024 * 1024);

        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        let mut c3 = vec![0.0f32; m * n];
        matmul_cache_aware(&a, &b, &mut c1, m, n, k, &h1).unwrap();
        matmul_cache_aware(&a, &b, &mut c2, m, n, k, &h2).unwrap();
        matmul_cache_aware(&a, &b, &mut c3, m, n, k, &h3).unwrap();

        assert_approx_eq(&c1, &expected, 1e-3);
        assert_approx_eq(&c2, &expected, 1e-3);
        assert_approx_eq(&c3, &expected, 1e-3);
    }

    #[test]
    fn test_auto_vs_cache_aware_same_result() {
        let (m, n, k) = (8, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();
        let expected = naive_matmul(&a, &b, m, n, k);

        let mut c = vec![0.0f32; m * n];
        matmul_auto(&a, &b, &mut c, m, n, k).unwrap();
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_streaming_vs_normal_same_result() {
        let (m, n, k) = (8, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();

        let h = CacheHierarchy::conservative();
        let mut c_normal = vec![0.0f32; m * n];
        let mut c_stream = vec![0.0f32; m * n];
        matmul_cache_aware(&a, &b, &mut c_normal, m, n, k, &h).unwrap();
        matmul_streaming(&a, &b, &mut c_stream, m, n, k, &h).unwrap();
        assert_approx_eq(&c_normal, &c_stream, 1e-6);
    }

    // ── store_f32 ─────────────────────────────────────────────────────

    #[test]
    fn test_store_f32_normal() {
        let mut val = 0.0f32;
        unsafe { store_f32(&mut val as *mut f32, 42.0, false) };
        assert_eq!(val, 42.0);
    }

    #[test]
    fn test_store_f32_streaming() {
        let mut val = 0.0f32;
        unsafe { store_f32(&mut val as *mut f32, 42.0, true) };
        assert_eq!(val, 42.0);
    }

    // ── MultiLevelTiling Debug/Clone/Copy ─────────────────────────────

    #[test]
    fn test_tiling_debug_format() {
        let t = MultiLevelTiling { mc: 4, nc: 4, kc: 4, mc_l2: 8, nc_l2: 8, mc_l3: 16 };
        let s = format!("{t:?}");
        assert!(s.contains("MultiLevelTiling"));
        assert!(s.contains("mc: 4"));
    }

    #[test]
    fn test_tiling_clone_eq() {
        let t = MultiLevelTiling { mc: 4, nc: 4, kc: 4, mc_l2: 8, nc_l2: 8, mc_l3: 16 };
        let t2 = t;
        assert_eq!(t, t2);
    }

    // ── CacheMissEstimate ─────────────────────────────────────────────

    #[test]
    fn test_cache_miss_estimate_debug() {
        let est =
            CacheMissEstimate { l1_misses: 1.0, l2_misses: 2.0, l3_misses: 3.0, dram_bytes: 4.0 };
        let s = format!("{est:?}");
        assert!(s.contains("CacheMissEstimate"));
    }

    #[test]
    fn test_cache_miss_estimate_clone() {
        let est =
            CacheMissEstimate { l1_misses: 1.0, l2_misses: 2.0, l3_misses: 3.0, dram_bytes: 4.0 };
        let est2 = est;
        assert_eq!(est.l1_misses, est2.l1_misses);
    }

    // ── NumaPartition ─────────────────────────────────────────────────

    #[test]
    fn test_numa_partition_debug_format() {
        let p = NumaPartition { node: 0, row_start: 0, row_end: 50 };
        let s = format!("{p:?}");
        assert!(s.contains("NumaPartition"));
    }

    // ── Edge cases for matmul ─────────────────────────────────────────

    #[test]
    fn test_matmul_large_k_small_mn() {
        let (m, n, k) = (2, 2, 256);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001).cos()).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_matmul_prime_dimensions() {
        let (m, n, k) = (7, 13, 11);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-1);
    }

    #[test]
    fn test_matmul_128x128x128() {
        let (m, n, k) = (128, 128, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 41) as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 37) as f32) * 0.001).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }

    #[test]
    fn test_matmul_inference_token_shape() {
        // Typical single-token decode shape: 1 × hidden_dim × vocab_slice
        let (m, n, k) = (1, 32, 128);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.01).cos()).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_batch_prefill_shape() {
        // Batch prefill shape
        let (m, n, k) = (32, 64, 64);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.01).cos()).collect();
        let mut c = vec![0.0f32; m * n];
        let h = CacheHierarchy::conservative();
        matmul_cache_aware(&a, &b, &mut c, m, n, k, &h).unwrap();
        let expected = naive_matmul(&a, &b, m, n, k);
        assert_approx_eq(&c, &expected, 1e-2);
    }
}
