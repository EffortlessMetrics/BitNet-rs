//! Pure-Rust CUDA occupancy calculator.
//!
//! Computes theoretical and achieved occupancy, optimal block sizes, and
//! register-pressure analysis for common GPU architectures (SM 7.0–10.0)
//! without requiring a CUDA runtime.

use std::fmt;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors returned by the occupancy calculator.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OccupancyError {
    #[error("threads per block ({0}) exceeds device maximum ({1})")]
    ThreadsExceedMax(u32, u32),

    #[error("threads per block ({0}) is not a multiple of warp size ({1})")]
    ThreadsNotMultipleOfWarp(u32, u32),

    #[error("shared memory per block ({0} B) exceeds per-SM limit ({1} B)")]
    SharedMemoryExceeded(u32, u32),

    #[error(
        "registers per thread ({0}) exceeds per-SM register file \
         ({1}) divided by threads per block ({2})"
    )]
    RegistersExceeded(u32, u32, u32),

    #[error("zero threads per block")]
    ZeroThreads,

    #[error("zero warp size in GPU properties")]
    ZeroWarpSize,

    #[error("block size ({0}) too small; minimum is one warp ({1})")]
    BlockSizeTooSmall(u32, u32),

    #[error("unknown compute capability {0}.{1}")]
    UnknownArch(u32, u32),
}

// ---------------------------------------------------------------------------
// GPU properties
// ---------------------------------------------------------------------------

/// Hardware limits for a specific GPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuProperties {
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Maximum resident threads per SM.
    pub max_threads_per_sm: u32,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Maximum resident blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Shared memory per SM in bytes.
    pub shared_mem_per_sm: u32,
    /// Total 32-bit registers per SM.
    pub registers_per_sm: u32,
    /// Warp size (threads executed in lock-step).
    pub warp_size: u32,
    /// Maximum warps per SM (derived: max_threads_per_sm / warp_size).
    pub max_warps_per_sm: u32,
    /// Register allocation granularity (registers allocated in multiples of this).
    pub register_alloc_granularity: u32,
    /// Warp allocation granularity (warps allocated in multiples of this).
    pub warp_alloc_granularity: u32,
    /// Shared memory allocation granularity in bytes.
    pub shared_mem_alloc_granularity: u32,
    /// Compute capability major version.
    pub cc_major: u32,
    /// Compute capability minor version.
    pub cc_minor: u32,
}

impl GpuProperties {
    /// Create properties for a known compute capability.
    pub fn for_cc(major: u32, minor: u32) -> Result<Self, OccupancyError> {
        match (major, minor) {
            (7, 0) => Ok(Self::volta()),
            (8, 0) => Ok(Self::ampere()),
            (8, 9) => Ok(Self::ada()),
            (10, 0) => Ok(Self::blackwell()),
            _ => Err(OccupancyError::UnknownArch(major, minor)),
        }
    }

    /// SM 7.0 – Volta (V100).
    pub fn volta() -> Self {
        Self {
            sm_count: 80,
            max_threads_per_sm: 2048,
            max_threads_per_block: 1024,
            max_blocks_per_sm: 32,
            shared_mem_per_sm: 96 * 1024,
            registers_per_sm: 65536,
            warp_size: 32,
            max_warps_per_sm: 64,
            register_alloc_granularity: 256,
            warp_alloc_granularity: 4,
            shared_mem_alloc_granularity: 256,
            cc_major: 7,
            cc_minor: 0,
        }
    }

    /// SM 8.0 – Ampere (A100).
    pub fn ampere() -> Self {
        Self {
            sm_count: 108,
            max_threads_per_sm: 2048,
            max_threads_per_block: 1024,
            max_blocks_per_sm: 32,
            shared_mem_per_sm: 164 * 1024,
            registers_per_sm: 65536,
            warp_size: 32,
            max_warps_per_sm: 64,
            register_alloc_granularity: 256,
            warp_alloc_granularity: 4,
            shared_mem_alloc_granularity: 128,
            cc_major: 8,
            cc_minor: 0,
        }
    }

    /// SM 8.9 – Ada Lovelace (RTX 4090).
    pub fn ada() -> Self {
        Self {
            sm_count: 128,
            max_threads_per_sm: 1536,
            max_threads_per_block: 1024,
            max_blocks_per_sm: 24,
            shared_mem_per_sm: 100 * 1024,
            registers_per_sm: 65536,
            warp_size: 32,
            max_warps_per_sm: 48,
            register_alloc_granularity: 256,
            warp_alloc_granularity: 4,
            shared_mem_alloc_granularity: 128,
            cc_major: 8,
            cc_minor: 9,
        }
    }

    /// SM 10.0 – Blackwell (B200).
    pub fn blackwell() -> Self {
        Self {
            sm_count: 160,
            max_threads_per_sm: 2048,
            max_threads_per_block: 1024,
            max_blocks_per_sm: 32,
            shared_mem_per_sm: 228 * 1024,
            registers_per_sm: 65536,
            warp_size: 32,
            max_warps_per_sm: 64,
            register_alloc_granularity: 256,
            warp_alloc_granularity: 4,
            shared_mem_alloc_granularity: 128,
            cc_major: 10,
            cc_minor: 0,
        }
    }
}

impl fmt::Display for GpuProperties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SM {}.{} ({} SMs, {} threads/SM, {} shmem/SM, {} regs/SM)",
            self.cc_major,
            self.cc_minor,
            self.sm_count,
            self.max_threads_per_sm,
            self.shared_mem_per_sm,
            self.registers_per_sm,
        )
    }
}

// ---------------------------------------------------------------------------
// Kernel resources
// ---------------------------------------------------------------------------

/// Resource requirements for a single kernel launch configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelResources {
    /// Threads per block.
    pub threads_per_block: u32,
    /// Dynamic shared memory per block in bytes.
    pub shared_mem_per_block: u32,
    /// 32-bit registers used per thread.
    pub registers_per_thread: u32,
}

impl KernelResources {
    pub fn new(
        threads_per_block: u32,
        shared_mem_per_block: u32,
        registers_per_thread: u32,
    ) -> Self {
        Self { threads_per_block, shared_mem_per_block, registers_per_thread }
    }
}

// ---------------------------------------------------------------------------
// Occupancy result
// ---------------------------------------------------------------------------

/// Full occupancy analysis for a kernel / GPU pair.
#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyResult {
    /// Active blocks per SM.
    pub active_blocks_per_sm: u32,
    /// Active warps per SM.
    pub active_warps_per_sm: u32,
    /// Theoretical occupancy as a fraction in [0, 1].
    pub theoretical_occupancy: f64,
    /// Which resource limits the block count.
    pub limiting_factor: LimitingFactor,
    /// Active threads per SM.
    pub active_threads_per_sm: u32,
    /// Total active threads across the entire GPU.
    pub total_active_threads: u32,
}

impl fmt::Display for OccupancyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1}% occupancy ({} blocks/SM, {} warps/SM, limited by {})",
            self.theoretical_occupancy * 100.0,
            self.active_blocks_per_sm,
            self.active_warps_per_sm,
            self.limiting_factor,
        )
    }
}

/// Which hardware resource limits occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitingFactor {
    Warps,
    Registers,
    SharedMemory,
    MaxBlocks,
}

impl fmt::Display for LimitingFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Warps => write!(f, "warps"),
            Self::Registers => write!(f, "registers"),
            Self::SharedMemory => write!(f, "shared memory"),
            Self::MaxBlocks => write!(f, "max blocks"),
        }
    }
}

// ---------------------------------------------------------------------------
// Register pressure analysis
// ---------------------------------------------------------------------------

/// Detailed register-pressure report.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisterPressureAnalysis {
    /// Registers requested per thread.
    pub registers_per_thread: u32,
    /// Registers actually allocated per thread (after granularity rounding).
    pub allocated_registers_per_thread: u32,
    /// Total registers consumed by one block.
    pub registers_per_block: u32,
    /// Maximum concurrent blocks allowed by register pressure alone.
    pub max_blocks_by_registers: u32,
    /// Fraction of the register file used by the maximum resident blocks.
    pub register_utilization: f64,
    /// Register spill risk assessment.
    pub spill_risk: SpillRisk,
}

/// Qualitative spill-risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillRisk {
    Low,
    Medium,
    High,
}

impl fmt::Display for SpillRisk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: round-up integer division
// ---------------------------------------------------------------------------

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

fn round_up(value: u32, granularity: u32) -> u32 {
    div_ceil(value, granularity) * granularity
}

// ---------------------------------------------------------------------------
// Core validation
// ---------------------------------------------------------------------------

fn validate(gpu: &GpuProperties, kernel: &KernelResources) -> Result<(), OccupancyError> {
    if gpu.warp_size == 0 {
        return Err(OccupancyError::ZeroWarpSize);
    }
    if kernel.threads_per_block == 0 {
        return Err(OccupancyError::ZeroThreads);
    }
    if kernel.threads_per_block > gpu.max_threads_per_block {
        return Err(OccupancyError::ThreadsExceedMax(
            kernel.threads_per_block,
            gpu.max_threads_per_block,
        ));
    }
    if !kernel.threads_per_block.is_multiple_of(gpu.warp_size) {
        return Err(OccupancyError::ThreadsNotMultipleOfWarp(
            kernel.threads_per_block,
            gpu.warp_size,
        ));
    }
    if kernel.shared_mem_per_block > gpu.shared_mem_per_sm {
        return Err(OccupancyError::SharedMemoryExceeded(
            kernel.shared_mem_per_block,
            gpu.shared_mem_per_sm,
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the maximum number of active blocks per SM.
pub fn max_active_blocks_per_sm(
    gpu: &GpuProperties,
    kernel: &KernelResources,
) -> Result<u32, OccupancyError> {
    validate(gpu, kernel)?;

    let warps_per_block = kernel.threads_per_block / gpu.warp_size;

    // --- warp limit ---
    let limit_warps = gpu.max_warps_per_sm / warps_per_block;

    // --- block limit ---
    let limit_blocks = gpu.max_blocks_per_sm;

    // --- register limit ---
    let limit_regs = if kernel.registers_per_thread == 0 {
        gpu.max_blocks_per_sm
    } else {
        let regs_per_warp =
            round_up(kernel.registers_per_thread * gpu.warp_size, gpu.register_alloc_granularity);
        let regs_per_block = regs_per_warp * warps_per_block;
        if regs_per_block == 0 {
            gpu.max_blocks_per_sm
        } else {
            gpu.registers_per_sm / regs_per_block
        }
    };

    // --- shared-memory limit ---
    let limit_smem = if kernel.shared_mem_per_block == 0 {
        gpu.max_blocks_per_sm
    } else {
        let alloc_smem = round_up(kernel.shared_mem_per_block, gpu.shared_mem_alloc_granularity);
        gpu.shared_mem_per_sm / alloc_smem
    };

    Ok(limit_warps.min(limit_blocks).min(limit_regs).min(limit_smem))
}

/// Full occupancy calculation including limiting-factor identification.
pub fn calculate_occupancy(
    gpu: &GpuProperties,
    kernel: &KernelResources,
) -> Result<OccupancyResult, OccupancyError> {
    validate(gpu, kernel)?;

    let warps_per_block = kernel.threads_per_block / gpu.warp_size;

    // Compute per-resource limits.
    let limit_warps = gpu.max_warps_per_sm / warps_per_block;
    let limit_blocks = gpu.max_blocks_per_sm;

    let limit_regs = if kernel.registers_per_thread == 0 {
        gpu.max_blocks_per_sm
    } else {
        let regs_per_warp =
            round_up(kernel.registers_per_thread * gpu.warp_size, gpu.register_alloc_granularity);
        let regs_per_block = regs_per_warp * warps_per_block;
        if regs_per_block == 0 {
            gpu.max_blocks_per_sm
        } else {
            gpu.registers_per_sm / regs_per_block
        }
    };

    let limit_smem = if kernel.shared_mem_per_block == 0 {
        gpu.max_blocks_per_sm
    } else {
        let alloc_smem = round_up(kernel.shared_mem_per_block, gpu.shared_mem_alloc_granularity);
        gpu.shared_mem_per_sm / alloc_smem
    };

    let active_blocks = limit_warps.min(limit_blocks).min(limit_regs).min(limit_smem);

    // Track which resource limits are "real" (actually constrained by the
    // kernel config) vs sentinel values (set to max_blocks when the resource
    // is unused, e.g. 0 regs or 0 smem).
    let regs_is_real = kernel.registers_per_thread > 0;
    let smem_is_real = kernel.shared_mem_per_block > 0;

    let limiting_factor = if active_blocks == 0 {
        if limit_regs == 0 {
            LimitingFactor::Registers
        } else if limit_smem == 0 {
            LimitingFactor::SharedMemory
        } else {
            LimitingFactor::Warps
        }
    } else {
        // Among the limits that actually equal the minimum, prefer real
        // constraints over sentinels, and use this priority order:
        // Registers > SharedMemory > Warps > MaxBlocks.
        let min = active_blocks;
        if regs_is_real
            && min == limit_regs
            && limit_regs <= limit_smem
            && limit_regs <= limit_warps
        {
            LimitingFactor::Registers
        } else if smem_is_real
            && min == limit_smem
            && limit_smem <= limit_warps
            && limit_smem <= limit_regs
        {
            LimitingFactor::SharedMemory
        } else if min == limit_warps && limit_warps <= limit_blocks {
            LimitingFactor::Warps
        } else {
            LimitingFactor::MaxBlocks
        }
    };

    let active_warps = active_blocks * warps_per_block;
    let active_threads = active_warps * gpu.warp_size;
    let occupancy = if gpu.max_warps_per_sm == 0 {
        0.0
    } else {
        active_warps as f64 / gpu.max_warps_per_sm as f64
    };

    Ok(OccupancyResult {
        active_blocks_per_sm: active_blocks,
        active_warps_per_sm: active_warps,
        theoretical_occupancy: occupancy,
        limiting_factor,
        active_threads_per_sm: active_threads,
        total_active_threads: active_threads * gpu.sm_count,
    })
}

/// Convenience wrapper: return only the theoretical occupancy fraction.
pub fn theoretical_occupancy(
    gpu: &GpuProperties,
    kernel: &KernelResources,
) -> Result<f64, OccupancyError> {
    calculate_occupancy(gpu, kernel).map(|r| r.theoretical_occupancy)
}

/// Estimate *achieved* occupancy given an active-cycle measurement.
///
/// `active_warps_avg` is the time-averaged number of active warps per SM
/// (e.g. from `sm__warps_active.avg` in Nsight Compute).
pub fn achieved_occupancy(gpu: &GpuProperties, active_warps_avg: f64) -> f64 {
    if gpu.max_warps_per_sm == 0 {
        return 0.0;
    }
    (active_warps_avg / gpu.max_warps_per_sm as f64).clamp(0.0, 1.0)
}

/// Search for the block size that maximises occupancy.
///
/// Iterates from `warp_size` up to `max_threads_per_block` in warp-size
/// increments and returns `(optimal_threads_per_block, OccupancyResult)`.
pub fn optimal_block_size(
    gpu: &GpuProperties,
    shared_mem_per_block: u32,
    registers_per_thread: u32,
) -> Result<(u32, OccupancyResult), OccupancyError> {
    if gpu.warp_size == 0 {
        return Err(OccupancyError::ZeroWarpSize);
    }

    let mut best: Option<(u32, OccupancyResult)> = None;

    let mut threads = gpu.warp_size;
    while threads <= gpu.max_threads_per_block {
        let kernel = KernelResources::new(threads, shared_mem_per_block, registers_per_thread);
        if let Ok(result) = calculate_occupancy(gpu, &kernel) {
            let replace = match &best {
                None => true,
                Some((_, prev)) => {
                    result.active_warps_per_sm > prev.active_warps_per_sm
                        || (result.active_warps_per_sm == prev.active_warps_per_sm
                            && threads < best.as_ref().unwrap().0)
                }
            };
            if replace {
                best = Some((threads, result));
            }
        }
        threads += gpu.warp_size;
    }

    best.ok_or(OccupancyError::BlockSizeTooSmall(0, gpu.warp_size))
}

/// Analyse register pressure for a given kernel configuration.
pub fn register_pressure_analysis(
    gpu: &GpuProperties,
    kernel: &KernelResources,
) -> Result<RegisterPressureAnalysis, OccupancyError> {
    validate(gpu, kernel)?;

    let warps_per_block = kernel.threads_per_block / gpu.warp_size;

    let (allocated_per_thread, regs_per_block, max_blocks) = if kernel.registers_per_thread == 0 {
        (0, 0, gpu.max_blocks_per_sm)
    } else {
        let regs_per_warp =
            round_up(kernel.registers_per_thread * gpu.warp_size, gpu.register_alloc_granularity);
        let allocated_per_thread = regs_per_warp / gpu.warp_size;
        let rpb = regs_per_warp * warps_per_block;
        let mb = if rpb == 0 { gpu.max_blocks_per_sm } else { gpu.registers_per_sm / rpb };
        (allocated_per_thread, rpb, mb)
    };

    let total_regs_used = regs_per_block * max_blocks;
    let utilization = if gpu.registers_per_sm == 0 {
        0.0
    } else {
        total_regs_used as f64 / gpu.registers_per_sm as f64
    };

    let spill_risk = if kernel.registers_per_thread <= 32 {
        SpillRisk::Low
    } else if kernel.registers_per_thread <= 64 {
        SpillRisk::Medium
    } else {
        SpillRisk::High
    };

    Ok(RegisterPressureAnalysis {
        registers_per_thread: kernel.registers_per_thread,
        allocated_registers_per_thread: allocated_per_thread,
        registers_per_block: regs_per_block,
        max_blocks_by_registers: max_blocks,
        register_utilization: utilization,
        spill_risk,
    })
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn volta() -> GpuProperties {
        GpuProperties::volta()
    }
    fn ampere() -> GpuProperties {
        GpuProperties::ampere()
    }
    fn ada() -> GpuProperties {
        GpuProperties::ada()
    }
    fn blackwell() -> GpuProperties {
        GpuProperties::blackwell()
    }

    fn simple_kernel(threads: u32, smem: u32, regs: u32) -> KernelResources {
        KernelResources::new(threads, smem, regs)
    }

    // =====================================================================
    // GpuProperties construction
    // =====================================================================

    #[test]
    fn volta_properties() {
        let g = volta();
        assert_eq!(g.cc_major, 7);
        assert_eq!(g.cc_minor, 0);
        assert_eq!(g.max_warps_per_sm, 64);
        assert_eq!(g.warp_size, 32);
    }

    #[test]
    fn ampere_properties() {
        let g = ampere();
        assert_eq!(g.cc_major, 8);
        assert_eq!(g.cc_minor, 0);
        assert_eq!(g.sm_count, 108);
    }

    #[test]
    fn ada_properties() {
        let g = ada();
        assert_eq!(g.cc_major, 8);
        assert_eq!(g.cc_minor, 9);
        assert_eq!(g.max_warps_per_sm, 48);
    }

    #[test]
    fn blackwell_properties() {
        let g = blackwell();
        assert_eq!(g.cc_major, 10);
        assert_eq!(g.cc_minor, 0);
        assert_eq!(g.sm_count, 160);
    }

    #[test]
    fn for_cc_known() {
        assert!(GpuProperties::for_cc(7, 0).is_ok());
        assert!(GpuProperties::for_cc(8, 0).is_ok());
        assert!(GpuProperties::for_cc(8, 9).is_ok());
        assert!(GpuProperties::for_cc(10, 0).is_ok());
    }

    #[test]
    fn for_cc_unknown() {
        assert_eq!(GpuProperties::for_cc(99, 0).unwrap_err(), OccupancyError::UnknownArch(99, 0),);
    }

    #[test]
    fn gpu_display() {
        let g = volta();
        let s = format!("{g}");
        assert!(s.contains("SM 7.0"));
        assert!(s.contains("80 SMs"));
    }

    // =====================================================================
    // Validation errors
    // =====================================================================

    #[test]
    fn err_zero_threads() {
        let r = calculate_occupancy(&volta(), &simple_kernel(0, 0, 32));
        assert_eq!(r.unwrap_err(), OccupancyError::ZeroThreads);
    }

    #[test]
    fn err_threads_exceed_max() {
        let r = calculate_occupancy(&volta(), &simple_kernel(2048, 0, 32));
        assert_eq!(r.unwrap_err(), OccupancyError::ThreadsExceedMax(2048, 1024),);
    }

    #[test]
    fn err_threads_not_multiple_of_warp() {
        let r = calculate_occupancy(&volta(), &simple_kernel(33, 0, 32));
        assert_eq!(r.unwrap_err(), OccupancyError::ThreadsNotMultipleOfWarp(33, 32),);
    }

    #[test]
    fn err_shared_memory_exceeded() {
        let big = volta().shared_mem_per_sm + 1;
        let r = calculate_occupancy(&volta(), &simple_kernel(32, big, 0));
        assert_eq!(
            r.unwrap_err(),
            OccupancyError::SharedMemoryExceeded(big, volta().shared_mem_per_sm),
        );
    }

    #[test]
    fn err_zero_warp_size() {
        let mut g = volta();
        g.warp_size = 0;
        assert_eq!(
            calculate_occupancy(&g, &simple_kernel(32, 0, 32)).unwrap_err(),
            OccupancyError::ZeroWarpSize,
        );
    }

    #[test]
    fn err_zero_warp_size_optimal() {
        let mut g = volta();
        g.warp_size = 0;
        assert_eq!(optimal_block_size(&g, 0, 32).unwrap_err(), OccupancyError::ZeroWarpSize,);
    }

    // =====================================================================
    // max_active_blocks_per_sm
    // =====================================================================

    #[test]
    fn max_blocks_full_occupancy_volta() {
        // 1024 threads, 0 smem, 32 regs → 2 blocks × 32 warps = 64 warps = 100%
        let b = max_active_blocks_per_sm(&volta(), &simple_kernel(1024, 0, 32)).unwrap();
        assert_eq!(b, 2);
    }

    #[test]
    fn max_blocks_small_block_volta() {
        // 32 threads → 1 warp/block → limited by max_blocks_per_sm = 32
        let b = max_active_blocks_per_sm(&volta(), &simple_kernel(32, 0, 0)).unwrap();
        assert_eq!(b, 32);
    }

    #[test]
    fn max_blocks_shared_mem_limited() {
        // 256 threads, 50 KB smem, 0 regs on Volta (96 KB smem/SM)
        let b = max_active_blocks_per_sm(&volta(), &simple_kernel(256, 50 * 1024, 0)).unwrap();
        // 96 KB / round_up(50 KB, 256) = 96 KB / 50 KB (rounded) → 1
        assert!(b >= 1);
    }

    #[test]
    fn max_blocks_register_limited() {
        // 256 threads, 0 smem, 128 regs/thread
        // 128 * 32 = 4096 per warp → round_up(4096, 256) = 4096
        // 256 threads = 8 warps → 8 * 4096 = 32768 regs/block
        // 65536 / 32768 = 2 blocks max
        let b = max_active_blocks_per_sm(&volta(), &simple_kernel(256, 0, 128)).unwrap();
        assert_eq!(b, 2);
    }

    #[test]
    fn max_blocks_zero_regs() {
        let b = max_active_blocks_per_sm(&volta(), &simple_kernel(256, 0, 0)).unwrap();
        // No register constraint → limited by warps (64 / 8 = 8) and blocks (32).
        assert_eq!(b, 8);
    }

    #[test]
    fn max_blocks_ampere_256t_32r() {
        let b = max_active_blocks_per_sm(&ampere(), &simple_kernel(256, 0, 32)).unwrap();
        assert!(b >= 1);
    }

    #[test]
    fn max_blocks_ada_256t_32r() {
        let b = max_active_blocks_per_sm(&ada(), &simple_kernel(256, 0, 32)).unwrap();
        assert!(b >= 1);
    }

    #[test]
    fn max_blocks_blackwell_256t_32r() {
        let b = max_active_blocks_per_sm(&blackwell(), &simple_kernel(256, 0, 32)).unwrap();
        assert!(b >= 1);
    }

    // =====================================================================
    // calculate_occupancy
    // =====================================================================

    #[test]
    fn full_occupancy_volta() {
        let r = calculate_occupancy(&volta(), &simple_kernel(1024, 0, 32)).unwrap();
        assert!((r.theoretical_occupancy - 1.0).abs() < 1e-9);
        assert_eq!(r.active_warps_per_sm, 64);
    }

    #[test]
    fn half_occupancy_volta_512t() {
        let r = calculate_occupancy(&volta(), &simple_kernel(512, 0, 64)).unwrap();
        // 512 threads = 16 warps/block. 64 regs → 64*32=2048 per warp
        // round_up(2048,256)=2048. 16 warps * 2048 = 32768 regs/block
        // 65536/32768 = 2 blocks → 32 warps → 50%
        assert!((r.theoretical_occupancy - 0.5).abs() < 1e-9);
    }

    #[test]
    fn occupancy_display() {
        let r = calculate_occupancy(&volta(), &simple_kernel(256, 0, 32)).unwrap();
        let s = format!("{r}");
        assert!(s.contains("occupancy"));
        assert!(s.contains("blocks/SM"));
    }

    #[test]
    fn occupancy_limiting_factor_regs() {
        let r = calculate_occupancy(&volta(), &simple_kernel(256, 0, 128)).unwrap();
        assert_eq!(r.limiting_factor, LimitingFactor::Registers);
    }

    #[test]
    fn occupancy_limiting_factor_smem() {
        // Large smem per block on Volta: 96 KB / block ⇒ 1 block
        let smem = 80 * 1024; // 80 KB
        let r = calculate_occupancy(&volta(), &simple_kernel(256, smem, 0)).unwrap();
        assert_eq!(r.limiting_factor, LimitingFactor::SharedMemory);
    }

    #[test]
    fn occupancy_limiting_factor_max_blocks() {
        // 32 threads (1 warp/block), 0 smem, 0 regs → warp limit = 64,
        // block limit = 32. The bottleneck is max-blocks.
        let r = calculate_occupancy(&volta(), &simple_kernel(32, 0, 0)).unwrap();
        assert_eq!(r.limiting_factor, LimitingFactor::MaxBlocks);
    }

    #[test]
    fn occupancy_total_active_threads() {
        let r = calculate_occupancy(&volta(), &simple_kernel(256, 0, 32)).unwrap();
        assert_eq!(r.total_active_threads, r.active_threads_per_sm * volta().sm_count);
    }

    #[test]
    fn occupancy_ampere_full() {
        let r = calculate_occupancy(&ampere(), &simple_kernel(256, 0, 32)).unwrap();
        assert!(r.theoretical_occupancy > 0.0);
        assert!(r.theoretical_occupancy <= 1.0);
    }

    #[test]
    fn occupancy_ada_full() {
        let r = calculate_occupancy(&ada(), &simple_kernel(256, 0, 32)).unwrap();
        assert!(r.theoretical_occupancy > 0.0);
    }

    #[test]
    fn occupancy_blackwell_full() {
        let r = calculate_occupancy(&blackwell(), &simple_kernel(256, 0, 32)).unwrap();
        assert!(r.theoretical_occupancy > 0.0);
    }

    // =====================================================================
    // theoretical_occupancy (wrapper)
    // =====================================================================

    #[test]
    fn theoretical_matches_calculate() {
        let gpu = volta();
        let k = simple_kernel(256, 0, 32);
        let t = theoretical_occupancy(&gpu, &k).unwrap();
        let c = calculate_occupancy(&gpu, &k).unwrap();
        assert!((t - c.theoretical_occupancy).abs() < 1e-12);
    }

    #[test]
    fn theoretical_error_propagates() {
        assert!(theoretical_occupancy(&volta(), &simple_kernel(0, 0, 0)).is_err());
    }

    // =====================================================================
    // achieved_occupancy
    // =====================================================================

    #[test]
    fn achieved_occupancy_basic() {
        let a = achieved_occupancy(&volta(), 32.0);
        assert!((a - 0.5).abs() < 1e-9);
    }

    #[test]
    fn achieved_occupancy_clamped_above() {
        let a = achieved_occupancy(&volta(), 200.0);
        assert!((a - 1.0).abs() < 1e-9);
    }

    #[test]
    fn achieved_occupancy_clamped_below() {
        let a = achieved_occupancy(&volta(), -5.0);
        assert!((a - 0.0).abs() < 1e-9);
    }

    #[test]
    fn achieved_occupancy_zero_max_warps() {
        let mut g = volta();
        g.max_warps_per_sm = 0;
        assert!((achieved_occupancy(&g, 10.0) - 0.0).abs() < 1e-9);
    }

    // =====================================================================
    // optimal_block_size
    // =====================================================================

    #[test]
    fn optimal_block_volta_low_regs() {
        let (threads, result) = optimal_block_size(&volta(), 0, 32).unwrap();
        assert!(threads >= 32);
        assert!(threads <= volta().max_threads_per_block);
        assert!(threads % 32 == 0);
        assert!(result.theoretical_occupancy > 0.0);
    }

    #[test]
    fn optimal_block_volta_high_regs() {
        let (threads, result) = optimal_block_size(&volta(), 0, 128).unwrap();
        assert!(threads >= 32);
        // High registers should still find a viable configuration
        assert!(result.active_blocks_per_sm >= 1);
    }

    #[test]
    fn optimal_prefers_smaller_at_equal_occupancy() {
        // With 0 regs and 0 smem, many block sizes hit 100%. Optimal should
        // pick the smallest (fewest threads) achieving max warps.
        let (threads, _) = optimal_block_size(&volta(), 0, 0).unwrap();
        // Anything that achieves 64 active warps is fine; just check it's a
        // multiple of 32.
        assert!(threads % 32 == 0);
    }

    #[test]
    fn optimal_block_ampere() {
        let (t, r) = optimal_block_size(&ampere(), 0, 32).unwrap();
        assert!(t >= 32 && t <= 1024);
        assert!(r.theoretical_occupancy > 0.0);
    }

    #[test]
    fn optimal_block_ada() {
        let (t, r) = optimal_block_size(&ada(), 0, 32).unwrap();
        assert!(t >= 32 && t <= 1024);
        assert!(r.theoretical_occupancy > 0.0);
    }

    #[test]
    fn optimal_block_blackwell() {
        let (t, r) = optimal_block_size(&blackwell(), 0, 32).unwrap();
        assert!(t >= 32 && t <= 1024);
        assert!(r.theoretical_occupancy > 0.0);
    }

    #[test]
    fn optimal_block_with_smem() {
        let (t, r) = optimal_block_size(&volta(), 48 * 1024, 32).unwrap();
        assert!(t >= 32);
        assert!(r.active_blocks_per_sm >= 1);
    }

    // =====================================================================
    // register_pressure_analysis
    // =====================================================================

    #[test]
    fn reg_pressure_low() {
        let a = register_pressure_analysis(&volta(), &simple_kernel(256, 0, 16)).unwrap();
        assert_eq!(a.spill_risk, SpillRisk::Low);
        assert_eq!(a.registers_per_thread, 16);
    }

    #[test]
    fn reg_pressure_medium() {
        let a = register_pressure_analysis(&volta(), &simple_kernel(256, 0, 48)).unwrap();
        assert_eq!(a.spill_risk, SpillRisk::Medium);
    }

    #[test]
    fn reg_pressure_high() {
        let a = register_pressure_analysis(&volta(), &simple_kernel(256, 0, 128)).unwrap();
        assert_eq!(a.spill_risk, SpillRisk::High);
    }

    #[test]
    fn reg_pressure_zero_regs() {
        let a = register_pressure_analysis(&volta(), &simple_kernel(256, 0, 0)).unwrap();
        assert_eq!(a.allocated_registers_per_thread, 0);
        assert_eq!(a.registers_per_block, 0);
    }

    #[test]
    fn reg_pressure_utilization_bounded() {
        let a = register_pressure_analysis(&volta(), &simple_kernel(256, 0, 32)).unwrap();
        assert!(a.register_utilization >= 0.0);
        assert!(a.register_utilization <= 1.0);
    }

    #[test]
    fn reg_pressure_allocated_ge_requested() {
        let a = register_pressure_analysis(&volta(), &simple_kernel(256, 0, 30)).unwrap();
        assert!(a.allocated_registers_per_thread >= a.registers_per_thread);
    }

    #[test]
    fn reg_pressure_ampere() {
        let a = register_pressure_analysis(&ampere(), &simple_kernel(256, 0, 32)).unwrap();
        assert!(a.max_blocks_by_registers >= 1);
    }

    #[test]
    fn reg_pressure_error_propagates() {
        assert!(register_pressure_analysis(&volta(), &simple_kernel(0, 0, 32)).is_err());
    }

    // =====================================================================
    // Cross-architecture consistency
    // =====================================================================

    #[test]
    fn all_archs_handle_256t_32r() {
        for gpu in [volta(), ampere(), ada(), blackwell()] {
            let r = calculate_occupancy(&gpu, &simple_kernel(256, 0, 32)).unwrap();
            assert!(r.theoretical_occupancy > 0.0);
            assert!(r.active_blocks_per_sm >= 1);
        }
    }

    #[test]
    fn all_archs_handle_1024t_0r() {
        for gpu in [volta(), ampere(), ada(), blackwell()] {
            let r = calculate_occupancy(&gpu, &simple_kernel(1024, 0, 0)).unwrap();
            assert!(
                (r.theoretical_occupancy - 1.0).abs() < 1e-9 || r.theoretical_occupancy > 0.5,
                "Expected high occupancy for {gpu}",
            );
        }
    }

    #[test]
    fn occupancy_never_exceeds_one() {
        for gpu in [volta(), ampere(), ada(), blackwell()] {
            for &threads in &[32, 64, 128, 256, 512, 1024] {
                for &regs in &[0, 16, 32, 64, 128, 255] {
                    if let Ok(r) = calculate_occupancy(&gpu, &simple_kernel(threads, 0, regs)) {
                        assert!(
                            r.theoretical_occupancy <= 1.0 + 1e-12,
                            "occupancy {:.4} > 1.0 for {gpu}, threads={threads}, regs={regs}",
                            r.theoretical_occupancy,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn active_warps_le_max() {
        for gpu in [volta(), ampere(), ada(), blackwell()] {
            for &threads in &[32, 128, 256, 512, 1024] {
                if let Ok(r) = calculate_occupancy(&gpu, &simple_kernel(threads, 0, 32)) {
                    assert!(r.active_warps_per_sm <= gpu.max_warps_per_sm);
                }
            }
        }
    }

    #[test]
    fn active_blocks_le_max() {
        for gpu in [volta(), ampere(), ada(), blackwell()] {
            for &threads in &[32, 128, 256, 512, 1024] {
                if let Ok(r) = calculate_occupancy(&gpu, &simple_kernel(threads, 0, 32)) {
                    assert!(r.active_blocks_per_sm <= gpu.max_blocks_per_sm);
                }
            }
        }
    }

    // =====================================================================
    // Edge-case / miscellaneous
    // =====================================================================

    #[test]
    fn single_warp_block() {
        let r = calculate_occupancy(&volta(), &simple_kernel(32, 0, 32)).unwrap();
        assert!(r.active_blocks_per_sm >= 1);
    }

    #[test]
    fn max_block_size_with_high_regs() {
        // 1024 threads, 255 regs → register file is exhausted (0 blocks).
        let r = calculate_occupancy(&volta(), &simple_kernel(1024, 0, 255)).unwrap();
        assert_eq!(r.active_blocks_per_sm, 0);
        assert_eq!(r.limiting_factor, LimitingFactor::Registers);
        assert!((r.theoretical_occupancy - 0.0).abs() < 1e-12);
    }

    #[test]
    fn smem_alloc_granularity_rounding() {
        // 1 byte of smem still allocates a full granularity chunk.
        let r = calculate_occupancy(&volta(), &simple_kernel(256, 1, 0)).unwrap();
        assert!(r.active_blocks_per_sm >= 1);
    }

    #[test]
    fn max_smem_per_block_exact() {
        // Use exactly the per-SM shared memory limit.
        let smem = volta().shared_mem_per_sm;
        let r = calculate_occupancy(&volta(), &simple_kernel(256, smem, 0)).unwrap();
        assert_eq!(r.active_blocks_per_sm, 1);
    }

    #[test]
    fn limiting_factor_display() {
        assert_eq!(format!("{}", LimitingFactor::Warps), "warps");
        assert_eq!(format!("{}", LimitingFactor::Registers), "registers");
        assert_eq!(format!("{}", LimitingFactor::SharedMemory), "shared memory");
        assert_eq!(format!("{}", LimitingFactor::MaxBlocks), "max blocks");
    }

    #[test]
    fn spill_risk_display() {
        assert_eq!(format!("{}", SpillRisk::Low), "low");
        assert_eq!(format!("{}", SpillRisk::Medium), "medium");
        assert_eq!(format!("{}", SpillRisk::High), "high");
    }

    #[test]
    fn kernel_resources_new() {
        let k = KernelResources::new(128, 1024, 32);
        assert_eq!(k.threads_per_block, 128);
        assert_eq!(k.shared_mem_per_block, 1024);
        assert_eq!(k.registers_per_thread, 32);
    }

    #[test]
    fn max_active_blocks_matches_calculate() {
        let gpu = volta();
        let k = simple_kernel(256, 0, 32);
        let blocks = max_active_blocks_per_sm(&gpu, &k).unwrap();
        let occ = calculate_occupancy(&gpu, &k).unwrap();
        assert_eq!(blocks, occ.active_blocks_per_sm);
    }

    #[test]
    fn round_up_helper() {
        assert_eq!(round_up(1, 256), 256);
        assert_eq!(round_up(256, 256), 256);
        assert_eq!(round_up(257, 256), 512);
    }

    #[test]
    fn div_ceil_helper() {
        assert_eq!(div_ceil(1, 1), 1);
        assert_eq!(div_ceil(5, 3), 2);
        assert_eq!(div_ceil(6, 3), 2);
    }

    // =====================================================================
    // proptest properties
    // =====================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_gpu() -> impl Strategy<Value = GpuProperties> {
            prop_oneof![Just(volta()), Just(ampere()), Just(ada()), Just(blackwell()),]
        }

        fn arb_threads(gpu: &GpuProperties) -> impl Strategy<Value = u32> {
            let ws = gpu.warp_size;
            let max = gpu.max_threads_per_block / ws;
            (1..=max).prop_map(move |n| n * ws)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(300))]

            #[test]
            fn occupancy_in_zero_one(gpu in arb_gpu(), regs in 0u32..256) {
                let threads_vals: Vec<u32> = (1..=(gpu.max_threads_per_block / gpu.warp_size))
                    .map(|n| n * gpu.warp_size)
                    .collect();
                for &threads in &threads_vals {
                    if let Ok(r) = calculate_occupancy(&gpu, &simple_kernel(threads, 0, regs)) {
                        prop_assert!(r.theoretical_occupancy >= 0.0);
                        prop_assert!(r.theoretical_occupancy <= 1.0 + 1e-12);
                    }
                }
            }

            #[test]
            fn active_warps_bounded(
                gpu in arb_gpu(),
                regs in 0u32..128,
            ) {
                for &threads in &[32u32, 128, 256, 512, 1024] {
                    if threads <= gpu.max_threads_per_block {
                        if let Ok(r) = calculate_occupancy(&gpu, &simple_kernel(threads, 0, regs)) {
                            prop_assert!(r.active_warps_per_sm <= gpu.max_warps_per_sm);
                        }
                    }
                }
            }

            #[test]
            fn more_regs_never_increases_occupancy(
                gpu in arb_gpu(),
                lo in 1u32..64,
                delta in 1u32..192,
            ) {
                let hi = lo + delta;
                let k_lo = simple_kernel(256, 0, lo);
                let k_hi = simple_kernel(256, 0, hi);
                if let (Ok(r_lo), Ok(r_hi)) = (
                    calculate_occupancy(&gpu, &k_lo),
                    calculate_occupancy(&gpu, &k_hi),
                ) {
                    prop_assert!(r_hi.active_warps_per_sm <= r_lo.active_warps_per_sm);
                }
            }

            #[test]
            fn more_smem_never_increases_occupancy(
                gpu in arb_gpu(),
                lo in 0u32..16384,
                delta in 1u32..16384,
            ) {
                let hi = lo.saturating_add(delta).min(gpu.shared_mem_per_sm);
                let k_lo = simple_kernel(256, lo, 32);
                let k_hi = simple_kernel(256, hi, 32);
                if let (Ok(r_lo), Ok(r_hi)) = (
                    calculate_occupancy(&gpu, &k_lo),
                    calculate_occupancy(&gpu, &k_hi),
                ) {
                    prop_assert!(r_hi.active_warps_per_sm <= r_lo.active_warps_per_sm);
                }
            }

            #[test]
            fn optimal_at_least_as_good_as_256(gpu in arb_gpu()) {
                if let Ok((_, opt)) = optimal_block_size(&gpu, 0, 32) {
                    if let Ok(baseline) = calculate_occupancy(&gpu, &simple_kernel(256, 0, 32)) {
                        prop_assert!(opt.active_warps_per_sm >= baseline.active_warps_per_sm);
                    }
                }
            }
        }
    }
}
