//! CUDA register pressure optimization and occupancy trade-off analysis.
//!
//! This module provides tools for estimating, analyzing, and optimizing register
//! usage in CUDA kernels. High register pressure reduces occupancy (fewer active
//! warps per SM), while too-aggressive spilling trades latency for occupancy.
//!
//! # Components
//!
//! - [`RegisterEstimate`] — per-kernel register usage estimation
//! - [`KernelPattern`] — categorized kernel patterns with expected register profiles
//! - [`SpillDetector`] — detection of register spilling and mitigation strategies
//! - [`OccupancyTradeoff`] — occupancy vs register trade-off analysis
//! - [`LaunchBoundsCalculator`] — `__launch_bounds__` parameter computation
//! - [`RegisterPartitioner`] — register file partitioning across warps
//! - [`LiveRangeAnalyzer`] — variable live range analysis for register optimization
//! - [`CompilerHintGenerator`] — `__launch_bounds__` and `maxrregcount` hint generation
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations are always available for analysis and testing.

use std::collections::HashMap;
use std::fmt;

use bitnet_common::{KernelError, Result};

// ── Constants ────────────────────────────────────────────────────────

/// Standard CUDA warp size (threads per warp).
const WARP_SIZE: u32 = 32;

/// Register allocation granularity on modern NVIDIA GPUs (registers are
/// allocated in chunks of this many per warp).
const REGISTER_ALLOC_GRANULARITY: u32 = 256;

/// Maximum registers per thread on most architectures.
const MAX_REGISTERS_PER_THREAD: u32 = 255;

// ── GpuArch ──────────────────────────────────────────────────────────

/// CUDA GPU architecture generation with register file specifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArch {
    /// SM 7.0 — Volta (V100).
    Volta,
    /// SM 7.5 — Turing (RTX 20xx).
    Turing,
    /// SM 8.0 — Ampere (A100).
    Ampere,
    /// SM 8.6 — Ampere consumer (RTX 30xx).
    AmpereConsumer,
    /// SM 8.9 — Ada Lovelace (RTX 40xx).
    Ada,
    /// SM 9.0 — Hopper (H100).
    Hopper,
}

impl GpuArch {
    /// Total 32-bit registers per SM.
    pub fn registers_per_sm(self) -> u32 {
        match self {
            Self::Volta | Self::Turing => 65_536,
            Self::Ampere | Self::AmpereConsumer => 65_536,
            Self::Ada => 65_536,
            Self::Hopper => 65_536,
        }
    }

    /// Maximum resident blocks per SM.
    pub fn max_blocks_per_sm(self) -> u32 {
        match self {
            Self::Volta | Self::Turing => 32,
            Self::Ampere | Self::AmpereConsumer | Self::Ada => 32,
            Self::Hopper => 32,
        }
    }

    /// Maximum resident threads per SM.
    pub fn max_threads_per_sm(self) -> u32 {
        match self {
            Self::Volta | Self::Turing => 2048,
            Self::Ampere | Self::AmpereConsumer | Self::Ada => 2048,
            Self::Hopper => 2048,
        }
    }

    /// Maximum shared memory per SM in bytes.
    pub fn max_shared_mem_per_sm(self) -> u32 {
        match self {
            Self::Volta => 96 * 1024,
            Self::Turing => 64 * 1024,
            Self::Ampere => 164 * 1024,
            Self::AmpereConsumer => 100 * 1024,
            Self::Ada => 100 * 1024,
            Self::Hopper => 228 * 1024,
        }
    }

    /// Register allocation unit size (registers per warp allocation chunk).
    pub fn register_alloc_unit(self) -> u32 {
        REGISTER_ALLOC_GRANULARITY
    }

    /// SM version string.
    pub fn sm_version(self) -> &'static str {
        match self {
            Self::Volta => "sm_70",
            Self::Turing => "sm_75",
            Self::Ampere => "sm_80",
            Self::AmpereConsumer => "sm_86",
            Self::Ada => "sm_89",
            Self::Hopper => "sm_90",
        }
    }
}

impl fmt::Display for GpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Volta => write!(f, "Volta (SM 7.0)"),
            Self::Turing => write!(f, "Turing (SM 7.5)"),
            Self::Ampere => write!(f, "Ampere (SM 8.0)"),
            Self::AmpereConsumer => write!(f, "Ampere (SM 8.6)"),
            Self::Ada => write!(f, "Ada Lovelace (SM 8.9)"),
            Self::Hopper => write!(f, "Hopper (SM 9.0)"),
        }
    }
}

// ── KernelPattern ────────────────────────────────────────────────────

/// Categorized kernel patterns with baseline register usage estimates.
///
/// Different kernel types have characteristic register profiles based on
/// the operations they perform. This enum encodes those patterns for
/// estimation before actual compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelPattern {
    /// Element-wise ops (add, mul, activation). Low register usage.
    Elementwise,
    /// Reduction (sum, max, softmax). Moderate registers for accumulators.
    Reduction,
    /// Dense GEMM / tiled matmul. High register usage for tiles.
    Gemm,
    /// Fused multi-head attention. Very high — QKV + softmax state.
    FusedAttention,
    /// Quantized GEMV (e.g. QK256). High — packed operands + accumulators.
    QuantizedGemv,
    /// Layer normalization / RMS normalization.
    LayerNorm,
    /// Embedding lookup. Low — index + output.
    Embedding,
    /// Convolution 1-D. Moderate — filter window state.
    Conv1d,
    /// Memory copy / transpose. Low — address computation only.
    MemoryOp,
    /// Custom kernel with explicit register estimate.
    Custom(u32),
}

impl KernelPattern {
    /// Estimated registers per thread for this kernel pattern.
    pub fn estimated_registers(self) -> u32 {
        match self {
            Self::Elementwise => 16,
            Self::Reduction => 24,
            Self::Gemm => 64,
            Self::FusedAttention => 96,
            Self::QuantizedGemv => 48,
            Self::LayerNorm => 28,
            Self::Embedding => 12,
            Self::Conv1d => 32,
            Self::MemoryOp => 10,
            Self::Custom(r) => r,
        }
    }

    /// Human-readable pattern name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Elementwise => "elementwise",
            Self::Reduction => "reduction",
            Self::Gemm => "gemm",
            Self::FusedAttention => "fused_attention",
            Self::QuantizedGemv => "quantized_gemv",
            Self::LayerNorm => "layer_norm",
            Self::Embedding => "embedding",
            Self::Conv1d => "conv1d",
            Self::MemoryOp => "memory_op",
            Self::Custom(_) => "custom",
        }
    }
}

impl fmt::Display for KernelPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ── RegisterEstimate ─────────────────────────────────────────────────

/// Per-kernel register usage estimate with breakdown.
#[derive(Debug, Clone)]
pub struct RegisterEstimate {
    /// Kernel identifier or name.
    pub kernel_name: String,
    /// Kernel pattern category.
    pub pattern: KernelPattern,
    /// Estimated registers per thread.
    pub registers_per_thread: u32,
    /// Breakdown of register usage by purpose.
    pub breakdown: RegisterBreakdown,
}

/// Breakdown of register usage by purpose.
#[derive(Debug, Clone, Default)]
pub struct RegisterBreakdown {
    /// Registers for address computation and indexing.
    pub address_regs: u32,
    /// Registers for loop induction variables.
    pub loop_regs: u32,
    /// Registers for accumulators and intermediate results.
    pub accumulator_regs: u32,
    /// Registers for operand staging (loaded from memory).
    pub operand_regs: u32,
    /// Registers for special values (constants, predicates).
    pub special_regs: u32,
}

impl RegisterBreakdown {
    /// Total registers across all categories.
    pub fn total(&self) -> u32 {
        self.address_regs
            + self.loop_regs
            + self.accumulator_regs
            + self.operand_regs
            + self.special_regs
    }
}

/// Estimate register usage for a kernel pattern.
pub fn estimate_registers(
    kernel_name: impl Into<String>,
    pattern: KernelPattern,
) -> RegisterEstimate {
    let name = kernel_name.into();
    let regs = pattern.estimated_registers();
    let breakdown = estimate_breakdown(pattern);
    RegisterEstimate { kernel_name: name, pattern, registers_per_thread: regs, breakdown }
}

fn estimate_breakdown(pattern: KernelPattern) -> RegisterBreakdown {
    let total = pattern.estimated_registers();
    match pattern {
        KernelPattern::Elementwise => RegisterBreakdown {
            address_regs: 4,
            loop_regs: 2,
            accumulator_regs: 4,
            operand_regs: 4,
            special_regs: total.saturating_sub(14),
        },
        KernelPattern::Reduction => RegisterBreakdown {
            address_regs: 4,
            loop_regs: 4,
            accumulator_regs: 8,
            operand_regs: 4,
            special_regs: total.saturating_sub(20),
        },
        KernelPattern::Gemm => RegisterBreakdown {
            address_regs: 8,
            loop_regs: 4,
            accumulator_regs: 32,
            operand_regs: 16,
            special_regs: total.saturating_sub(60),
        },
        KernelPattern::FusedAttention => RegisterBreakdown {
            address_regs: 12,
            loop_regs: 8,
            accumulator_regs: 40,
            operand_regs: 24,
            special_regs: total.saturating_sub(84),
        },
        KernelPattern::QuantizedGemv => RegisterBreakdown {
            address_regs: 8,
            loop_regs: 4,
            accumulator_regs: 16,
            operand_regs: 16,
            special_regs: total.saturating_sub(44),
        },
        KernelPattern::LayerNorm => RegisterBreakdown {
            address_regs: 4,
            loop_regs: 4,
            accumulator_regs: 12,
            operand_regs: 4,
            special_regs: total.saturating_sub(24),
        },
        KernelPattern::Embedding => RegisterBreakdown {
            address_regs: 4,
            loop_regs: 2,
            accumulator_regs: 2,
            operand_regs: 2,
            special_regs: total.saturating_sub(10),
        },
        KernelPattern::Conv1d => RegisterBreakdown {
            address_regs: 6,
            loop_regs: 4,
            accumulator_regs: 12,
            operand_regs: 6,
            special_regs: total.saturating_sub(28),
        },
        KernelPattern::MemoryOp => RegisterBreakdown {
            address_regs: 4,
            loop_regs: 2,
            accumulator_regs: 0,
            operand_regs: 2,
            special_regs: total.saturating_sub(8),
        },
        KernelPattern::Custom(r) => {
            let addr = (r / 5).max(2);
            let acc = r / 3;
            let operand = r / 5;
            let loop_r = 2;
            let used = addr + acc + operand + loop_r;
            RegisterBreakdown {
                address_regs: addr,
                loop_regs: loop_r,
                accumulator_regs: acc,
                operand_regs: operand,
                special_regs: r.saturating_sub(used),
            }
        }
    }
}

// ── SpillDetector ────────────────────────────────────────────────────

/// Severity of register spilling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpillSeverity {
    /// No spilling detected.
    None,
    /// Minor spilling — a few registers, negligible impact.
    Minor,
    /// Moderate spilling — noticeable latency impact.
    Moderate,
    /// Severe spilling — significant performance degradation.
    Severe,
    /// Critical — kernel is essentially memory-bound from spills.
    Critical,
}

impl fmt::Display for SpillSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Minor => write!(f, "minor"),
            Self::Moderate => write!(f, "moderate"),
            Self::Severe => write!(f, "severe"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Result of spill detection analysis.
#[derive(Debug, Clone)]
pub struct SpillAnalysis {
    /// Whether spilling is expected to occur.
    pub spills_detected: bool,
    /// Estimated number of spilled registers.
    pub estimated_spilled_regs: u32,
    /// Severity classification.
    pub severity: SpillSeverity,
    /// Mitigation strategies ordered by effectiveness.
    pub mitigations: Vec<SpillMitigation>,
}

/// A mitigation strategy for register spilling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpillMitigation {
    /// Reduce thread block size to lower register pressure per SM.
    ReduceBlockSize { suggested_threads: u32 },
    /// Apply `__launch_bounds__` to cap register allocation.
    ApplyLaunchBounds { max_threads: u32, min_blocks: u32 },
    /// Use shared memory to stage operands instead of registers.
    UseSharedMemory { bytes_needed: u32 },
    /// Split kernel into smaller fused stages.
    SplitKernel { suggested_stages: u32 },
    /// Move loop-invariant values to constant memory.
    UseConstantMemory { values: u32 },
    /// Apply `maxrregcount` compiler flag.
    LimitRegCount { max_regs: u32 },
}

impl fmt::Display for SpillMitigation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReduceBlockSize { suggested_threads } => {
                write!(f, "reduce block size to {suggested_threads} threads")
            }
            Self::ApplyLaunchBounds { max_threads, min_blocks } => {
                write!(f, "__launch_bounds__({max_threads}, {min_blocks})")
            }
            Self::UseSharedMemory { bytes_needed } => {
                write!(f, "stage {bytes_needed} bytes in shared memory")
            }
            Self::SplitKernel { suggested_stages } => {
                write!(f, "split into {suggested_stages} fused stages")
            }
            Self::UseConstantMemory { values } => {
                write!(f, "move {values} loop-invariant values to __constant__")
            }
            Self::LimitRegCount { max_regs } => {
                write!(f, "-maxrregcount={max_regs}")
            }
        }
    }
}

/// Detect register spilling and suggest mitigations.
pub struct SpillDetector {
    arch: GpuArch,
}

impl SpillDetector {
    /// Create a detector targeting a specific GPU architecture.
    pub fn new(arch: GpuArch) -> Self {
        Self { arch }
    }

    /// Analyze a kernel for potential register spilling.
    pub fn analyze(
        &self,
        registers_per_thread: u32,
        threads_per_block: u32,
        shared_mem_per_block: u32,
    ) -> SpillAnalysis {
        let regs_per_sm = self.arch.registers_per_sm();
        let max_threads = self.arch.max_threads_per_sm();
        let warps_per_block = threads_per_block.div_ceil(WARP_SIZE);

        // Registers allocated per warp (rounded up to allocation granularity).
        let regs_per_warp_raw = registers_per_thread * WARP_SIZE;
        let alloc_unit = self.arch.register_alloc_unit();
        let regs_per_warp = regs_per_warp_raw.div_ceil(alloc_unit) * alloc_unit;

        let regs_per_block = regs_per_warp * warps_per_block;
        let max_blocks_by_regs = if regs_per_block > 0 {
            regs_per_sm / regs_per_block
        } else {
            self.arch.max_blocks_per_sm()
        };

        let max_blocks_by_threads = max_threads / threads_per_block.max(1);
        let max_blocks_by_smem = if shared_mem_per_block > 0 {
            self.arch.max_shared_mem_per_sm() / shared_mem_per_block
        } else {
            self.arch.max_blocks_per_sm()
        };

        let achievable_blocks =
            max_blocks_by_regs.min(max_blocks_by_threads).min(max_blocks_by_smem);

        // Spilling occurs when register demand exceeds what the hardware can
        // provide for at least one block.
        let spills = registers_per_thread > MAX_REGISTERS_PER_THREAD || achievable_blocks == 0;
        let excess = registers_per_thread.saturating_sub(MAX_REGISTERS_PER_THREAD);

        let severity = if !spills {
            if achievable_blocks <= 1 && registers_per_thread > 128 {
                SpillSeverity::Minor
            } else {
                SpillSeverity::None
            }
        } else if excess <= 8 {
            SpillSeverity::Minor
        } else if excess <= 32 {
            SpillSeverity::Moderate
        } else if excess <= 64 {
            SpillSeverity::Severe
        } else {
            SpillSeverity::Critical
        };

        let mut mitigations = Vec::new();
        if spills || achievable_blocks <= 1 {
            // Suggest launch bounds.
            let target_regs = MAX_REGISTERS_PER_THREAD.min(registers_per_thread);
            mitigations.push(SpillMitigation::ApplyLaunchBounds {
                max_threads: threads_per_block,
                min_blocks: 2,
            });

            if registers_per_thread > 64 {
                mitigations.push(SpillMitigation::LimitRegCount { max_regs: target_regs.min(128) });
            }

            if threads_per_block > 128 {
                mitigations.push(SpillMitigation::ReduceBlockSize {
                    suggested_threads: (threads_per_block / 2).max(64),
                });
            }

            if registers_per_thread > 48 {
                let spillable_regs = registers_per_thread.saturating_sub(40);
                mitigations.push(SpillMitigation::UseSharedMemory {
                    bytes_needed: spillable_regs * 4 * threads_per_block,
                });
            }

            if registers_per_thread > 80 {
                mitigations.push(SpillMitigation::SplitKernel { suggested_stages: 2 });
            }

            if registers_per_thread > 32 {
                let movable = (registers_per_thread / 8).min(8);
                mitigations.push(SpillMitigation::UseConstantMemory { values: movable });
            }
        }

        SpillAnalysis {
            spills_detected: spills,
            estimated_spilled_regs: excess,
            severity,
            mitigations,
        }
    }
}

// ── OccupancyTradeoff ────────────────────────────────────────────────

/// A single point in the register-occupancy trade-off space.
#[derive(Debug, Clone, Copy)]
pub struct TradeoffPoint {
    /// Registers per thread at this point.
    pub registers: u32,
    /// Theoretical occupancy (0.0–1.0).
    pub occupancy: f64,
    /// Active warps per SM.
    pub active_warps: u32,
    /// Active blocks per SM.
    pub active_blocks: u32,
}

/// Analyzes the occupancy vs register usage trade-off for a kernel.
pub struct OccupancyTradeoff {
    arch: GpuArch,
}

impl OccupancyTradeoff {
    /// Create a new analyzer for the given architecture.
    pub fn new(arch: GpuArch) -> Self {
        Self { arch }
    }

    /// Sweep register counts and compute occupancy at each point.
    pub fn sweep(
        &self,
        threads_per_block: u32,
        shared_mem_per_block: u32,
        min_regs: u32,
        max_regs: u32,
        step: u32,
    ) -> Vec<TradeoffPoint> {
        let step = step.max(1);
        let mut points = Vec::new();
        let mut regs = min_regs;
        while regs <= max_regs {
            let point = self.evaluate(threads_per_block, shared_mem_per_block, regs);
            points.push(point);
            regs += step;
        }
        points
    }

    /// Evaluate occupancy for a specific register count.
    pub fn evaluate(
        &self,
        threads_per_block: u32,
        shared_mem_per_block: u32,
        registers_per_thread: u32,
    ) -> TradeoffPoint {
        let warps_per_block = threads_per_block.div_ceil(WARP_SIZE);
        let regs_per_sm = self.arch.registers_per_sm();
        let max_warps = self.arch.max_threads_per_sm() / WARP_SIZE;

        // Register-limited blocks.
        let alloc_unit = self.arch.register_alloc_unit();
        let regs_per_warp_raw = registers_per_thread * WARP_SIZE;
        let regs_per_warp = if regs_per_warp_raw == 0 {
            0
        } else {
            regs_per_warp_raw.div_ceil(alloc_unit) * alloc_unit
        };
        let regs_per_block = regs_per_warp * warps_per_block;
        let blocks_by_regs = if regs_per_block > 0 {
            regs_per_sm / regs_per_block
        } else {
            self.arch.max_blocks_per_sm()
        };

        // Thread-limited blocks.
        let blocks_by_threads = self.arch.max_threads_per_sm() / threads_per_block.max(1);

        // Shared-memory-limited blocks.
        let blocks_by_smem = if shared_mem_per_block > 0 {
            self.arch.max_shared_mem_per_sm() / shared_mem_per_block
        } else {
            self.arch.max_blocks_per_sm()
        };

        let blocks_by_max = self.arch.max_blocks_per_sm();

        let active_blocks =
            blocks_by_regs.min(blocks_by_threads).min(blocks_by_smem).min(blocks_by_max);

        let active_warps = active_blocks * warps_per_block;
        let occupancy =
            if max_warps > 0 { (active_warps as f64 / max_warps as f64).min(1.0) } else { 0.0 };

        TradeoffPoint { registers: registers_per_thread, occupancy, active_warps, active_blocks }
    }

    /// Find the register count that maximizes occupancy.
    pub fn optimal_registers(
        &self,
        threads_per_block: u32,
        shared_mem_per_block: u32,
    ) -> TradeoffPoint {
        let points = self.sweep(threads_per_block, shared_mem_per_block, 8, 255, 1);
        points
            .into_iter()
            .max_by(|a, b| {
                a.occupancy.partial_cmp(&b.occupancy).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(TradeoffPoint {
                registers: 32,
                occupancy: 0.0,
                active_warps: 0,
                active_blocks: 0,
            })
    }

    /// Find register thresholds where occupancy changes (cliff points).
    pub fn find_cliffs(
        &self,
        threads_per_block: u32,
        shared_mem_per_block: u32,
    ) -> Vec<TradeoffPoint> {
        let points = self.sweep(threads_per_block, shared_mem_per_block, 8, 255, 1);
        let mut cliffs = Vec::new();
        for window in points.windows(2) {
            if (window[0].occupancy - window[1].occupancy).abs() > 1e-6 {
                cliffs.push(window[1]);
            }
        }
        cliffs
    }
}

// ── LaunchBoundsCalculator ───────────────────────────────────────────

/// Computed `__launch_bounds__` parameters for a CUDA kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchBounds {
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Minimum blocks per SM for occupancy.
    pub min_blocks_per_sm: u32,
}

impl LaunchBounds {
    /// Generate the CUDA `__launch_bounds__` attribute string.
    pub fn as_cuda_attr(&self) -> String {
        format!("__launch_bounds__({}, {})", self.max_threads_per_block, self.min_blocks_per_sm)
    }
}

impl fmt::Display for LaunchBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "__launch_bounds__({}, {})", self.max_threads_per_block, self.min_blocks_per_sm)
    }
}

/// Calculator for `__launch_bounds__` parameters.
pub struct LaunchBoundsCalculator {
    arch: GpuArch,
}

impl LaunchBoundsCalculator {
    /// Create a calculator for the given architecture.
    pub fn new(arch: GpuArch) -> Self {
        Self { arch }
    }

    /// Compute launch bounds for a kernel with known register/shared-memory usage.
    pub fn compute(
        &self,
        registers_per_thread: u32,
        threads_per_block: u32,
        shared_mem_per_block: u32,
    ) -> LaunchBounds {
        let tradeoff = OccupancyTradeoff::new(self.arch);
        let point =
            tradeoff.evaluate(threads_per_block, shared_mem_per_block, registers_per_thread);

        let min_blocks = point.active_blocks.max(1);

        LaunchBounds { max_threads_per_block: threads_per_block, min_blocks_per_sm: min_blocks }
    }

    /// Compute launch bounds from a kernel pattern (auto-estimated registers).
    pub fn compute_for_pattern(
        &self,
        pattern: KernelPattern,
        threads_per_block: u32,
        shared_mem_per_block: u32,
    ) -> LaunchBounds {
        self.compute(pattern.estimated_registers(), threads_per_block, shared_mem_per_block)
    }

    /// Compute launch bounds that maximize occupancy within a register budget.
    pub fn maximize_occupancy(
        &self,
        max_registers: u32,
        shared_mem_per_block: u32,
    ) -> LaunchBounds {
        let tradeoff = OccupancyTradeoff::new(self.arch);
        let mut best = LaunchBounds { max_threads_per_block: 64, min_blocks_per_sm: 1 };
        let mut best_occupancy = 0.0_f64;

        for threads in [64, 128, 192, 256, 384, 512, 768, 1024] {
            let point = tradeoff.evaluate(threads, shared_mem_per_block, max_registers);
            if point.occupancy > best_occupancy {
                best_occupancy = point.occupancy;
                best = LaunchBounds {
                    max_threads_per_block: threads,
                    min_blocks_per_sm: point.active_blocks.max(1),
                };
            }
        }
        best
    }
}

// ── RegisterPartitioner ──────────────────────────────────────────────

/// How register resources are partitioned across warps on an SM.
#[derive(Debug, Clone)]
pub struct RegisterPartition {
    /// Registers allocated per warp (after granularity rounding).
    pub regs_per_warp: u32,
    /// Total warps that can be resident.
    pub resident_warps: u32,
    /// Total registers consumed by resident warps.
    pub total_regs_consumed: u32,
    /// Registers left unused (wasted due to granularity).
    pub wasted_regs: u32,
    /// Register file utilization (0.0–1.0).
    pub utilization: f64,
}

/// Analyzes register file partitioning across warps.
pub struct RegisterPartitioner {
    arch: GpuArch,
}

impl RegisterPartitioner {
    /// Create a partitioner for the given architecture.
    pub fn new(arch: GpuArch) -> Self {
        Self { arch }
    }

    /// Compute register partitioning for a kernel configuration.
    pub fn partition(
        &self,
        registers_per_thread: u32,
        threads_per_block: u32,
    ) -> RegisterPartition {
        let regs_per_sm = self.arch.registers_per_sm();
        let alloc_unit = self.arch.register_alloc_unit();

        let regs_per_warp_raw = registers_per_thread * WARP_SIZE;
        let regs_per_warp = if regs_per_warp_raw == 0 {
            0
        } else {
            regs_per_warp_raw.div_ceil(alloc_unit) * alloc_unit
        };

        let warps_per_block = threads_per_block.div_ceil(WARP_SIZE);
        let regs_per_block = regs_per_warp * warps_per_block;

        let blocks = if regs_per_block > 0 {
            (regs_per_sm / regs_per_block).min(self.arch.max_blocks_per_sm())
        } else {
            self.arch.max_blocks_per_sm()
        };

        let max_blocks_by_threads = self.arch.max_threads_per_sm() / threads_per_block.max(1);
        let effective_blocks = blocks.min(max_blocks_by_threads);

        let resident_warps = effective_blocks * warps_per_block;
        let total_consumed = resident_warps * regs_per_warp;
        let wasted = total_consumed - (resident_warps * registers_per_thread * WARP_SIZE);
        let utilization =
            if regs_per_sm > 0 { total_consumed as f64 / regs_per_sm as f64 } else { 0.0 };

        RegisterPartition {
            regs_per_warp,
            resident_warps,
            total_regs_consumed: total_consumed,
            wasted_regs: wasted,
            utilization,
        }
    }

    /// Find the most register-efficient thread block size.
    pub fn optimal_block_size(&self, registers_per_thread: u32) -> u32 {
        let mut best_threads = 64_u32;
        let mut best_util = 0.0_f64;

        for threads in [64, 128, 192, 256, 384, 512, 768, 1024] {
            let part = self.partition(registers_per_thread, threads);
            if part.utilization > best_util {
                best_util = part.utilization;
                best_threads = threads;
            }
        }
        best_threads
    }
}

// ── LiveRangeAnalyzer ────────────────────────────────────────────────

/// A variable's live range within a kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRange {
    /// Variable name or identifier.
    pub name: String,
    /// Instruction index where the variable is first defined.
    pub def_point: u32,
    /// Instruction index where the variable is last used.
    pub last_use: u32,
    /// Number of instructions the variable is live.
    pub span: u32,
    /// Whether the variable interferes with high-pressure regions.
    pub in_high_pressure_region: bool,
}

impl LiveRange {
    /// Create a new live range.
    pub fn new(name: impl Into<String>, def_point: u32, last_use: u32) -> Self {
        let span = last_use.saturating_sub(def_point);
        Self { name: name.into(), def_point, last_use, span, in_high_pressure_region: false }
    }

    /// Check if this range overlaps with another.
    pub fn overlaps(&self, other: &LiveRange) -> bool {
        self.def_point <= other.last_use && other.def_point <= self.last_use
    }
}

/// Result of live range analysis.
#[derive(Debug, Clone)]
pub struct LiveRangeReport {
    /// All variable live ranges.
    pub ranges: Vec<LiveRange>,
    /// Peak number of simultaneously live variables.
    pub peak_pressure: u32,
    /// Instruction index where peak pressure occurs.
    pub peak_point: u32,
    /// Variables that are candidates for spilling (long ranges in high pressure).
    pub spill_candidates: Vec<String>,
    /// Variables that could be rematerialized instead of spilled.
    pub remat_candidates: Vec<String>,
}

/// Analyzes variable live ranges for register optimization opportunities.
pub struct LiveRangeAnalyzer {
    ranges: Vec<LiveRange>,
    total_instructions: u32,
}

impl LiveRangeAnalyzer {
    /// Create a new analyzer.
    pub fn new(total_instructions: u32) -> Self {
        Self { ranges: Vec::new(), total_instructions }
    }

    /// Add a variable live range.
    pub fn add_range(&mut self, range: LiveRange) {
        self.ranges.push(range);
    }

    /// Add a range from components.
    pub fn add(&mut self, name: impl Into<String>, def_point: u32, last_use: u32) {
        self.add_range(LiveRange::new(name, def_point, last_use));
    }

    /// Run the analysis and produce a report.
    pub fn analyze(&self) -> LiveRangeReport {
        if self.ranges.is_empty() {
            return LiveRangeReport {
                ranges: Vec::new(),
                peak_pressure: 0,
                peak_point: 0,
                spill_candidates: Vec::new(),
                remat_candidates: Vec::new(),
            };
        }

        // Sweep-line to find peak pressure.
        let mut peak_pressure: u32 = 0;
        let mut peak_point: u32 = 0;

        // Build pressure at each instruction point.
        let mut pressure_at: HashMap<u32, u32> = HashMap::new();
        for r in &self.ranges {
            for i in r.def_point..=r.last_use.min(self.total_instructions) {
                *pressure_at.entry(i).or_insert(0) += 1;
            }
        }
        for (&point, &pressure) in &pressure_at {
            if pressure > peak_pressure {
                peak_pressure = pressure;
                peak_point = point;
            }
        }

        // Mark ranges in high-pressure regions and find spill candidates.
        let high_threshold = (peak_pressure * 3) / 4;
        let mut ranges = self.ranges.clone();
        let mut spill_candidates = Vec::new();
        let mut remat_candidates = Vec::new();

        for range in &mut ranges {
            // Check if this range covers any high-pressure point.
            let in_high = (range.def_point..=range.last_use)
                .any(|i| pressure_at.get(&i).copied().unwrap_or(0) >= high_threshold);
            range.in_high_pressure_region = in_high;

            if in_high && range.span > self.total_instructions / 4 {
                spill_candidates.push(range.name.clone());
            }

            // Short-lived values with simple definitions are remat candidates.
            if in_high && range.span <= 2 {
                remat_candidates.push(range.name.clone());
            }
        }

        LiveRangeReport { ranges, peak_pressure, peak_point, spill_candidates, remat_candidates }
    }
}

// ── CompilerHintGenerator ────────────────────────────────────────────

/// A compiler hint for register allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilerHint {
    /// `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`.
    LaunchBounds { max_threads: u32, min_blocks: u32 },
    /// `-maxrregcount=N` compiler flag.
    MaxRegCount(u32),
    /// `__restrict__` pointer qualifier recommendation.
    RestrictPointers { pointer_names: Vec<String> },
    /// `#pragma unroll N` loop annotation.
    PragmaUnroll { loop_label: String, factor: u32 },
    /// Recommend `__forceinline__` for a function.
    ForceInline { function_name: String },
    /// Recommend moving data to `__constant__` memory.
    ConstantMemory { variable_names: Vec<String> },
}

impl fmt::Display for CompilerHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LaunchBounds { max_threads, min_blocks } => {
                write!(f, "__launch_bounds__({max_threads}, {min_blocks})")
            }
            Self::MaxRegCount(n) => write!(f, "-maxrregcount={n}"),
            Self::RestrictPointers { pointer_names } => {
                write!(f, "__restrict__ on: {}", pointer_names.join(", "))
            }
            Self::PragmaUnroll { loop_label, factor } => {
                write!(f, "#pragma unroll {factor} // {loop_label}")
            }
            Self::ForceInline { function_name } => {
                write!(f, "__forceinline__ {function_name}")
            }
            Self::ConstantMemory { variable_names } => {
                write!(f, "__constant__ for: {}", variable_names.join(", "))
            }
        }
    }
}

/// Configuration for compiler hint generation.
#[derive(Debug, Clone)]
pub struct HintConfig {
    /// Target architecture.
    pub arch: GpuArch,
    /// Kernel pattern.
    pub pattern: KernelPattern,
    /// Threads per block.
    pub threads_per_block: u32,
    /// Dynamic shared memory per block (bytes).
    pub shared_mem_per_block: u32,
    /// Known register count (if available from profiling).
    pub known_registers: Option<u32>,
    /// Pointer parameter names for `__restrict__` hints.
    pub pointer_params: Vec<String>,
    /// Loop labels for unroll hints.
    pub loops: Vec<(String, u32)>,
    /// Candidate constant variables.
    pub constant_candidates: Vec<String>,
    /// Inline function candidates.
    pub inline_candidates: Vec<String>,
}

impl HintConfig {
    /// Create a minimal config.
    pub fn new(arch: GpuArch, pattern: KernelPattern, threads_per_block: u32) -> Self {
        Self {
            arch,
            pattern,
            threads_per_block,
            shared_mem_per_block: 0,
            known_registers: None,
            pointer_params: Vec::new(),
            loops: Vec::new(),
            constant_candidates: Vec::new(),
            inline_candidates: Vec::new(),
        }
    }
}

/// Generates compiler hints to optimize register allocation.
pub struct CompilerHintGenerator;

impl CompilerHintGenerator {
    /// Generate all applicable hints for the given configuration.
    pub fn generate(config: &HintConfig) -> Vec<CompilerHint> {
        let regs = config.known_registers.unwrap_or_else(|| config.pattern.estimated_registers());

        let mut hints = Vec::new();

        // Launch bounds.
        let calc = LaunchBoundsCalculator::new(config.arch);
        let bounds = calc.compute(regs, config.threads_per_block, config.shared_mem_per_block);
        hints.push(CompilerHint::LaunchBounds {
            max_threads: bounds.max_threads_per_block,
            min_blocks: bounds.min_blocks_per_sm,
        });

        // Max reg count if register pressure is high.
        if regs > 64 {
            hints.push(CompilerHint::MaxRegCount(regs.min(128)));
        }

        // Restrict pointers.
        if !config.pointer_params.is_empty() {
            hints.push(CompilerHint::RestrictPointers {
                pointer_names: config.pointer_params.clone(),
            });
        }

        // Unroll annotations.
        for (label, factor) in &config.loops {
            hints.push(CompilerHint::PragmaUnroll { loop_label: label.clone(), factor: *factor });
        }

        // Constant memory.
        if !config.constant_candidates.is_empty() && regs > 32 {
            hints.push(CompilerHint::ConstantMemory {
                variable_names: config.constant_candidates.clone(),
            });
        }

        // Force-inline for small helpers that inflate register pressure.
        for func in &config.inline_candidates {
            hints.push(CompilerHint::ForceInline { function_name: func.clone() });
        }

        hints
    }

    /// Generate a CUDA source annotation block from hints.
    pub fn emit_cuda_annotations(hints: &[CompilerHint]) -> String {
        let mut lines = Vec::new();
        lines.push("// === Register optimization hints ===".to_string());
        for hint in hints {
            match hint {
                CompilerHint::LaunchBounds { max_threads, min_blocks } => {
                    lines.push(format!("// Apply: __launch_bounds__({max_threads}, {min_blocks})"));
                }
                CompilerHint::MaxRegCount(n) => {
                    lines.push(format!("// Compile with: -maxrregcount={n}"));
                }
                CompilerHint::RestrictPointers { pointer_names } => {
                    for name in pointer_names {
                        lines.push(format!("// Add __restrict__ to parameter: {name}"));
                    }
                }
                CompilerHint::PragmaUnroll { loop_label, factor } => {
                    lines.push(format!("// Before loop '{loop_label}': #pragma unroll {factor}"));
                }
                CompilerHint::ForceInline { function_name } => {
                    lines.push(format!("// Mark as __forceinline__: {function_name}"));
                }
                CompilerHint::ConstantMemory { variable_names } => {
                    for name in variable_names {
                        lines.push(format!("// Move to __constant__: {name}"));
                    }
                }
            }
        }
        lines.join("\n")
    }
}

// ── Convenience: full analysis pipeline ──────────────────────────────

/// Complete register analysis result for a kernel.
#[derive(Debug, Clone)]
pub struct RegisterAnalysis {
    /// Register estimate.
    pub estimate: RegisterEstimate,
    /// Spill analysis.
    pub spill: SpillAnalysis,
    /// Occupancy trade-off at the estimated register count.
    pub occupancy: TradeoffPoint,
    /// Recommended launch bounds.
    pub launch_bounds: LaunchBounds,
    /// Register partition info.
    pub partition: RegisterPartition,
    /// Generated compiler hints.
    pub hints: Vec<CompilerHint>,
}

/// Run a complete register pressure analysis for a kernel.
pub fn analyze_kernel(
    kernel_name: impl Into<String>,
    pattern: KernelPattern,
    arch: GpuArch,
    threads_per_block: u32,
    shared_mem_per_block: u32,
) -> Result<RegisterAnalysis> {
    let name = kernel_name.into();

    if threads_per_block == 0 || threads_per_block > 1024 {
        return Err(KernelError::InvalidArguments {
            reason: format!("threads_per_block must be in 1..=1024, got {threads_per_block}"),
        }
        .into());
    }

    let estimate = estimate_registers(&name, pattern);
    let regs = estimate.registers_per_thread;

    let detector = SpillDetector::new(arch);
    let spill = detector.analyze(regs, threads_per_block, shared_mem_per_block);

    let tradeoff = OccupancyTradeoff::new(arch);
    let occupancy = tradeoff.evaluate(threads_per_block, shared_mem_per_block, regs);

    let calc = LaunchBoundsCalculator::new(arch);
    let launch_bounds = calc.compute(regs, threads_per_block, shared_mem_per_block);

    let partitioner = RegisterPartitioner::new(arch);
    let partition = partitioner.partition(regs, threads_per_block);

    let hint_config = HintConfig::new(arch, pattern, threads_per_block);
    let hints = CompilerHintGenerator::generate(&hint_config);

    Ok(RegisterAnalysis { estimate, spill, occupancy, launch_bounds, partition, hints })
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- GpuArch --

    #[test]
    fn gpu_arch_registers_per_sm() {
        assert_eq!(GpuArch::Ampere.registers_per_sm(), 65_536);
        assert_eq!(GpuArch::Hopper.registers_per_sm(), 65_536);
        assert_eq!(GpuArch::Volta.registers_per_sm(), 65_536);
    }

    #[test]
    fn gpu_arch_max_threads_per_sm() {
        assert_eq!(GpuArch::Ampere.max_threads_per_sm(), 2048);
        assert_eq!(GpuArch::Turing.max_threads_per_sm(), 2048);
    }

    #[test]
    fn gpu_arch_max_blocks_per_sm() {
        assert_eq!(GpuArch::Ampere.max_blocks_per_sm(), 32);
        assert_eq!(GpuArch::Ada.max_blocks_per_sm(), 32);
    }

    #[test]
    fn gpu_arch_shared_mem_varies_by_gen() {
        assert!(GpuArch::Hopper.max_shared_mem_per_sm() > GpuArch::Turing.max_shared_mem_per_sm());
        assert!(GpuArch::Ampere.max_shared_mem_per_sm() > GpuArch::Turing.max_shared_mem_per_sm());
    }

    #[test]
    fn gpu_arch_register_alloc_unit() {
        assert_eq!(GpuArch::Ampere.register_alloc_unit(), REGISTER_ALLOC_GRANULARITY);
    }

    #[test]
    fn gpu_arch_sm_version_strings() {
        assert_eq!(GpuArch::Volta.sm_version(), "sm_70");
        assert_eq!(GpuArch::Turing.sm_version(), "sm_75");
        assert_eq!(GpuArch::Ampere.sm_version(), "sm_80");
        assert_eq!(GpuArch::AmpereConsumer.sm_version(), "sm_86");
        assert_eq!(GpuArch::Ada.sm_version(), "sm_89");
        assert_eq!(GpuArch::Hopper.sm_version(), "sm_90");
    }

    #[test]
    fn gpu_arch_display() {
        let s = format!("{}", GpuArch::Ampere);
        assert!(s.contains("Ampere"));
        assert!(s.contains("SM 8.0"));
    }

    // -- KernelPattern --

    #[test]
    fn kernel_pattern_elementwise_low_regs() {
        assert!(KernelPattern::Elementwise.estimated_registers() <= 20);
    }

    #[test]
    fn kernel_pattern_gemm_high_regs() {
        assert!(KernelPattern::Gemm.estimated_registers() >= 48);
    }

    #[test]
    fn kernel_pattern_fused_attention_very_high() {
        assert!(KernelPattern::FusedAttention.estimated_registers() >= 80);
    }

    #[test]
    fn kernel_pattern_quantized_gemv_moderate() {
        let regs = KernelPattern::QuantizedGemv.estimated_registers();
        assert!(regs >= 32 && regs <= 80);
    }

    #[test]
    fn kernel_pattern_custom_exact() {
        assert_eq!(KernelPattern::Custom(42).estimated_registers(), 42);
    }

    #[test]
    fn kernel_pattern_ordering() {
        assert!(
            KernelPattern::Elementwise.estimated_registers()
                < KernelPattern::Gemm.estimated_registers()
        );
        assert!(
            KernelPattern::Gemm.estimated_registers()
                < KernelPattern::FusedAttention.estimated_registers()
        );
    }

    #[test]
    fn kernel_pattern_names() {
        assert_eq!(KernelPattern::Elementwise.name(), "elementwise");
        assert_eq!(KernelPattern::Gemm.name(), "gemm");
        assert_eq!(KernelPattern::FusedAttention.name(), "fused_attention");
        assert_eq!(KernelPattern::Custom(10).name(), "custom");
    }

    #[test]
    fn kernel_pattern_display() {
        assert_eq!(format!("{}", KernelPattern::Reduction), "reduction");
    }

    #[test]
    fn kernel_pattern_memory_op_minimal() {
        assert!(KernelPattern::MemoryOp.estimated_registers() <= 16);
    }

    #[test]
    fn kernel_pattern_embedding_low() {
        assert!(KernelPattern::Embedding.estimated_registers() <= 16);
    }

    #[test]
    fn kernel_pattern_conv1d_moderate() {
        let regs = KernelPattern::Conv1d.estimated_registers();
        assert!(regs >= 20 && regs <= 48);
    }

    #[test]
    fn kernel_pattern_layer_norm_moderate() {
        let regs = KernelPattern::LayerNorm.estimated_registers();
        assert!(regs >= 20 && regs <= 40);
    }

    // -- RegisterEstimate / RegisterBreakdown --

    #[test]
    fn estimate_registers_matches_pattern() {
        let est = estimate_registers("test_kernel", KernelPattern::Gemm);
        assert_eq!(est.registers_per_thread, KernelPattern::Gemm.estimated_registers());
        assert_eq!(est.kernel_name, "test_kernel");
    }

    #[test]
    fn breakdown_total_matches_estimate() {
        for pattern in [
            KernelPattern::Elementwise,
            KernelPattern::Reduction,
            KernelPattern::Gemm,
            KernelPattern::FusedAttention,
            KernelPattern::QuantizedGemv,
            KernelPattern::LayerNorm,
            KernelPattern::Embedding,
            KernelPattern::Conv1d,
            KernelPattern::MemoryOp,
        ] {
            let est = estimate_registers("k", pattern);
            assert_eq!(
                est.breakdown.total(),
                est.registers_per_thread,
                "breakdown mismatch for {pattern}"
            );
        }
    }

    #[test]
    fn breakdown_custom_total_matches() {
        let est = estimate_registers("k", KernelPattern::Custom(50));
        assert_eq!(est.breakdown.total(), 50);
    }

    #[test]
    fn breakdown_gemm_accumulator_dominant() {
        let est = estimate_registers("k", KernelPattern::Gemm);
        assert!(est.breakdown.accumulator_regs >= est.breakdown.address_regs);
        assert!(est.breakdown.accumulator_regs >= est.breakdown.loop_regs);
    }

    #[test]
    fn breakdown_elementwise_small_accumulators() {
        let est = estimate_registers("k", KernelPattern::Elementwise);
        assert!(est.breakdown.accumulator_regs <= 8);
    }

    // -- SpillDetector --

    #[test]
    fn spill_detector_no_spill_low_regs() {
        let det = SpillDetector::new(GpuArch::Ampere);
        let result = det.analyze(32, 256, 0);
        assert!(!result.spills_detected);
        assert_eq!(result.severity, SpillSeverity::None);
    }

    #[test]
    fn spill_detector_spill_on_excess_regs() {
        let det = SpillDetector::new(GpuArch::Ampere);
        // 256 registers per thread exceeds MAX_REGISTERS_PER_THREAD (255).
        let result = det.analyze(256, 256, 0);
        assert!(result.spills_detected);
        assert!(result.severity >= SpillSeverity::Minor);
    }

    #[test]
    fn spill_detector_critical_for_extreme_regs() {
        let det = SpillDetector::new(GpuArch::Ampere);
        let result = det.analyze(400, 256, 0);
        assert!(result.spills_detected);
        assert_eq!(result.severity, SpillSeverity::Critical);
    }

    #[test]
    fn spill_detector_mitigations_present_on_spill() {
        let det = SpillDetector::new(GpuArch::Ampere);
        let result = det.analyze(256, 256, 0);
        assert!(!result.mitigations.is_empty());
    }

    #[test]
    fn spill_detector_no_mitigations_when_fine() {
        let det = SpillDetector::new(GpuArch::Ampere);
        let result = det.analyze(16, 256, 0);
        assert!(result.mitigations.is_empty());
    }

    #[test]
    fn spill_severity_ordering() {
        assert!(SpillSeverity::None < SpillSeverity::Minor);
        assert!(SpillSeverity::Minor < SpillSeverity::Moderate);
        assert!(SpillSeverity::Moderate < SpillSeverity::Severe);
        assert!(SpillSeverity::Severe < SpillSeverity::Critical);
    }

    #[test]
    fn spill_severity_display() {
        assert_eq!(format!("{}", SpillSeverity::None), "none");
        assert_eq!(format!("{}", SpillSeverity::Critical), "critical");
    }

    #[test]
    fn spill_detector_shared_mem_pressure() {
        let det = SpillDetector::new(GpuArch::Ampere);
        // Huge shared memory but normal registers — should not spill.
        let result = det.analyze(32, 256, 48 * 1024);
        assert!(!result.spills_detected);
    }

    #[test]
    fn spill_mitigation_launch_bounds_present() {
        let det = SpillDetector::new(GpuArch::Ampere);
        let result = det.analyze(256, 256, 0);
        assert!(
            result
                .mitigations
                .iter()
                .any(|m| matches!(m, SpillMitigation::ApplyLaunchBounds { .. }))
        );
    }

    #[test]
    fn spill_mitigation_reduce_block_suggested() {
        let det = SpillDetector::new(GpuArch::Ampere);
        let result = det.analyze(256, 256, 0);
        assert!(
            result.mitigations.iter().any(|m| matches!(m, SpillMitigation::ReduceBlockSize { .. }))
        );
    }

    #[test]
    fn spill_mitigation_display() {
        let m = SpillMitigation::LimitRegCount { max_regs: 64 };
        assert_eq!(format!("{m}"), "-maxrregcount=64");

        let m2 = SpillMitigation::ApplyLaunchBounds { max_threads: 256, min_blocks: 2 };
        assert!(format!("{m2}").contains("__launch_bounds__"));
    }

    #[test]
    fn spill_mitigation_shared_mem_for_high_regs() {
        let det = SpillDetector::new(GpuArch::Ampere);
        let result = det.analyze(256, 256, 0);
        assert!(
            result.mitigations.iter().any(|m| matches!(m, SpillMitigation::UseSharedMemory { .. }))
        );
    }

    #[test]
    fn spill_mitigation_constant_mem_for_moderate_regs() {
        let det = SpillDetector::new(GpuArch::Ampere);
        let result = det.analyze(256, 256, 0);
        assert!(
            result
                .mitigations
                .iter()
                .any(|m| matches!(m, SpillMitigation::UseConstantMemory { .. }))
        );
    }

    #[test]
    fn spill_mitigation_split_kernel_for_very_high_regs() {
        let det = SpillDetector::new(GpuArch::Ampere);
        let result = det.analyze(400, 256, 0);
        assert!(
            result.mitigations.iter().any(|m| matches!(m, SpillMitigation::SplitKernel { .. }))
        );
    }

    // -- OccupancyTradeoff --

    #[test]
    fn occupancy_low_regs_high_occupancy() {
        let t = OccupancyTradeoff::new(GpuArch::Ampere);
        let point = t.evaluate(256, 0, 16);
        assert!(point.occupancy > 0.5);
    }

    #[test]
    fn occupancy_high_regs_low_occupancy() {
        let t = OccupancyTradeoff::new(GpuArch::Ampere);
        let point = t.evaluate(256, 0, 128);
        let low_point = t.evaluate(256, 0, 16);
        assert!(point.occupancy <= low_point.occupancy);
    }

    #[test]
    fn occupancy_sweep_returns_points() {
        let t = OccupancyTradeoff::new(GpuArch::Ampere);
        let points = t.sweep(256, 0, 16, 128, 16);
        assert!(!points.is_empty());
        assert_eq!(points.first().unwrap().registers, 16);
    }

    #[test]
    fn occupancy_sweep_monotone_decreasing_or_flat() {
        let t = OccupancyTradeoff::new(GpuArch::Ampere);
        let points = t.sweep(256, 0, 8, 255, 1);
        for window in points.windows(2) {
            assert!(
                window[0].occupancy >= window[1].occupancy - 1e-9,
                "occupancy increased at {} → {} regs",
                window[0].registers,
                window[1].registers
            );
        }
    }

    #[test]
    fn occupancy_optimal_returns_valid() {
        let t = OccupancyTradeoff::new(GpuArch::Ampere);
        let opt = t.optimal_registers(256, 0);
        assert!(opt.registers >= 8 && opt.registers <= 255);
        assert!(opt.occupancy > 0.0);
    }

    #[test]
    fn occupancy_cliffs_non_empty() {
        let t = OccupancyTradeoff::new(GpuArch::Ampere);
        let cliffs = t.find_cliffs(256, 0);
        assert!(!cliffs.is_empty(), "should have at least one occupancy cliff");
    }

    #[test]
    fn occupancy_zero_regs_max_occupancy() {
        let t = OccupancyTradeoff::new(GpuArch::Ampere);
        // Zero registers (unrealistic but valid edge case).
        let point = t.evaluate(256, 0, 0);
        assert!(point.occupancy > 0.0);
    }

    #[test]
    fn occupancy_shared_mem_reduces_blocks() {
        let t = OccupancyTradeoff::new(GpuArch::Ampere);
        let no_smem = t.evaluate(256, 0, 32);
        let with_smem = t.evaluate(256, 48 * 1024, 32);
        assert!(with_smem.active_blocks <= no_smem.active_blocks);
    }

    #[test]
    fn occupancy_max_regs_very_low() {
        let t = OccupancyTradeoff::new(GpuArch::Ampere);
        let point = t.evaluate(256, 0, 255);
        assert!(point.active_blocks <= 2);
    }

    // -- LaunchBoundsCalculator --

    #[test]
    fn launch_bounds_basic() {
        let calc = LaunchBoundsCalculator::new(GpuArch::Ampere);
        let bounds = calc.compute(32, 256, 0);
        assert_eq!(bounds.max_threads_per_block, 256);
        assert!(bounds.min_blocks_per_sm >= 1);
    }

    #[test]
    fn launch_bounds_from_pattern() {
        let calc = LaunchBoundsCalculator::new(GpuArch::Ampere);
        let bounds = calc.compute_for_pattern(KernelPattern::Elementwise, 256, 0);
        assert_eq!(bounds.max_threads_per_block, 256);
        assert!(bounds.min_blocks_per_sm >= 2);
    }

    #[test]
    fn launch_bounds_cuda_attr_format() {
        let bounds = LaunchBounds { max_threads_per_block: 256, min_blocks_per_sm: 4 };
        assert_eq!(bounds.as_cuda_attr(), "__launch_bounds__(256, 4)");
    }

    #[test]
    fn launch_bounds_display() {
        let bounds = LaunchBounds { max_threads_per_block: 128, min_blocks_per_sm: 8 };
        assert_eq!(format!("{bounds}"), "__launch_bounds__(128, 8)");
    }

    #[test]
    fn launch_bounds_maximize_occupancy() {
        let calc = LaunchBoundsCalculator::new(GpuArch::Ampere);
        let bounds = calc.maximize_occupancy(32, 0);
        assert!(bounds.max_threads_per_block >= 64);
        assert!(bounds.min_blocks_per_sm >= 1);
    }

    #[test]
    fn launch_bounds_high_reg_fewer_blocks() {
        let calc = LaunchBoundsCalculator::new(GpuArch::Ampere);
        let low_reg = calc.compute(16, 256, 0);
        let high_reg = calc.compute(128, 256, 0);
        assert!(high_reg.min_blocks_per_sm <= low_reg.min_blocks_per_sm);
    }

    // -- RegisterPartitioner --

    #[test]
    fn partition_basic() {
        let part = RegisterPartitioner::new(GpuArch::Ampere);
        let result = part.partition(32, 256);
        assert!(result.regs_per_warp > 0);
        assert!(result.resident_warps > 0);
        assert!(result.utilization > 0.0 && result.utilization <= 1.0);
    }

    #[test]
    fn partition_granularity_rounding() {
        let part = RegisterPartitioner::new(GpuArch::Ampere);
        let result = part.partition(33, 256);
        // Allocation must be a multiple of the granularity.
        assert_eq!(result.regs_per_warp % REGISTER_ALLOC_GRANULARITY, 0);
    }

    #[test]
    fn partition_wasted_regs_nonnegative() {
        let part = RegisterPartitioner::new(GpuArch::Ampere);
        for regs in [16, 32, 48, 64, 96, 128] {
            let result = part.partition(regs, 256);
            assert!(result.total_regs_consumed >= result.wasted_regs);
        }
    }

    #[test]
    fn partition_low_regs_high_utilization() {
        let part = RegisterPartitioner::new(GpuArch::Ampere);
        let result = part.partition(16, 256);
        assert!(result.utilization > 0.3);
    }

    #[test]
    fn partition_optimal_block_size() {
        let part = RegisterPartitioner::new(GpuArch::Ampere);
        let optimal = part.optimal_block_size(32);
        assert!(optimal >= 64 && optimal <= 1024);
    }

    #[test]
    fn partition_zero_regs() {
        let part = RegisterPartitioner::new(GpuArch::Ampere);
        let result = part.partition(0, 256);
        assert_eq!(result.regs_per_warp, 0);
    }

    // -- LiveRangeAnalyzer --

    #[test]
    fn live_range_basic() {
        let r = LiveRange::new("x", 0, 10);
        assert_eq!(r.span, 10);
        assert!(!r.in_high_pressure_region);
    }

    #[test]
    fn live_range_overlap() {
        let a = LiveRange::new("a", 0, 10);
        let b = LiveRange::new("b", 5, 15);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn live_range_no_overlap() {
        let a = LiveRange::new("a", 0, 5);
        let b = LiveRange::new("b", 6, 10);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn live_range_adjacent_overlaps() {
        let a = LiveRange::new("a", 0, 5);
        let b = LiveRange::new("b", 5, 10);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn live_range_analyzer_empty() {
        let analyzer = LiveRangeAnalyzer::new(100);
        let report = analyzer.analyze();
        assert_eq!(report.peak_pressure, 0);
        assert!(report.ranges.is_empty());
    }

    #[test]
    fn live_range_analyzer_single_var() {
        let mut analyzer = LiveRangeAnalyzer::new(20);
        analyzer.add("x", 0, 10);
        let report = analyzer.analyze();
        assert_eq!(report.peak_pressure, 1);
        assert_eq!(report.ranges.len(), 1);
    }

    #[test]
    fn live_range_analyzer_peak_pressure() {
        let mut analyzer = LiveRangeAnalyzer::new(20);
        analyzer.add("a", 0, 10);
        analyzer.add("b", 5, 15);
        analyzer.add("c", 8, 12);
        let report = analyzer.analyze();
        assert_eq!(report.peak_pressure, 3);
        assert!(report.peak_point >= 8 && report.peak_point <= 10);
    }

    #[test]
    fn live_range_analyzer_spill_candidates() {
        let mut analyzer = LiveRangeAnalyzer::new(20);
        // Long range covering high-pressure region.
        analyzer.add("long_var", 0, 19);
        analyzer.add("a", 5, 15);
        analyzer.add("b", 8, 12);
        analyzer.add("c", 9, 11);
        let report = analyzer.analyze();
        assert!(report.spill_candidates.contains(&"long_var".to_string()));
    }

    #[test]
    fn live_range_analyzer_remat_candidates() {
        let mut analyzer = LiveRangeAnalyzer::new(20);
        analyzer.add("long_var", 0, 19);
        analyzer.add("a", 5, 15);
        // Short-lived in high-pressure region.
        analyzer.add("remat_me", 10, 11);
        let report = analyzer.analyze();
        assert!(report.remat_candidates.contains(&"remat_me".to_string()));
    }

    #[test]
    fn live_range_analyzer_no_false_spill_candidates() {
        let mut analyzer = LiveRangeAnalyzer::new(100);
        analyzer.add("x", 0, 5);
        analyzer.add("y", 50, 55);
        let report = analyzer.analyze();
        assert!(report.spill_candidates.is_empty());
    }

    // -- CompilerHintGenerator --

    #[test]
    fn hint_generator_produces_launch_bounds() {
        let config = HintConfig::new(GpuArch::Ampere, KernelPattern::Gemm, 256);
        let hints = CompilerHintGenerator::generate(&config);
        assert!(hints.iter().any(|h| matches!(h, CompilerHint::LaunchBounds { .. })));
    }

    #[test]
    fn hint_generator_maxreg_for_high_pressure() {
        let config = HintConfig::new(GpuArch::Ampere, KernelPattern::FusedAttention, 256);
        let hints = CompilerHintGenerator::generate(&config);
        assert!(hints.iter().any(|h| matches!(h, CompilerHint::MaxRegCount(_))));
    }

    #[test]
    fn hint_generator_no_maxreg_for_low_pressure() {
        let config = HintConfig::new(GpuArch::Ampere, KernelPattern::Elementwise, 256);
        let hints = CompilerHintGenerator::generate(&config);
        assert!(!hints.iter().any(|h| matches!(h, CompilerHint::MaxRegCount(_))));
    }

    #[test]
    fn hint_generator_restrict_pointers() {
        let mut config = HintConfig::new(GpuArch::Ampere, KernelPattern::Gemm, 256);
        config.pointer_params = vec!["input".to_string(), "output".to_string()];
        let hints = CompilerHintGenerator::generate(&config);
        assert!(hints.iter().any(|h| matches!(h, CompilerHint::RestrictPointers { .. })));
    }

    #[test]
    fn hint_generator_unroll_loops() {
        let mut config = HintConfig::new(GpuArch::Ampere, KernelPattern::Gemm, 256);
        config.loops = vec![("inner_loop".to_string(), 4)];
        let hints = CompilerHintGenerator::generate(&config);
        assert!(hints.iter().any(|h| matches!(h, CompilerHint::PragmaUnroll { .. })));
    }

    #[test]
    fn hint_generator_constant_memory() {
        let mut config = HintConfig::new(GpuArch::Ampere, KernelPattern::Gemm, 256);
        config.constant_candidates = vec!["lookup_table".to_string()];
        let hints = CompilerHintGenerator::generate(&config);
        assert!(hints.iter().any(|h| matches!(h, CompilerHint::ConstantMemory { .. })));
    }

    #[test]
    fn hint_generator_force_inline() {
        let mut config = HintConfig::new(GpuArch::Ampere, KernelPattern::Gemm, 256);
        config.inline_candidates = vec!["helper_fn".to_string()];
        let hints = CompilerHintGenerator::generate(&config);
        assert!(hints.iter().any(|h| matches!(h, CompilerHint::ForceInline { .. })));
    }

    #[test]
    fn hint_generator_known_registers_override() {
        let mut config = HintConfig::new(GpuArch::Ampere, KernelPattern::Elementwise, 256);
        config.known_registers = Some(100);
        let hints = CompilerHintGenerator::generate(&config);
        // With 100 regs, should get maxrregcount hint.
        assert!(hints.iter().any(|h| matches!(h, CompilerHint::MaxRegCount(_))));
    }

    #[test]
    fn hint_display_launch_bounds() {
        let h = CompilerHint::LaunchBounds { max_threads: 256, min_blocks: 4 };
        assert_eq!(format!("{h}"), "__launch_bounds__(256, 4)");
    }

    #[test]
    fn hint_display_maxreg() {
        let h = CompilerHint::MaxRegCount(64);
        assert_eq!(format!("{h}"), "-maxrregcount=64");
    }

    #[test]
    fn hint_display_restrict() {
        let h = CompilerHint::RestrictPointers {
            pointer_names: vec!["a".to_string(), "b".to_string()],
        };
        let s = format!("{h}");
        assert!(s.contains("__restrict__"));
        assert!(s.contains("a, b"));
    }

    #[test]
    fn emit_cuda_annotations_contains_hints() {
        let hints = vec![
            CompilerHint::LaunchBounds { max_threads: 256, min_blocks: 4 },
            CompilerHint::MaxRegCount(128),
        ];
        let output = CompilerHintGenerator::emit_cuda_annotations(&hints);
        assert!(output.contains("__launch_bounds__(256, 4)"));
        assert!(output.contains("-maxrregcount=128"));
        assert!(output.contains("Register optimization hints"));
    }

    #[test]
    fn emit_cuda_annotations_restrict_pointers() {
        let hints =
            vec![CompilerHint::RestrictPointers { pointer_names: vec!["input".to_string()] }];
        let output = CompilerHintGenerator::emit_cuda_annotations(&hints);
        assert!(output.contains("__restrict__"));
        assert!(output.contains("input"));
    }

    #[test]
    fn emit_cuda_annotations_pragma_unroll() {
        let hints =
            vec![CompilerHint::PragmaUnroll { loop_label: "main_loop".to_string(), factor: 8 }];
        let output = CompilerHintGenerator::emit_cuda_annotations(&hints);
        assert!(output.contains("#pragma unroll 8"));
        assert!(output.contains("main_loop"));
    }

    // -- Full analysis pipeline --

    #[test]
    fn analyze_kernel_elementwise() {
        let result =
            analyze_kernel("eltwise_add", KernelPattern::Elementwise, GpuArch::Ampere, 256, 0)
                .unwrap();
        assert!(!result.spill.spills_detected);
        assert!(result.occupancy.occupancy > 0.5);
        assert!(!result.hints.is_empty());
    }

    #[test]
    fn analyze_kernel_gemm() {
        let result =
            analyze_kernel("gemm_f32", KernelPattern::Gemm, GpuArch::Ampere, 256, 0).unwrap();
        assert!(result.estimate.registers_per_thread >= 48);
        assert!(result.launch_bounds.min_blocks_per_sm >= 1);
    }

    #[test]
    fn analyze_kernel_fused_attention() {
        let result =
            analyze_kernel("fused_mha", KernelPattern::FusedAttention, GpuArch::Ampere, 256, 0)
                .unwrap();
        assert!(result.estimate.registers_per_thread >= 80);
    }

    #[test]
    fn analyze_kernel_invalid_threads_zero() {
        let result = analyze_kernel("bad", KernelPattern::Elementwise, GpuArch::Ampere, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn analyze_kernel_invalid_threads_too_many() {
        let result = analyze_kernel("bad", KernelPattern::Elementwise, GpuArch::Ampere, 2048, 0);
        assert!(result.is_err());
    }

    #[test]
    fn analyze_kernel_with_shared_mem() {
        let result = analyze_kernel(
            "reduce_smem",
            KernelPattern::Reduction,
            GpuArch::Ampere,
            256,
            16 * 1024,
        )
        .unwrap();
        assert!(result.occupancy.active_blocks >= 1);
    }

    #[test]
    fn analyze_kernel_different_archs() {
        let r_ampere = analyze_kernel("k", KernelPattern::Gemm, GpuArch::Ampere, 256, 0).unwrap();
        let r_turing = analyze_kernel("k", KernelPattern::Gemm, GpuArch::Turing, 256, 0).unwrap();
        // Both should produce valid results.
        assert!(r_ampere.occupancy.occupancy > 0.0);
        assert!(r_turing.occupancy.occupancy > 0.0);
    }

    #[test]
    fn analyze_kernel_quantized_gemv() {
        let result =
            analyze_kernel("qk256_gemv", KernelPattern::QuantizedGemv, GpuArch::Ampere, 256, 0)
                .unwrap();
        assert!(result.partition.regs_per_warp > 0);
    }

    #[test]
    fn analyze_kernel_custom_pattern() {
        let result =
            analyze_kernel("custom_k", KernelPattern::Custom(40), GpuArch::Hopper, 128, 0).unwrap();
        assert_eq!(result.estimate.registers_per_thread, 40);
    }

    // -- Cross-component integration --

    #[test]
    fn launch_bounds_consistent_with_occupancy() {
        let arch = GpuArch::Ampere;
        let calc = LaunchBoundsCalculator::new(arch);
        let tradeoff = OccupancyTradeoff::new(arch);

        let bounds = calc.compute(32, 256, 0);
        let point = tradeoff.evaluate(256, 0, 32);

        assert_eq!(bounds.min_blocks_per_sm, point.active_blocks.max(1));
    }

    #[test]
    fn partition_consistent_with_occupancy() {
        let arch = GpuArch::Ampere;
        let part = RegisterPartitioner::new(arch);
        let tradeoff = OccupancyTradeoff::new(arch);

        let partition = part.partition(32, 256);
        let point = tradeoff.evaluate(256, 0, 32);

        assert_eq!(partition.resident_warps, point.active_warps);
    }

    #[test]
    fn all_archs_produce_valid_analysis() {
        for arch in [
            GpuArch::Volta,
            GpuArch::Turing,
            GpuArch::Ampere,
            GpuArch::AmpereConsumer,
            GpuArch::Ada,
            GpuArch::Hopper,
        ] {
            let result = analyze_kernel("k", KernelPattern::Gemm, arch, 256, 0).unwrap();
            assert!(result.occupancy.occupancy > 0.0, "{arch} produced zero occupancy");
        }
    }

    #[test]
    fn all_patterns_produce_valid_analysis() {
        for pattern in [
            KernelPattern::Elementwise,
            KernelPattern::Reduction,
            KernelPattern::Gemm,
            KernelPattern::FusedAttention,
            KernelPattern::QuantizedGemv,
            KernelPattern::LayerNorm,
            KernelPattern::Embedding,
            KernelPattern::Conv1d,
            KernelPattern::MemoryOp,
            KernelPattern::Custom(50),
        ] {
            let result = analyze_kernel("test", pattern, GpuArch::Ampere, 256, 0).unwrap();
            assert!(result.estimate.registers_per_thread > 0, "{pattern} produced zero registers");
        }
    }
}
