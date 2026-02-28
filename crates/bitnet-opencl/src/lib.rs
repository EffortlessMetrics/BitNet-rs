//! OpenCL backend for BitNet GPU inference.
//!
//! Provides optimized OpenCL primitives for Intel Arc GPU inference,
//! including context pooling, local memory optimizations, prefetch
//! pipelines, KV cache, paged attention, and multi-backend GPU dispatch
//! with automatic selection.
//! pipelines, KV cache, paged attention, and CPU reference implementations
//! with OpenCL kernel sources for I2_S dequantization, QK256 block
//! dequantization, and ternary matrix multiply.
//! pipelines, KV cache, paged attention, SPIR-V compilation, and kernel registry.

pub mod backend_dispatcher;
pub mod backend_registry;
pub mod context_pool;
pub mod kv_cache;
pub mod paged_attention;

pub use backend_dispatcher::{
    BackendCapabilityMatrix, BackendDispatcher, BackendStatus, DispatchDecision, DispatchError,
    DispatchLog, DispatchStrategy, Operation,
};
pub use backend_registry::{BackendInfo, BackendProvider, BackendRegistry};
pub mod quantized_kernels;
pub mod quantized_ops;
pub mod spirv;
pub mod spirv_kernels;

// Re-exports for convenience.
pub use spirv::{
    CompileOptions, CompilerBackend, OptimizationLevel, SPIRV_MAGIC, SpirVCache, SpirVCompiler,
    SpirVError, SpirVModule, SpirVValidator,
};
pub use spirv_kernels::{KernelSource, SpirvKernelRegistry};
//! `OpenCL` backend and GPU stress testing utilities for `BitNet` inference.
//!
//! This crate provides:
//! - Mock `OpenCL` backend for testing without real GPU hardware
//! - Stress testing utilities (`StressTestRunner`, `LoadGenerator`, `ResultCollector`)
//! - GPU memory budget simulation

pub mod stress_utils;

use bitnet_common::{KernelError, QuantizationType, Result};
use bitnet_kernels::KernelProvider;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Simulated memory budget for GPU stress testing.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    total_bytes: u64,
    allocated: Arc<AtomicU64>,
}

impl MemoryBudget {
    #[must_use]
    pub fn new(total_bytes: u64) -> Self {
        Self { total_bytes, allocated: Arc::new(AtomicU64::new(0)) }
    }

    /// Try to allocate `bytes`. Returns `true` on success.
    pub fn try_alloc(&self, bytes: u64) -> bool {
        let mut current = self.allocated.load(Ordering::Relaxed);
        loop {
            if current + bytes > self.total_bytes {
                return false;
            }
            match self.allocated.compare_exchange_weak(
                current,
                current + bytes,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    pub fn free(&self, bytes: u64) {
        self.allocated.fetch_sub(bytes, Ordering::Release);
    }

    #[must_use]
    pub fn used(&self) -> u64 {
        self.allocated.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn available(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used())
    }
}

/// Mock `OpenCL` kernel that delegates to CPU fallback for testing.
pub struct MockOpenClKernel {
    fallback: bitnet_kernels::FallbackKernel,
    available: AtomicBool,
    dispatch_count: AtomicUsize,
    error_injection: AtomicBool,
    memory_budget: Option<MemoryBudget>,
}

impl MockOpenClKernel {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            fallback: bitnet_kernels::FallbackKernel,
            available: AtomicBool::new(true),
            dispatch_count: AtomicUsize::new(0),
            error_injection: AtomicBool::new(false),
            memory_budget: None,
        }
    }

    #[must_use]
    pub fn with_memory_budget(mut self, budget: MemoryBudget) -> Self {
        self.memory_budget = Some(budget);
        self
    }

    pub fn set_available(&self, available: bool) {
        self.available.store(available, Ordering::Release);
    }

    pub fn set_error_injection(&self, inject: bool) {
        self.error_injection.store(inject, Ordering::Release);
    }

    #[must_use]
    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count.load(Ordering::Acquire)
    }

    #[must_use]
    pub const fn memory_budget(&self) -> Option<&MemoryBudget> {
        self.memory_budget.as_ref()
    }
}

impl Default for MockOpenClKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelProvider for MockOpenClKernel {
    fn name(&self) -> &'static str {
        "mock-opencl"
    }

    fn is_available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }

    #[allow(clippy::many_single_char_names)]
    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);

        if self.error_injection.load(Ordering::Acquire) {
            return Err(bitnet_common::BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: "injected error during matmul".into(),
            }));
        }

        if let Some(budget) = &self.memory_budget {
            let needed = (m * n * 4) as u64;
            if !budget.try_alloc(needed) {
                return Err(bitnet_common::BitNetError::Kernel(KernelError::GpuError {
                    reason: format!(
                        "memory budget exceeded: need {needed} bytes, \
                             available {}",
                        budget.available()
                    ),
                }));
            }
            let result = self.fallback.matmul_i2s(a, b, c, m, n, k);
            budget.free(needed);
            return result;
        }

        self.fallback.matmul_i2s(a, b, c, m, n, k)
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        self.dispatch_count.fetch_add(1, Ordering::Relaxed);

        if self.error_injection.load(Ordering::Acquire) {
            return Err(bitnet_common::BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: "injected error during quantize".into(),
            }));
        }

        self.fallback.quantize(input, output, scales, qtype)
    }
}

/// Configuration for the mock `OpenCL` backend, adjustable at runtime.
#[derive(Debug, Clone)]
pub struct OpenClConfig {
    pub max_queue_depth: usize,
    pub memory_budget_bytes: u64,
    pub use_gpu: bool,
    pub batch_size: usize,
}

impl Default for OpenClConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: 64,
            memory_budget_bytes: 1024 * 1024 * 512, // 512 MiB
            use_gpu: false,
            batch_size: 1,
        }
    }
}
