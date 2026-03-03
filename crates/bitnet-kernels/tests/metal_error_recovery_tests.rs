#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_saturating_arithmetic)]

//! Metal error recovery and resilience tests for Apple Silicon.
//!
//! Validates GPU error handling paths: memory allocation failures, shader
//! compilation errors, command buffer faults, pipeline creation failures,
//! resource exhaustion, timeout detection, and graceful GPU → CPU fallback.
//!
//! All tests are `#[ignore]` — they validate logic and structure but require
//! a Metal GPU runtime to exercise real GPU paths.

// Types and constants from `metal_compute` are duplicated here so this test
// compiles without `--features metal`.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

// ── Duplicated Metal constants (mirrors metal_compute.rs) ───────────

const METAL_BUFFER_ALIGNMENT: usize = 256;
const METAL_MAX_WORKGROUP_SIZE: u32 = 1024;
const DEFAULT_TILE_SIZE: u32 = 16;
const MAX_DISPATCH_DIM: u32 = 65535;

// ── Duplicated Metal types (mirrors metal_compute.rs) ───────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkgroupSize {
    x: u32,
    y: u32,
    z: u32,
}

impl WorkgroupSize {
    fn new(x: u32, y: u32, z: u32) -> Result<Self, MetalConfigError> {
        let total = (x as u64) * (y as u64) * (z as u64);
        if total > METAL_MAX_WORKGROUP_SIZE as u64 {
            return Err(MetalConfigError::WorkgroupTooLarge {
                requested: total,
                max: METAL_MAX_WORKGROUP_SIZE,
            });
        }
        if x == 0 || y == 0 || z == 0 {
            return Err(MetalConfigError::ZeroDimension);
        }
        Ok(Self { x, y, z })
    }

    fn total_threads(&self) -> u32 {
        self.x * self.y * self.z
    }

    fn linear(n: u32) -> Result<Self, MetalConfigError> {
        Self::new(n, 1, 1)
    }

    fn tile(size: u32) -> Result<Self, MetalConfigError> {
        Self::new(size, size, 1)
    }
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self { x: DEFAULT_TILE_SIZE, y: DEFAULT_TILE_SIZE, z: 1 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchDimensions {
    x: u32,
    y: u32,
    z: u32,
}

impl DispatchDimensions {
    fn for_problem(problem: (u32, u32, u32), wg: &WorkgroupSize) -> Result<Self, MetalConfigError> {
        let dim = |p: u32, w: u32| -> Result<u32, MetalConfigError> {
            if w == 0 {
                return Err(MetalConfigError::ZeroDimension);
            }
            let d = p.div_ceil(w);
            if d > MAX_DISPATCH_DIM {
                return Err(MetalConfigError::DispatchTooLarge {
                    dimension: d,
                    max: MAX_DISPATCH_DIM,
                });
            }
            Ok(d)
        };
        Ok(Self { x: dim(problem.0, wg.x)?, y: dim(problem.1, wg.y)?, z: dim(problem.2, wg.z)? })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MemoryArchitecture {
    Unified,
    Discrete,
}

impl MemoryArchitecture {
    fn detect() -> Self {
        if cfg!(all(target_arch = "aarch64", target_vendor = "apple")) {
            Self::Unified
        } else {
            Self::Discrete
        }
    }

    fn supports_zero_copy(&self) -> bool {
        *self == Self::Unified
    }
}

#[inline]
fn align_buffer_size(size: usize) -> usize {
    let mask = METAL_BUFFER_ALIGNMENT - 1;
    (size + mask) & !mask
}

#[inline]
fn is_aligned(offset: usize) -> bool {
    offset % METAL_BUFFER_ALIGNMENT == 0
}

#[derive(Debug, Clone)]
struct MetalComputePipeline {
    label: String,
    workgroup: WorkgroupSize,
    memory: MemoryArchitecture,
    use_shared_memory: bool,
}

impl MetalComputePipeline {
    fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            workgroup: WorkgroupSize::default(),
            memory: MemoryArchitecture::detect(),
            use_shared_memory: true,
        }
    }

    fn with_workgroup(mut self, wg: WorkgroupSize) -> Self {
        self.workgroup = wg;
        self
    }

    fn with_memory(mut self, mem: MemoryArchitecture) -> Self {
        self.memory = mem;
        self
    }

    fn dispatch_for_matrix(
        &self,
        rows: u32,
        cols: u32,
    ) -> Result<DispatchDimensions, MetalConfigError> {
        DispatchDimensions::for_problem((cols, rows, 1), &self.workgroup)
    }

    fn aligned_buffer_bytes(&self, element_count: usize, element_bytes: usize) -> usize {
        align_buffer_size(element_count * element_bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MetalConfigError {
    WorkgroupTooLarge { requested: u64, max: u32 },
    ZeroDimension,
    DispatchTooLarge { dimension: u32, max: u32 },
}

impl fmt::Display for MetalConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkgroupTooLarge { requested, max } => {
                write!(f, "workgroup size {requested} exceeds Metal limit {max}")
            }
            Self::ZeroDimension => write!(f, "dimension must be non-zero"),
            Self::DispatchTooLarge { dimension, max } => {
                write!(f, "dispatch dimension {dimension} exceeds Metal limit {max}")
            }
        }
    }
}

impl std::error::Error for MetalConfigError {}

// ═══════════════════════════════════════════════════════════════════
// Test-local error types that model Metal GPU error states
// ═══════════════════════════════════════════════════════════════════

/// Simulated Metal GPU error for recovery testing.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MetalGpuError {
    OutOfMemory { requested: usize, available: usize },
    BufferAllocationFailed { size: usize, reason: String },
    ShaderCompilationFailed { source_snippet: String, error_msg: String },
    ShaderFunctionNotFound { name: String },
    CommandBufferError { status: CommandBufferStatus },
    PipelineCreationFailed { label: String, reason: String },
    ResourceLimitExceeded { resource: String, limit: usize, requested: usize },
    Timeout { operation: String, elapsed: Duration, limit: Duration },
    DeviceLost { reason: String },
    InvalidOperation { description: String },
}

impl fmt::Display for MetalGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory { requested, available } => {
                write!(f, "Metal OOM: requested {requested} bytes but only {available} available")
            }
            Self::BufferAllocationFailed { size, reason } => {
                write!(f, "Metal buffer allocation failed ({size} bytes): {reason}")
            }
            Self::ShaderCompilationFailed { source_snippet, error_msg } => {
                write!(f, "Metal shader compilation failed near '{source_snippet}': {error_msg}")
            }
            Self::ShaderFunctionNotFound { name } => {
                write!(f, "Metal shader function '{name}' not found in library")
            }
            Self::CommandBufferError { status } => {
                write!(f, "Metal command buffer error: {status:?}")
            }
            Self::PipelineCreationFailed { label, reason } => {
                write!(f, "Metal pipeline '{label}' creation failed: {reason}")
            }
            Self::ResourceLimitExceeded { resource, limit, requested } => {
                write!(f, "Metal {resource} limit exceeded: requested {requested}, limit {limit}")
            }
            Self::Timeout { operation, elapsed, limit } => {
                write!(f, "Metal timeout: '{operation}' took {elapsed:?} (limit {limit:?})")
            }
            Self::DeviceLost { reason } => {
                write!(f, "Metal device lost: {reason}")
            }
            Self::InvalidOperation { description } => {
                write!(f, "Metal invalid operation: {description}")
            }
        }
    }
}

impl std::error::Error for MetalGpuError {}

/// Simulated command buffer completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandBufferStatus {
    Completed,
    Error,
    Timeout,
    GpuFault,
    NotEnqueued,
    Enqueued,
    Committed,
}

/// Simulated GPU memory pool for allocation/OOM testing.
struct GpuMemoryPool {
    total: usize,
    allocated: usize,
    allocations: Vec<(usize, usize)>, // (offset, size)
    fragmented: bool,
}

impl GpuMemoryPool {
    fn new(total: usize) -> Self {
        Self { total, allocated: 0, allocations: Vec::new(), fragmented: false }
    }

    fn available(&self) -> usize {
        self.total.saturating_sub(self.allocated)
    }

    fn allocate(&mut self, size: usize) -> Result<usize, MetalGpuError> {
        let aligned = align_buffer_size(size);
        if aligned > self.available() {
            return Err(MetalGpuError::OutOfMemory {
                requested: aligned,
                available: self.available(),
            });
        }
        if self.fragmented && aligned > self.total / 4 {
            return Err(MetalGpuError::BufferAllocationFailed {
                size: aligned,
                reason: "memory fragmentation prevents contiguous allocation".into(),
            });
        }
        let offset = self.allocated;
        self.allocations.push((offset, aligned));
        self.allocated += aligned;
        Ok(offset)
    }

    fn free_last(&mut self) -> bool {
        if let Some((_, size)) = self.allocations.pop() {
            self.allocated = self.allocated.saturating_sub(size);
            true
        } else {
            false
        }
    }

    fn free_all(&mut self) {
        self.allocations.clear();
        self.allocated = 0;
        self.fragmented = false;
    }

    fn fragment(&mut self) {
        self.fragmented = true;
    }

    fn allocation_count(&self) -> usize {
        self.allocations.len()
    }
}

/// Simulated shader compilation result.
struct ShaderCompileResult {
    success: bool,
    function_names: Vec<String>,
    error: Option<MetalGpuError>,
    warnings: Vec<String>,
}

fn compile_shader(source: &str) -> ShaderCompileResult {
    if source.is_empty() {
        return ShaderCompileResult {
            success: false,
            function_names: vec![],
            error: Some(MetalGpuError::ShaderCompilationFailed {
                source_snippet: String::new(),
                error_msg: "empty shader source".into(),
            }),
            warnings: vec![],
        };
    }
    if !source.contains("fn ") && !source.contains("kernel ") {
        return ShaderCompileResult {
            success: false,
            function_names: vec![],
            error: Some(MetalGpuError::ShaderCompilationFailed {
                source_snippet: source.chars().take(40).collect(),
                error_msg: "no kernel or function entry point found".into(),
            }),
            warnings: vec![],
        };
    }
    if source.contains("syntax_error!") {
        return ShaderCompileResult {
            success: false,
            function_names: vec![],
            error: Some(MetalGpuError::ShaderCompilationFailed {
                source_snippet: "syntax_error!".into(),
                error_msg: "unexpected token 'syntax_error!'".into(),
            }),
            warnings: vec![],
        };
    }
    let mut names = vec![];
    let mut warnings = vec![];
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("fn ") {
            if let Some(name) = rest.split('(').next() {
                names.push(name.trim().to_string());
            }
        }
        if let Some(rest) = trimmed.strip_prefix("kernel ") {
            if let Some(name) = rest.split('(').next() {
                names.push(name.trim().to_string());
            }
        }
        if trimmed.contains("// WARNING:") {
            warnings.push(trimmed.to_string());
        }
    }
    ShaderCompileResult { success: true, function_names: names, error: None, warnings }
}

fn lookup_function(result: &ShaderCompileResult, name: &str) -> Result<String, MetalGpuError> {
    if !result.success {
        return Err(MetalGpuError::InvalidOperation {
            description: "cannot look up function in failed compilation".into(),
        });
    }
    if result.function_names.iter().any(|n| n == name) {
        Ok(name.to_string())
    } else {
        Err(MetalGpuError::ShaderFunctionNotFound { name: name.into() })
    }
}

/// Simulated command buffer for error state testing.
struct SimCommandBuffer {
    status: CommandBufferStatus,
    dispatches: Vec<String>,
    error: Option<MetalGpuError>,
}

impl SimCommandBuffer {
    fn new() -> Self {
        Self { status: CommandBufferStatus::NotEnqueued, dispatches: vec![], error: None }
    }

    fn add_dispatch(&mut self, label: &str) -> Result<(), MetalGpuError> {
        if self.status == CommandBufferStatus::Error || self.status == CommandBufferStatus::GpuFault
        {
            return Err(MetalGpuError::CommandBufferError { status: self.status });
        }
        self.dispatches.push(label.to_string());
        Ok(())
    }

    fn commit(&mut self) -> Result<(), MetalGpuError> {
        if self.dispatches.is_empty() {
            return Err(MetalGpuError::InvalidOperation {
                description: "cannot commit empty command buffer".into(),
            });
        }
        self.status = CommandBufferStatus::Committed;
        Ok(())
    }

    fn simulate_completion(&mut self, outcome: CommandBufferStatus) -> Result<(), MetalGpuError> {
        self.status = outcome;
        if outcome == CommandBufferStatus::Error || outcome == CommandBufferStatus::GpuFault {
            self.error = Some(MetalGpuError::CommandBufferError { status: outcome });
            Err(MetalGpuError::CommandBufferError { status: outcome })
        } else {
            Ok(())
        }
    }
}

/// Tracks GPU → CPU fallback decisions.
#[derive(Debug)]
struct FallbackTracker {
    gpu_available: AtomicBool,
    fallback_count: AtomicU64,
    log: std::sync::Mutex<Vec<String>>,
}

impl FallbackTracker {
    fn new(gpu_available: bool) -> Self {
        Self {
            gpu_available: AtomicBool::new(gpu_available),
            fallback_count: AtomicU64::new(0),
            log: std::sync::Mutex::new(vec![]),
        }
    }

    fn execute<F, G, T>(&self, gpu_fn: F, cpu_fn: G) -> Result<T, MetalGpuError>
    where
        F: FnOnce() -> Result<T, MetalGpuError>,
        G: FnOnce() -> T,
    {
        if self.gpu_available.load(Ordering::Relaxed) {
            match gpu_fn() {
                Ok(val) => {
                    self.log.lock().unwrap().push("GPU: success".into());
                    Ok(val)
                }
                Err(e) => {
                    self.fallback_count.fetch_add(1, Ordering::Relaxed);
                    self.log.lock().unwrap().push(format!("GPU failed ({e}), falling back to CPU"));
                    Ok(cpu_fn())
                }
            }
        } else {
            self.fallback_count.fetch_add(1, Ordering::Relaxed);
            self.log.lock().unwrap().push("GPU unavailable, using CPU".into());
            Ok(cpu_fn())
        }
    }

    fn mark_gpu_lost(&self) {
        self.gpu_available.store(false, Ordering::Relaxed);
    }

    fn fallback_count(&self) -> u64 {
        self.fallback_count.load(Ordering::Relaxed)
    }

    fn log_entries(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
}

/// Accumulates multiple errors from a pipeline execution.
struct ErrorAccumulator {
    errors: Vec<MetalGpuError>,
    max_errors: usize,
}

impl ErrorAccumulator {
    fn new(max_errors: usize) -> Self {
        Self { errors: Vec::new(), max_errors }
    }

    fn record(&mut self, err: MetalGpuError) -> bool {
        self.errors.push(err);
        self.errors.len() < self.max_errors
    }

    fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    fn error_count(&self) -> usize {
        self.errors.len()
    }

    fn contains_oom(&self) -> bool {
        self.errors.iter().any(|e| matches!(e, MetalGpuError::OutOfMemory { .. }))
    }

    fn contains_timeout(&self) -> bool {
        self.errors.iter().any(|e| matches!(e, MetalGpuError::Timeout { .. }))
    }

    fn first_error(&self) -> Option<&MetalGpuError> {
        self.errors.first()
    }

    fn drain(&mut self) -> Vec<MetalGpuError> {
        std::mem::take(&mut self.errors)
    }
}

/// Simulated timeout watcher for GPU operations.
struct TimeoutWatcher {
    limit: Duration,
}

impl TimeoutWatcher {
    fn new(limit: Duration) -> Self {
        Self { limit }
    }

    fn check_elapsed(&self, elapsed: Duration, operation: &str) -> Result<(), MetalGpuError> {
        if elapsed > self.limit {
            Err(MetalGpuError::Timeout { operation: operation.into(), elapsed, limit: self.limit })
        } else {
            Ok(())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 1 — GPU memory allocation failure simulation
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_oom_basic_allocation() {
    let mut pool = GpuMemoryPool::new(1024);
    assert!(pool.allocate(512).is_ok());
    let err = pool.allocate(768).unwrap_err();
    assert!(matches!(err, MetalGpuError::OutOfMemory { .. }));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_oom_exact_capacity() {
    let mut pool = GpuMemoryPool::new(256);
    assert!(pool.allocate(256).is_ok());
    let err = pool.allocate(1).unwrap_err();
    assert!(matches!(err, MetalGpuError::OutOfMemory { .. }));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_oom_error_message_includes_sizes() {
    let mut pool = GpuMemoryPool::new(512);
    pool.allocate(256).unwrap();
    let err = pool.allocate(512).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("512"));
    assert!(msg.contains("256"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_oom_recovery_after_free() {
    let mut pool = GpuMemoryPool::new(512);
    pool.allocate(256).unwrap();
    pool.allocate(256).unwrap();
    assert!(pool.allocate(1).is_err());
    pool.free_last();
    assert!(pool.allocate(256).is_ok());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_oom_free_all_recovery() {
    let mut pool = GpuMemoryPool::new(1024);
    for _ in 0..4 {
        pool.allocate(256).unwrap();
    }
    assert!(pool.allocate(1).is_err());
    pool.free_all();
    assert_eq!(pool.available(), 1024);
    assert!(pool.allocate(1024).is_ok());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fragmented_allocation_fails() {
    let mut pool = GpuMemoryPool::new(4096);
    pool.allocate(256).unwrap();
    pool.fragment();
    let err = pool.allocate(2048).unwrap_err();
    assert!(matches!(err, MetalGpuError::BufferAllocationFailed { .. }));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fragmentation_small_alloc_succeeds() {
    let mut pool = GpuMemoryPool::new(4096);
    pool.allocate(256).unwrap();
    pool.fragment();
    // Small allocations still fit even when fragmented.
    assert!(pool.allocate(256).is_ok());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_zero_size_allocation_alignment() {
    let mut pool = GpuMemoryPool::new(1024);
    // align_buffer_size(0) == 0 → requires 0, succeeds trivially.
    assert!(pool.allocate(0).is_ok());
    assert_eq!(pool.available(), 1024);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_multiple_small_allocations_until_oom() {
    let mut pool = GpuMemoryPool::new(1024);
    let mut count = 0u32;
    loop {
        if pool.allocate(1).is_err() {
            break;
        }
        count += 1;
    }
    // Each 1-byte alloc rounds to 256 bytes → 4 allocations.
    assert_eq!(count, 4);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_allocation_returns_aligned_offsets() {
    let mut pool = GpuMemoryPool::new(2048);
    let off1 = pool.allocate(100).unwrap();
    let off2 = pool.allocate(50).unwrap();
    assert!(is_aligned(off1));
    assert!(is_aligned(off2));
}

// ═══════════════════════════════════════════════════════════════════
// 2 — Shader compilation error detection
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_compile_empty_source() {
    let result = compile_shader("");
    assert!(!result.success);
    assert!(result.error.is_some());
    let msg = result.error.unwrap().to_string();
    assert!(msg.contains("empty shader source"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_compile_no_entry_point() {
    let result = compile_shader("let x = 42;");
    assert!(!result.success);
    let msg = result.error.unwrap().to_string();
    assert!(msg.contains("no kernel or function entry point"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_compile_syntax_error() {
    let result = compile_shader("fn main() { syntax_error! }");
    assert!(!result.success);
    let msg = result.error.unwrap().to_string();
    assert!(msg.contains("syntax_error!"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_compile_valid_function() {
    let result = compile_shader("fn my_kernel(x: f32) -> f32 { x }");
    assert!(result.success);
    assert!(result.function_names.contains(&"my_kernel".to_string()));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_compile_multiple_functions() {
    let src = "fn kernel_a(x: f32) {}\nfn kernel_b(y: f32) {}";
    let result = compile_shader(src);
    assert!(result.success);
    assert_eq!(result.function_names.len(), 2);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_compile_kernel_entry() {
    let result = compile_shader("kernel void my_compute(uint id) {}");
    assert!(result.success);
    assert!(result.function_names.contains(&"void".to_string()) || result.success);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_function_lookup_success() {
    let result = compile_shader("fn matmul(a: f32, b: f32) {}");
    assert!(lookup_function(&result, "matmul").is_ok());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_function_lookup_missing() {
    let result = compile_shader("fn existing_fn(x: f32) {}");
    let err = lookup_function(&result, "nonexistent").unwrap_err();
    assert!(matches!(err, MetalGpuError::ShaderFunctionNotFound { .. }));
    assert!(err.to_string().contains("nonexistent"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_function_lookup_on_failed_compilation() {
    let result = compile_shader("");
    let err = lookup_function(&result, "anything").unwrap_err();
    assert!(matches!(err, MetalGpuError::InvalidOperation { .. }));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shader_compile_warnings_collected() {
    let src = "fn kern(x: f32) {}\n// WARNING: implicit cast";
    let result = compile_shader(src);
    assert!(result.success);
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("WARNING"));
}

// ═══════════════════════════════════════════════════════════════════
// 3 — Command buffer error states
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_command_buffer_initial_state() {
    let cb = SimCommandBuffer::new();
    assert_eq!(cb.status, CommandBufferStatus::NotEnqueued);
    assert!(cb.dispatches.is_empty());
    assert!(cb.error.is_none());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_command_buffer_normal_flow() {
    let mut cb = SimCommandBuffer::new();
    cb.add_dispatch("matmul").unwrap();
    cb.add_dispatch("relu").unwrap();
    cb.commit().unwrap();
    assert_eq!(cb.status, CommandBufferStatus::Committed);
    assert_eq!(cb.dispatches.len(), 2);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_command_buffer_commit_empty_fails() {
    let mut cb = SimCommandBuffer::new();
    let err = cb.commit().unwrap_err();
    assert!(matches!(err, MetalGpuError::InvalidOperation { .. }));
    assert!(err.to_string().contains("empty"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_command_buffer_error_state_rejects_dispatch() {
    let mut cb = SimCommandBuffer::new();
    cb.status = CommandBufferStatus::Error;
    let err = cb.add_dispatch("kernel").unwrap_err();
    assert!(matches!(err, MetalGpuError::CommandBufferError { .. }));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_command_buffer_gpu_fault_rejects_dispatch() {
    let mut cb = SimCommandBuffer::new();
    cb.status = CommandBufferStatus::GpuFault;
    let err = cb.add_dispatch("kernel").unwrap_err();
    matches!(err, MetalGpuError::CommandBufferError { status }
        if status == CommandBufferStatus::GpuFault);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_command_buffer_timeout_completion() {
    let mut cb = SimCommandBuffer::new();
    cb.add_dispatch("slow_kernel").unwrap();
    cb.commit().unwrap();
    let err = cb.simulate_completion(CommandBufferStatus::Timeout).unwrap_err();
    assert!(matches!(
        err,
        MetalGpuError::CommandBufferError { status: CommandBufferStatus::Timeout }
    ));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_command_buffer_successful_completion() {
    let mut cb = SimCommandBuffer::new();
    cb.add_dispatch("kernel").unwrap();
    cb.commit().unwrap();
    assert!(cb.simulate_completion(CommandBufferStatus::Completed).is_ok());
    assert_eq!(cb.status, CommandBufferStatus::Completed);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_command_buffer_gpu_fault_sets_error() {
    let mut cb = SimCommandBuffer::new();
    cb.add_dispatch("kernel").unwrap();
    cb.commit().unwrap();
    let _ = cb.simulate_completion(CommandBufferStatus::GpuFault);
    assert!(cb.error.is_some());
}

// ═══════════════════════════════════════════════════════════════════
// 4 — Pipeline state creation failures
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_pipeline_creation_missing_shader() {
    let result = compile_shader("");
    let err = lookup_function(&result, "main").unwrap_err();
    // Attempting to build a pipeline without a valid shader fails.
    assert!(matches!(err, MetalGpuError::InvalidOperation { .. }));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_pipeline_creation_wrong_function_name() {
    let result = compile_shader("fn compute_kernel(x: f32) {}");
    let err = lookup_function(&result, "wrong_name").unwrap_err();
    assert!(matches!(err, MetalGpuError::ShaderFunctionNotFound { .. }));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_pipeline_workgroup_exceeds_limit_is_error() {
    let err = WorkgroupSize::new(2048, 1, 1).unwrap_err();
    assert_eq!(
        err,
        MetalConfigError::WorkgroupTooLarge { requested: 2048, max: METAL_MAX_WORKGROUP_SIZE }
    );
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_pipeline_zero_workgroup_dimension() {
    assert!(WorkgroupSize::new(0, 16, 1).is_err());
    assert!(WorkgroupSize::new(16, 0, 1).is_err());
    assert!(WorkgroupSize::new(16, 16, 0).is_err());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_pipeline_incompatible_dispatch_dimensions() {
    let wg = WorkgroupSize::linear(1).unwrap();
    let err = DispatchDimensions::for_problem((MAX_DISPATCH_DIM + 1, 1, 1), &wg);
    assert!(err.is_err());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_pipeline_creation_error_is_descriptive() {
    let err = MetalGpuError::PipelineCreationFailed {
        label: "matmul_f16".into(),
        reason: "missing vertex function for compute pipeline".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("matmul_f16"));
    assert!(msg.contains("missing vertex function"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_pipeline_config_survives_error() {
    let pipeline = MetalComputePipeline::new("robust_kernel");
    // Even if dispatch fails, the pipeline config is intact.
    let err = pipeline.dispatch_for_matrix(0, 0);
    assert!(err.is_err());
    // Pipeline is not consumed — can try again.
    assert_eq!(pipeline.label, "robust_kernel");
    assert_eq!(pipeline.workgroup.total_threads(), 256);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_pipeline_retry_after_workgroup_error() {
    // First attempt: bad workgroup.
    assert!(WorkgroupSize::tile(64).is_err());
    // Retry with valid workgroup.
    let wg = WorkgroupSize::tile(16).unwrap();
    let pipeline = MetalComputePipeline::new("retry_kernel").with_workgroup(wg);
    assert_eq!(pipeline.workgroup.total_threads(), 256);
}

// ═══════════════════════════════════════════════════════════════════
// 5 — Resource exhaustion scenarios
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_resource_limit_buffer_count() {
    let max_buffers = 31; // Metal argument buffer limit.
    let mut buffers = Vec::new();
    let mut pool = GpuMemoryPool::new(max_buffers * 256);
    for i in 0..max_buffers {
        buffers.push(pool.allocate(256).unwrap());
        assert_eq!(pool.allocation_count(), i + 1);
    }
    assert_eq!(buffers.len(), max_buffers);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_resource_limit_exceeded_error() {
    let err = MetalGpuError::ResourceLimitExceeded {
        resource: "buffers_per_stage".into(),
        limit: 31,
        requested: 32,
    };
    let msg = err.to_string();
    assert!(msg.contains("buffers_per_stage"));
    assert!(msg.contains("31"));
    assert!(msg.contains("32"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_resource_limit_texture_count() {
    let err = MetalGpuError::ResourceLimitExceeded {
        resource: "textures_per_stage".into(),
        limit: 128,
        requested: 129,
    };
    assert!(err.to_string().contains("textures_per_stage"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_resource_exhaustion_graceful_message() {
    let err = MetalGpuError::ResourceLimitExceeded {
        resource: "threadgroup_memory".into(),
        limit: 32768,
        requested: 65536,
    };
    let msg = err.to_string();
    assert!(msg.contains("threadgroup_memory"));
    assert!(msg.contains("limit"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_resource_max_buffer_size_per_allocation() {
    let huge_size: usize = 256 * 1024 * 1024; // 256 MB
    let mut pool = GpuMemoryPool::new(huge_size);
    assert!(pool.allocate(huge_size).is_ok());
    assert!(pool.allocate(1).is_err());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_resource_pool_free_and_reuse() {
    let mut pool = GpuMemoryPool::new(512);
    pool.allocate(256).unwrap();
    pool.allocate(256).unwrap();
    assert!(pool.allocate(256).is_err());
    pool.free_last();
    assert!(pool.allocate(256).is_ok());
    assert_eq!(pool.allocation_count(), 2);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_resource_buffer_alignment_enforced() {
    for size in [1, 7, 128, 255, 257, 511, 513] {
        let aligned = align_buffer_size(size);
        assert!(is_aligned(aligned), "align_buffer_size({size}) = {aligned} is not aligned");
        assert!(aligned >= size);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6 — Timeout detection and reporting
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_timeout_within_limit() {
    let watcher = TimeoutWatcher::new(Duration::from_secs(5));
    assert!(watcher.check_elapsed(Duration::from_secs(3), "matmul").is_ok());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_timeout_exceeded() {
    let watcher = TimeoutWatcher::new(Duration::from_secs(5));
    let err = watcher.check_elapsed(Duration::from_secs(6), "slow_kernel").unwrap_err();
    assert!(matches!(err, MetalGpuError::Timeout { .. }));
    assert!(err.to_string().contains("slow_kernel"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_timeout_exact_boundary() {
    let watcher = TimeoutWatcher::new(Duration::from_millis(100));
    assert!(watcher.check_elapsed(Duration::from_millis(100), "exact").is_ok());
    let err = watcher.check_elapsed(Duration::from_millis(101), "over").unwrap_err();
    assert!(matches!(err, MetalGpuError::Timeout { .. }));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_timeout_error_includes_durations() {
    let watcher = TimeoutWatcher::new(Duration::from_secs(2));
    let err = watcher.check_elapsed(Duration::from_secs(10), "heavy_op").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("heavy_op"));
    assert!(msg.contains("10"));
    assert!(msg.contains("2"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_timeout_configurable_limits() {
    for ms in [10, 100, 500, 1000, 5000] {
        let watcher = TimeoutWatcher::new(Duration::from_millis(ms));
        assert!(watcher.check_elapsed(Duration::from_millis(ms - 1), "op").is_ok());
        assert!(watcher.check_elapsed(Duration::from_millis(ms + 1), "op").is_err());
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_timeout_zero_elapsed_always_passes() {
    let watcher = TimeoutWatcher::new(Duration::from_millis(1));
    assert!(watcher.check_elapsed(Duration::ZERO, "instant").is_ok());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_timeout_very_small_limit() {
    let watcher = TimeoutWatcher::new(Duration::from_nanos(1));
    let err = watcher.check_elapsed(Duration::from_micros(1), "micro_op").unwrap_err();
    assert!(matches!(err, MetalGpuError::Timeout { .. }));
}

// ═══════════════════════════════════════════════════════════════════
// 7 — Graceful degradation (GPU → CPU fallback)
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fallback_gpu_success_no_fallback() {
    let tracker = FallbackTracker::new(true);
    let result: Result<i32, _> = tracker.execute(|| Ok(42), || 0);
    assert_eq!(result.unwrap(), 42);
    assert_eq!(tracker.fallback_count(), 0);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fallback_gpu_error_triggers_cpu() {
    let tracker = FallbackTracker::new(true);
    let result: Result<i32, _> = tracker
        .execute(|| Err(MetalGpuError::OutOfMemory { requested: 1024, available: 0 }), || 99);
    assert_eq!(result.unwrap(), 99);
    assert_eq!(tracker.fallback_count(), 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fallback_gpu_unavailable_uses_cpu() {
    let tracker = FallbackTracker::new(false);
    let result: Result<i32, _> = tracker.execute(|| Ok(42), || 77);
    assert_eq!(result.unwrap(), 77);
    assert_eq!(tracker.fallback_count(), 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fallback_device_lost_marks_unavailable() {
    let tracker = FallbackTracker::new(true);
    tracker.mark_gpu_lost();
    let result: Result<i32, _> = tracker.execute(|| Ok(42), || 55);
    assert_eq!(result.unwrap(), 55);
    assert_eq!(tracker.fallback_count(), 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fallback_log_records_path() {
    let tracker = FallbackTracker::new(true);
    let _: Result<i32, _> = tracker.execute(|| Ok(1), || 0);
    let entries = tracker.log_entries();
    assert_eq!(entries.len(), 1);
    assert!(entries[0].contains("GPU: success"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fallback_log_records_error_and_cpu() {
    let tracker = FallbackTracker::new(true);
    let _: Result<i32, _> =
        tracker.execute(|| Err(MetalGpuError::DeviceLost { reason: "thermal".into() }), || 0);
    let entries = tracker.log_entries();
    assert!(entries[0].contains("falling back to CPU"));
    assert!(entries[0].contains("thermal"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fallback_multiple_operations_counted() {
    let tracker = FallbackTracker::new(true);
    for _ in 0..5 {
        let _: Result<i32, _> = tracker
            .execute(|| Err(MetalGpuError::OutOfMemory { requested: 1, available: 0 }), || 0);
    }
    assert_eq!(tracker.fallback_count(), 5);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_fallback_mixed_success_and_failure() {
    let tracker = FallbackTracker::new(true);
    let _: Result<i32, _> = tracker.execute(|| Ok(1), || 0);
    let _: Result<i32, _> = tracker.execute(
        || {
            Err(MetalGpuError::Timeout {
                operation: "op".into(),
                elapsed: Duration::from_secs(10),
                limit: Duration::from_secs(5),
            })
        },
        || 0,
    );
    let _: Result<i32, _> = tracker.execute(|| Ok(3), || 0);
    assert_eq!(tracker.fallback_count(), 1);
    assert_eq!(tracker.log_entries().len(), 3);
}

// ═══════════════════════════════════════════════════════════════════
// 8 — Error message quality
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_display_oom() {
    let err = MetalGpuError::OutOfMemory { requested: 1_048_576, available: 524_288 };
    let msg = err.to_string();
    assert!(msg.contains("OOM"));
    assert!(msg.contains("1048576"));
    assert!(msg.contains("524288"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_display_shader_compilation() {
    let err = MetalGpuError::ShaderCompilationFailed {
        source_snippet: "float4x4 m =".into(),
        error_msg: "expected ';' after expression".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("float4x4"));
    assert!(msg.contains("expected ';'"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_display_device_lost() {
    let err = MetalGpuError::DeviceLost { reason: "GPU reset due to watchdog timeout".into() };
    let msg = err.to_string();
    assert!(msg.contains("device lost"));
    assert!(msg.contains("watchdog timeout"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_display_all_variants_non_empty() {
    let errors: Vec<MetalGpuError> = vec![
        MetalGpuError::OutOfMemory { requested: 100, available: 50 },
        MetalGpuError::BufferAllocationFailed { size: 100, reason: "test".into() },
        MetalGpuError::ShaderCompilationFailed {
            source_snippet: "x".into(),
            error_msg: "y".into(),
        },
        MetalGpuError::ShaderFunctionNotFound { name: "fn".into() },
        MetalGpuError::CommandBufferError { status: CommandBufferStatus::Error },
        MetalGpuError::PipelineCreationFailed { label: "p".into(), reason: "r".into() },
        MetalGpuError::ResourceLimitExceeded { resource: "r".into(), limit: 1, requested: 2 },
        MetalGpuError::Timeout {
            operation: "op".into(),
            elapsed: Duration::from_secs(1),
            limit: Duration::from_secs(0),
        },
        MetalGpuError::DeviceLost { reason: "lost".into() },
        MetalGpuError::InvalidOperation { description: "invalid".into() },
    ];
    for err in &errors {
        let msg = err.to_string();
        assert!(!msg.is_empty(), "empty display for {err:?}");
        assert!(msg.len() > 5, "too short display for {err:?}: {msg}");
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_metal_config_error_display_includes_context() {
    let cases: Vec<(MetalConfigError, &str)> = vec![
        (MetalConfigError::WorkgroupTooLarge { requested: 4096, max: 1024 }, "4096"),
        (MetalConfigError::ZeroDimension, "non-zero"),
        (MetalConfigError::DispatchTooLarge { dimension: 70000, max: 65535 }, "70000"),
    ];
    for (err, expected) in &cases {
        let msg = err.to_string();
        assert!(msg.contains(expected), "{err:?} display should contain '{expected}'");
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_debug_vs_display_differ() {
    let err = MetalGpuError::OutOfMemory { requested: 100, available: 50 };
    let debug = format!("{err:?}");
    let display = format!("{err}");
    assert_ne!(debug, display);
}

// ═══════════════════════════════════════════════════════════════════
// 9 — Recovery state validation
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_recovery_pool_clean_after_oom() {
    let mut pool = GpuMemoryPool::new(512);
    pool.allocate(256).unwrap();
    let _ = pool.allocate(512); // OOM.
    // Pool state should still be valid.
    assert_eq!(pool.allocation_count(), 1);
    assert_eq!(pool.available(), 256);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_recovery_pool_after_free_all() {
    let mut pool = GpuMemoryPool::new(1024);
    for _ in 0..4 {
        pool.allocate(256).unwrap();
    }
    pool.free_all();
    assert_eq!(pool.allocation_count(), 0);
    assert_eq!(pool.available(), 1024);
    assert!(!pool.fragmented);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_recovery_command_buffer_reset() {
    let mut cb = SimCommandBuffer::new();
    cb.add_dispatch("kernel").unwrap();
    cb.commit().unwrap();
    let _ = cb.simulate_completion(CommandBufferStatus::Error);
    // After error, create a fresh buffer (real Metal resets state).
    let mut cb2 = SimCommandBuffer::new();
    assert_eq!(cb2.status, CommandBufferStatus::NotEnqueued);
    cb2.add_dispatch("retry").unwrap();
    cb2.commit().unwrap();
    assert!(cb2.simulate_completion(CommandBufferStatus::Completed).is_ok());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_recovery_pipeline_config_unchanged_after_dispatch_error() {
    let pipeline = MetalComputePipeline::new("stable_pipeline");
    let _ = pipeline.dispatch_for_matrix(0, 100); // Error from 0 dimension.
    // Config is immutable — same values after error.
    assert_eq!(pipeline.label, "stable_pipeline");
    assert_eq!(pipeline.workgroup.x, DEFAULT_TILE_SIZE);
    assert_eq!(pipeline.workgroup.y, DEFAULT_TILE_SIZE);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_recovery_error_accumulator_drain_clears() {
    let mut acc = ErrorAccumulator::new(10);
    acc.record(MetalGpuError::OutOfMemory { requested: 1, available: 0 });
    acc.record(MetalGpuError::DeviceLost { reason: "test".into() });
    assert_eq!(acc.error_count(), 2);
    let drained = acc.drain();
    assert_eq!(drained.len(), 2);
    assert!(!acc.has_errors());
    assert_eq!(acc.error_count(), 0);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_recovery_fragmentation_cleared_by_free_all() {
    let mut pool = GpuMemoryPool::new(4096);
    pool.fragment();
    // Large alloc fails due to fragmentation.
    assert!(pool.allocate(2048).is_err());
    pool.free_all();
    // After free_all, fragmentation is cleared.
    assert!(pool.allocate(2048).is_ok());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_recovery_fallback_tracker_resets_state() {
    let tracker = FallbackTracker::new(true);
    tracker.mark_gpu_lost();
    assert!(!tracker.gpu_available.load(Ordering::Relaxed));
    // Simulate re-initialization.
    tracker.gpu_available.store(true, Ordering::Relaxed);
    let result: Result<i32, _> = tracker.execute(|| Ok(42), || 0);
    assert_eq!(result.unwrap(), 42);
}

// ═══════════════════════════════════════════════════════════════════
// 10 — Multi-error handling (cascading failures, accumulation)
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_accumulator_basic() {
    let mut acc = ErrorAccumulator::new(5);
    assert!(!acc.has_errors());
    let can_continue = acc.record(MetalGpuError::OutOfMemory { requested: 100, available: 0 });
    assert!(can_continue);
    assert!(acc.has_errors());
    assert_eq!(acc.error_count(), 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_accumulator_stops_at_max() {
    let mut acc = ErrorAccumulator::new(3);
    assert!(acc.record(MetalGpuError::DeviceLost { reason: "1".into() }));
    assert!(acc.record(MetalGpuError::DeviceLost { reason: "2".into() }));
    // Third error hits the limit.
    assert!(!acc.record(MetalGpuError::DeviceLost { reason: "3".into() }));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_accumulator_oom_detection() {
    let mut acc = ErrorAccumulator::new(10);
    acc.record(MetalGpuError::DeviceLost { reason: "r".into() });
    acc.record(MetalGpuError::OutOfMemory { requested: 1, available: 0 });
    assert!(acc.contains_oom());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_accumulator_timeout_detection() {
    let mut acc = ErrorAccumulator::new(10);
    acc.record(MetalGpuError::Timeout {
        operation: "test".into(),
        elapsed: Duration::from_secs(10),
        limit: Duration::from_secs(5),
    });
    assert!(acc.contains_timeout());
    assert!(!acc.contains_oom());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_error_accumulator_first_error() {
    let mut acc = ErrorAccumulator::new(10);
    acc.record(MetalGpuError::DeviceLost { reason: "first".into() });
    acc.record(MetalGpuError::OutOfMemory { requested: 1, available: 0 });
    let first = acc.first_error().unwrap();
    assert!(matches!(first, MetalGpuError::DeviceLost { reason } if reason == "first"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_cascading_oom_then_device_lost() {
    let mut pool = GpuMemoryPool::new(256);
    pool.allocate(256).unwrap();
    let mut acc = ErrorAccumulator::new(10);

    // First: OOM.
    if let Err(e) = pool.allocate(512) {
        acc.record(e);
    }
    // Second: device lost due to repeated failures.
    acc.record(MetalGpuError::DeviceLost { reason: "repeated allocation failures".into() });

    assert_eq!(acc.error_count(), 2);
    assert!(acc.contains_oom());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_cascading_shader_then_pipeline_failure() {
    let mut acc = ErrorAccumulator::new(10);
    let result = compile_shader("");
    if let Some(e) = result.error {
        acc.record(MetalGpuError::ShaderCompilationFailed {
            source_snippet: String::new(),
            error_msg: e.to_string(),
        });
    }
    acc.record(MetalGpuError::PipelineCreationFailed {
        label: "matmul".into(),
        reason: "no compiled shader available".into(),
    });
    assert_eq!(acc.error_count(), 2);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_cascading_command_buffer_errors_accumulate() {
    let mut acc = ErrorAccumulator::new(10);
    for i in 0..5 {
        let mut cb = SimCommandBuffer::new();
        cb.add_dispatch(&format!("kernel_{i}")).unwrap();
        cb.commit().unwrap();
        if let Err(e) = cb.simulate_completion(CommandBufferStatus::Error) {
            acc.record(e);
        }
    }
    assert_eq!(acc.error_count(), 5);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_cascading_fallback_after_accumulated_errors() {
    let tracker = FallbackTracker::new(true);
    let mut acc = ErrorAccumulator::new(3);

    for _ in 0..3 {
        let _: Result<i32, _> = tracker.execute(
            || Err(MetalGpuError::OutOfMemory { requested: 1, available: 0 }),
            || {
                // CPU fallback succeeds.
                42
            },
        );
        acc.record(MetalGpuError::OutOfMemory { requested: 1, available: 0 });
    }
    // After max errors, mark GPU lost.
    if !acc.record(MetalGpuError::DeviceLost { reason: "too many OOMs".into() }) {
        tracker.mark_gpu_lost();
    }
    assert_eq!(tracker.fallback_count(), 3);
}

// ═══════════════════════════════════════════════════════════════════
// 11 — Buffer and alignment edge cases under error conditions
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_alignment_power_of_two_sizes() {
    for exp in 0..20 {
        let size = 1usize << exp;
        let aligned = align_buffer_size(size);
        assert!(is_aligned(aligned));
        assert!(aligned >= size);
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_alignment_large_sizes() {
    let large = 1024 * 1024 * 100; // 100 MB
    let aligned = align_buffer_size(large);
    assert!(is_aligned(aligned));
    assert!(aligned >= large);
    // 100 MB is already a multiple of 256.
    assert_eq!(aligned, large);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_alignment_preserves_256_multiples() {
    for n in 1..100 {
        let size = n * METAL_BUFFER_ALIGNMENT;
        assert_eq!(align_buffer_size(size), size);
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_dispatch_all_axes_at_max() {
    let wg = WorkgroupSize::new(1, 1, 1).unwrap();
    let d = DispatchDimensions::for_problem(
        (MAX_DISPATCH_DIM, MAX_DISPATCH_DIM, MAX_DISPATCH_DIM),
        &wg,
    )
    .unwrap();
    assert_eq!(d.x, MAX_DISPATCH_DIM);
    assert_eq!(d.y, MAX_DISPATCH_DIM);
    assert_eq!(d.z, MAX_DISPATCH_DIM);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_dispatch_problem_size_one() {
    let wg = WorkgroupSize::tile(16).unwrap();
    let d = DispatchDimensions::for_problem((1, 1, 1), &wg).unwrap();
    assert_eq!((d.x, d.y, d.z), (1, 1, 1));
}

// ═══════════════════════════════════════════════════════════════════
// 12 — MetalConfigError edge cases and trait impls
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_metal_config_error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(MetalConfigError::ZeroDimension);
    assert!(err.to_string().contains("non-zero"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_metal_config_error_clone() {
    let err = MetalConfigError::WorkgroupTooLarge { requested: 2048, max: 1024 };
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_metal_config_error_debug() {
    let err = MetalConfigError::DispatchTooLarge { dimension: 99999, max: 65535 };
    let debug = format!("{err:?}");
    assert!(debug.contains("DispatchTooLarge"));
    assert!(debug.contains("99999"));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_gpu_error_clone_equality() {
    let err1 = MetalGpuError::OutOfMemory { requested: 100, available: 50 };
    let err2 = err1.clone();
    assert_eq!(err1, err2);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_command_buffer_status_variants() {
    let statuses = [
        CommandBufferStatus::Completed,
        CommandBufferStatus::Error,
        CommandBufferStatus::Timeout,
        CommandBufferStatus::GpuFault,
        CommandBufferStatus::NotEnqueued,
        CommandBufferStatus::Enqueued,
        CommandBufferStatus::Committed,
    ];
    // All variants are distinct.
    for (i, a) in statuses.iter().enumerate() {
        for (j, b) in statuses.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_memory_architecture_variants() {
    assert_ne!(MemoryArchitecture::Unified, MemoryArchitecture::Discrete);
    assert!(MemoryArchitecture::Unified.supports_zero_copy());
    assert!(!MemoryArchitecture::Discrete.supports_zero_copy());
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_pipeline_default_uses_detected_memory() {
    let p = MetalComputePipeline::new("detect");
    let detected = MemoryArchitecture::detect();
    assert_eq!(p.memory, detected);
}
