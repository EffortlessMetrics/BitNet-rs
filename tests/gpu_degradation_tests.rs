//! Graceful GPU failure degradation tests.
//!
//! Verifies that GPU failures are handled gracefully with proper fallbacks,
//! clean recovery, and meaningful error messages.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Simulated GPU error types for testing degradation behavior.
#[derive(Debug, Clone, PartialEq)]
enum GpuFailure {
    OutOfMemory { requested: usize, available: usize },
    DeviceLost { reason: String },
    KernelCompilationFailed { kernel: String, error: String },
    InvalidDeviceIndex { index: u32, max: u32 },
    Timeout { operation: String },
}

impl std::fmt::Display for GpuFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory { requested, available } => {
                write!(f, "GPU OOM: requested {} bytes but only {} available", requested, available)
            }
            Self::DeviceLost { reason } => write!(f, "GPU device lost: {reason}"),
            Self::KernelCompilationFailed { kernel, error } => {
                write!(f, "Kernel '{kernel}' compilation failed: {error}")
            }
            Self::InvalidDeviceIndex { index, max } => {
                write!(f, "Invalid GPU device index {index} (max: {max})")
            }
            Self::Timeout { operation } => write!(f, "GPU timeout during: {operation}"),
        }
    }
}

/// Result of a compute operation that may fall back to CPU.
#[derive(Debug, Clone, PartialEq)]
enum ComputeResult {
    GpuSuccess(Vec<f32>),
    CpuFallback(Vec<f32>),
    ScalarFallback(Vec<f32>),
    Error(String),
}

/// Mock matmul that simulates GPU OOM and falls back to CPU.
fn matmul_with_fallback(
    a: &[f32],
    b: &[f32],
    n: usize,
    gpu_available_memory: usize,
) -> ComputeResult {
    let required_memory = n * n * std::mem::size_of::<f32>() * 3; // a, b, result
    if required_memory > gpu_available_memory {
        // Fall back to CPU
        let mut result = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += a[i * n + k] * b[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        return ComputeResult::CpuFallback(result);
    }
    // Simulate GPU success
    let mut result = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    ComputeResult::GpuSuccess(result)
}

/// Mock kernel compilation with fallback to scalar.
fn compile_kernel_with_fallback(source: &str, allow_scalar: bool) -> ComputeResult {
    // Simulate compilation failure
    if source.contains("INVALID_SYNTAX") {
        if allow_scalar {
            return ComputeResult::ScalarFallback(vec![1.0]);
        }
        return ComputeResult::Error("Kernel compilation failed and no fallback available".into());
    }
    ComputeResult::GpuSuccess(vec![1.0])
}

/// Mock device context that can simulate failures.
struct MockGpuContext {
    device_lost: AtomicBool,
    available_memory: Mutex<usize>,
    device_index: u32,
    max_devices: u32,
    operation_count: AtomicU32,
    errors: Mutex<Vec<GpuFailure>>,
}

impl std::fmt::Debug for MockGpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockGpuContext")
            .field("device_index", &self.device_index)
            .field("max_devices", &self.max_devices)
            .finish()
    }
}

impl MockGpuContext {
    fn new(device_index: u32, max_devices: u32, memory: usize) -> Result<Self, GpuFailure> {
        if device_index >= max_devices {
            return Err(GpuFailure::InvalidDeviceIndex {
                index: device_index,
                max: max_devices.saturating_sub(1),
            });
        }
        Ok(Self {
            device_lost: AtomicBool::new(false),
            available_memory: Mutex::new(memory),
            device_index,
            max_devices,
            operation_count: AtomicU32::new(0),
            errors: Mutex::new(Vec::new()),
        })
    }

    fn simulate_device_lost(&self, reason: &str) {
        self.device_lost.store(true, Ordering::SeqCst);
        let mut errors = self.errors.lock().unwrap();
        errors.push(GpuFailure::DeviceLost { reason: reason.to_string() });
    }

    fn is_device_available(&self) -> bool {
        !self.device_lost.load(Ordering::SeqCst)
    }

    fn try_allocate(&self, bytes: usize) -> Result<(), GpuFailure> {
        if !self.is_device_available() {
            return Err(GpuFailure::DeviceLost { reason: "device previously lost".into() });
        }
        let mut mem = self.available_memory.lock().unwrap();
        if bytes > *mem {
            let err = GpuFailure::OutOfMemory { requested: bytes, available: *mem };
            let mut errors = self.errors.lock().unwrap();
            errors.push(err.clone());
            return Err(err);
        }
        *mem -= bytes;
        self.operation_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn reset_device(&self, new_memory: usize) {
        self.device_lost.store(false, Ordering::SeqCst);
        *self.available_memory.lock().unwrap() = new_memory;
    }

    fn error_count(&self) -> usize {
        self.errors.lock().unwrap().len()
    }

    fn last_error(&self) -> Option<GpuFailure> {
        self.errors.lock().unwrap().last().cloned()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_gpu_oom_during_matmul_falls_back_to_cpu() {
    let n = 4;
    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];

    // Very limited GPU memory → OOM → CPU fallback
    let result = matmul_with_fallback(&a, &b, n, 16);
    match &result {
        ComputeResult::CpuFallback(data) => {
            assert_eq!(data.len(), n * n);
            // Each element of identity * ones_matrix = n
            assert!((data[0] - n as f32).abs() < f32::EPSILON);
        }
        other => panic!("Expected CpuFallback, got {other:?}"),
    }
}

#[test]
fn test_gpu_oom_cpu_fallback_produces_correct_results() {
    let n = 2;
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let gpu_result = matmul_with_fallback(&a, &b, n, 1_000_000);
    let cpu_result = matmul_with_fallback(&a, &b, n, 0);

    let gpu_data = match gpu_result {
        ComputeResult::GpuSuccess(d) => d,
        other => panic!("Expected GpuSuccess, got {other:?}"),
    };
    let cpu_data = match cpu_result {
        ComputeResult::CpuFallback(d) => d,
        other => panic!("Expected CpuFallback, got {other:?}"),
    };

    // GPU and CPU fallback should produce identical results
    assert_eq!(gpu_data, cpu_data);
    assert!((cpu_data[0] - 19.0).abs() < f32::EPSILON);
    assert!((cpu_data[1] - 22.0).abs() < f32::EPSILON);
    assert!((cpu_data[2] - 43.0).abs() < f32::EPSILON);
    assert!((cpu_data[3] - 50.0).abs() < f32::EPSILON);
}

#[test]
fn test_device_lost_during_inference_clean_recovery() {
    let ctx = MockGpuContext::new(0, 2, 1_000_000).unwrap();

    // First operation succeeds
    assert!(ctx.try_allocate(1024).is_ok());
    assert!(ctx.is_device_available());

    // Simulate device loss
    ctx.simulate_device_lost("thermal shutdown");
    assert!(!ctx.is_device_available());

    // Subsequent operations fail with DeviceLost
    let err = ctx.try_allocate(1024).unwrap_err();
    assert!(matches!(err, GpuFailure::DeviceLost { .. }));

    // Recovery: reset device
    ctx.reset_device(1_000_000);
    assert!(ctx.is_device_available());
    assert!(ctx.try_allocate(1024).is_ok());
}

#[test]
fn test_kernel_compilation_failure_fallback_to_scalar() {
    let result = compile_kernel_with_fallback("__kernel void bad() { INVALID_SYNTAX }", true);
    assert!(matches!(result, ComputeResult::ScalarFallback(_)));
}

#[test]
fn test_kernel_compilation_failure_no_fallback_gives_error() {
    let result = compile_kernel_with_fallback("__kernel void bad() { INVALID_SYNTAX }", false);
    match result {
        ComputeResult::Error(msg) => {
            assert!(msg.contains("compilation failed"));
            assert!(msg.contains("no fallback"));
        }
        other => panic!("Expected Error, got {other:?}"),
    }
}

#[test]
fn test_valid_kernel_compilation_succeeds() {
    let result =
        compile_kernel_with_fallback("__kernel void good(global float* x) { x[0] = 1.0f; }", true);
    assert!(matches!(result, ComputeResult::GpuSuccess(_)));
}

#[test]
fn test_invalid_device_index_meaningful_error() {
    let err = MockGpuContext::new(5, 2, 1_000_000).unwrap_err();
    match err {
        GpuFailure::InvalidDeviceIndex { index, max } => {
            assert_eq!(index, 5);
            assert_eq!(max, 1);
            let msg = format!("{err}");
            assert!(msg.contains("Invalid GPU device index 5"));
            assert!(msg.contains("max: 1"));
        }
        other => panic!("Expected InvalidDeviceIndex, got {other:?}"),
    }
}

#[test]
fn test_invalid_device_index_zero_devices() {
    let err = MockGpuContext::new(0, 0, 0).unwrap_err();
    assert!(matches!(err, GpuFailure::InvalidDeviceIndex { index: 0, .. }));
}

#[test]
fn test_concurrent_gpu_errors_no_panic() {
    let ctx = Arc::new(MockGpuContext::new(0, 1, 256).unwrap());
    let mut handles = Vec::new();

    for _ in 0..10 {
        let ctx_clone = ctx.clone();
        let handle = std::thread::spawn(move || {
            // All threads try to allocate more than available
            let _ = ctx_clone.try_allocate(1024);
        });
        handles.push(handle);
    }

    // All threads should complete without panic
    for handle in handles {
        handle.join().expect("thread should not panic");
    }

    // Context should still be usable
    assert!(ctx.is_device_available());
}

#[test]
fn test_concurrent_device_lost_and_allocate_no_corruption() {
    let ctx = Arc::new(MockGpuContext::new(0, 1, 10_000_000).unwrap());
    let mut handles = Vec::new();

    // Half the threads do allocations, half trigger device loss
    for i in 0..20 {
        let ctx_clone = ctx.clone();
        let handle = std::thread::spawn(move || {
            if i % 2 == 0 {
                let _ = ctx_clone.try_allocate(100);
            } else {
                ctx_clone.simulate_device_lost("concurrent test");
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("thread should not panic");
    }

    // After concurrent chaos, device state should be consistent
    // (device_lost should be true since some threads triggered it)
    assert!(!ctx.is_device_available());

    // Recovery still works
    ctx.reset_device(10_000_000);
    assert!(ctx.is_device_available());
}

#[test]
fn test_oom_error_message_includes_sizes() {
    let ctx = MockGpuContext::new(0, 1, 1000).unwrap();
    let err = ctx.try_allocate(2000).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("2000"));
    assert!(msg.contains("1000"));
}

#[test]
fn test_device_lost_error_preserves_reason() {
    let ctx = MockGpuContext::new(0, 1, 1000).unwrap();
    ctx.simulate_device_lost("ECC memory error detected");
    let last = ctx.last_error().unwrap();
    match last {
        GpuFailure::DeviceLost { reason } => {
            assert_eq!(reason, "ECC memory error detected");
        }
        other => panic!("Expected DeviceLost, got {other:?}"),
    }
}

#[test]
fn test_error_accumulation_tracking() {
    let ctx = MockGpuContext::new(0, 1, 100).unwrap();
    assert_eq!(ctx.error_count(), 0);

    let _ = ctx.try_allocate(200); // OOM
    assert_eq!(ctx.error_count(), 1);

    let _ = ctx.try_allocate(300); // OOM again
    assert_eq!(ctx.error_count(), 2);

    ctx.simulate_device_lost("test");
    assert_eq!(ctx.error_count(), 3);
}

#[test]
fn test_multiple_fallback_scenarios_in_sequence() {
    let n = 2;
    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];

    // Scenario 1: GPU succeeds
    let r1 = matmul_with_fallback(&a, &b, n, 1_000_000);
    assert!(matches!(r1, ComputeResult::GpuSuccess(_)));

    // Scenario 2: GPU OOM → CPU fallback
    let r2 = matmul_with_fallback(&a, &b, n, 0);
    assert!(matches!(r2, ComputeResult::CpuFallback(_)));

    // Scenario 3: Kernel fails → scalar fallback
    let r3 = compile_kernel_with_fallback("INVALID_SYNTAX code", true);
    assert!(matches!(r3, ComputeResult::ScalarFallback(_)));

    // All fallbacks produce valid results
    match (r1, r2) {
        (ComputeResult::GpuSuccess(d1), ComputeResult::CpuFallback(d2)) => {
            assert_eq!(d1, d2);
        }
        _ => unreachable!(),
    }
}
