//! GPU test harness for running kernel tests without hardware.
//!
//! Provides mock GPU contexts, reference CPU implementations,
//! and numerical comparison utilities.

use std::collections::HashMap;

/// Mock GPU context that executes kernels on CPU.
pub struct MockGpuContext {
    device_name: String,
    total_memory: u64,
    buffers: Vec<Vec<u8>>,
    kernels: HashMap<String, MockKernel>,
}

/// Handle to a buffer allocated in the mock context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferHandle(pub usize);

/// Mock kernel that delegates to a CPU reference implementation.
pub struct MockKernel {
    #[allow(dead_code)]
    name: String,
    reference_fn: Box<dyn Fn(&[&[u8]], &mut [u8]) -> Result<(), String>>,
}

/// Result of numerical comparison between expected and actual values.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub max_abs_diff: f32,
    pub max_rel_diff: f32,
    pub num_mismatches: usize,
    pub worst_index: usize,
}

/// Numerical comparison utilities for GPU output validation.
#[derive(Debug, Clone)]
pub struct NumericalValidator {
    pub abs_tolerance: f32,
    pub rel_tolerance: f32,
    pub max_ulp_diff: u32,
}

impl NumericalValidator {
    /// Strict tolerances suitable for exact-match kernels.
    pub fn strict() -> Self {
        Self {
            abs_tolerance: 1e-6,
            rel_tolerance: 1e-5,
            max_ulp_diff: 4,
        }
    }

    /// Relaxed tolerances for kernels with known numerical drift.
    pub fn relaxed() -> Self {
        Self {
            abs_tolerance: 1e-3,
            rel_tolerance: 1e-2,
            max_ulp_diff: 64,
        }
    }

    /// Compare two f32 slices element-wise.
    pub fn compare_f32(
        &self,
        expected: &[f32],
        actual: &[f32],
    ) -> ValidationResult {
        assert_eq!(
            expected.len(),
            actual.len(),
            "length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        );

        let mut max_abs_diff: f32 = 0.0;
        let mut max_rel_diff: f32 = 0.0;
        let mut num_mismatches: usize = 0;
        let mut worst_index: usize = 0;

        for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            let abs_diff = (e - a).abs();
            let rel_diff = if e.abs() > f32::EPSILON {
                abs_diff / e.abs()
            } else {
                abs_diff
            };
            let ulp = ulp_diff_f32(e, a);

            let mismatch = abs_diff > self.abs_tolerance
                && rel_diff > self.rel_tolerance
                && ulp > self.max_ulp_diff;

            if mismatch {
                num_mismatches += 1;
            }

            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
                worst_index = i;
            }
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
        }

        ValidationResult {
            passed: num_mismatches == 0,
            max_abs_diff,
            max_rel_diff,
            num_mismatches,
            worst_index,
        }
    }

    /// Compare two f16 slices (stored as `u16` bit patterns).
    pub fn compare_f16(
        &self,
        expected: &[u16],
        actual: &[u16],
    ) -> ValidationResult {
        let exp_f32: Vec<f32> = expected.iter().map(|&v| f16_to_f32(v)).collect();
        let act_f32: Vec<f32> = actual.iter().map(|&v| f16_to_f32(v)).collect();
        self.compare_f32(&exp_f32, &act_f32)
    }
}

/// Compute ULP distance between two f32 values.
fn ulp_diff_f32(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    (ai.wrapping_sub(bi)).unsigned_abs()
}

/// Convert an f16 bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        let val = (mant as f32) * (1.0 / (1 << 24) as f32);
        if sign == 1 { -val } else { val }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let f_exp = (exp as i32) - 15 + 127;
        let f_bits = (sign << 31) | ((f_exp as u32) << 23) | (mant << 13);
        f32::from_bits(f_bits)
    }
}

impl MockGpuContext {
    /// Create a new mock context with the given device name and memory size.
    pub fn new(device_name: &str, total_memory: u64) -> Self {
        Self {
            device_name: device_name.to_string(),
            total_memory,
            buffers: Vec::new(),
            kernels: HashMap::new(),
        }
    }

    /// Return the device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Return total device memory in bytes.
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }

    /// Allocate a buffer of the given size, returning a handle.
    pub fn alloc_buffer(&mut self, size: usize) -> BufferHandle {
        let idx = self.buffers.len();
        self.buffers.push(vec![0u8; size]);
        BufferHandle(idx)
    }

    /// Write data to a buffer.
    pub fn write_buffer(
        &mut self,
        handle: BufferHandle,
        data: &[u8],
    ) -> Result<(), String> {
        let buf = self
            .buffers
            .get_mut(handle.0)
            .ok_or_else(|| format!("invalid buffer handle {}", handle.0))?;
        if data.len() > buf.len() {
            return Err(format!(
                "data size {} exceeds buffer size {}",
                data.len(),
                buf.len()
            ));
        }
        buf[..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read data from a buffer.
    pub fn read_buffer(
        &self,
        handle: BufferHandle,
    ) -> Result<&[u8], String> {
        self.buffers
            .get(handle.0)
            .map(|b| b.as_slice())
            .ok_or_else(|| format!("invalid buffer handle {}", handle.0))
    }

    /// Register a mock kernel with its CPU reference function.
    pub fn register_kernel<F>(&mut self, name: &str, func: F)
    where
        F: Fn(&[&[u8]], &mut [u8]) -> Result<(), String> + 'static,
    {
        self.kernels.insert(
            name.to_string(),
            MockKernel {
                name: name.to_string(),
                reference_fn: Box::new(func),
            },
        );
    }

    /// Execute a registered kernel with the given input/output buffers.
    pub fn execute_kernel(
        &mut self,
        name: &str,
        input_handles: &[BufferHandle],
        output_handle: BufferHandle,
    ) -> Result<(), String> {
        let kernel = self
            .kernels
            .get(name)
            .ok_or_else(|| format!("kernel '{}' not registered", name))?;

        let inputs: Vec<&[u8]> = input_handles
            .iter()
            .map(|h| {
                self.buffers
                    .get(h.0)
                    .map(|b| b.as_slice())
                    .ok_or_else(|| format!("invalid input buffer {}", h.0))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Clone output buffer so we can pass &mut without aliasing.
        let mut out = self.buffers[output_handle.0].clone();
        (kernel.reference_fn)(&inputs, &mut out)?;
        self.buffers[output_handle.0] = out;
        Ok(())
    }

    /// Number of allocated buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ulp_diff_identical() {
        assert_eq!(ulp_diff_f32(1.0, 1.0), 0);
    }

    #[test]
    fn test_ulp_diff_nan() {
        assert_eq!(ulp_diff_f32(f32::NAN, 1.0), u32::MAX);
    }

    #[test]
    fn test_f16_roundtrip_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_one() {
        // f16 1.0 = 0x3C00
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);
    }
}
