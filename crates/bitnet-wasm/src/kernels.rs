//! WebAssembly-optimized CPU kernels for BitNet inference

use std::arch::wasm32::*;
use wasm_bindgen::prelude::*;

use bitnet_common::BitNetError;
use bitnet_quantization::QuantizationType;

/// WebAssembly-optimized kernel provider
pub struct WasmKernelProvider {
    simd_supported: bool,
    bulk_memory_supported: bool,
}

impl WasmKernelProvider {
    /// Create a new WASM kernel provider with feature detection
    pub fn new() -> Self {
        let simd_supported = Self::detect_simd_support();
        let bulk_memory_supported = Self::detect_bulk_memory_support();

        web_sys::console::log_1(
            &format!(
                "WASM kernels initialized - SIMD: {}, Bulk Memory: {}",
                simd_supported, bulk_memory_supported
            )
            .into(),
        );

        WasmKernelProvider {
            simd_supported,
            bulk_memory_supported,
        }
    }

    /// Detect WASM SIMD support
    fn detect_simd_support() -> bool {
        // In a real implementation, this would check for WASM SIMD support
        // For now, we'll assume it's not available for maximum compatibility
        false
    }

    /// Detect bulk memory operations support
    fn detect_bulk_memory_support() -> bool {
        // Most modern browsers support bulk memory operations
        true
    }

    /// Optimized matrix multiplication for I2S quantization
    pub fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BitNetError> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(BitNetError::Kernel("Invalid matrix dimensions".into()));
        }

        if self.simd_supported {
            self.matmul_i2s_simd(a, b, c, m, n, k)
        } else {
            self.matmul_i2s_scalar(a, b, c, m, n, k)
        }
    }

    /// SIMD-optimized matrix multiplication (when available)
    fn matmul_i2s_simd(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BitNetError> {
        // WASM SIMD implementation using v128 operations
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let mut acc = f32x4_splat(0.0);

                for l in (0..k).step_by(16) {
                    if l + 16 <= k && j + 4 <= n {
                        // Load 16 i8 values from matrix A
                        let a_base = i * k + l;
                        let a_vec = v128_load(&a[a_base] as *const i8 as *const v128);

                        // Load and process 4 columns from matrix B
                        for col in 0..4 {
                            if j + col < n {
                                let b_base = l * n + j + col;
                                let b_vec = v128_load(&b[b_base] as *const u8 as *const v128);

                                // Convert and multiply (simplified)
                                let prod = self.simd_multiply_i8_u8(a_vec, b_vec);
                                acc = f32x4_add(acc, prod);
                            }
                        }
                    } else {
                        // Handle remaining elements with scalar code
                        for col in 0..4 {
                            if j + col < n {
                                let mut sum = 0.0f32;
                                for elem in l..k.min(l + 16) {
                                    let a_val = a[i * k + elem] as f32;
                                    let b_val = b[elem * n + j + col] as f32;
                                    sum += a_val * b_val;
                                }
                                c[i * n + j + col] += sum;
                            }
                        }
                    }
                }

                // Store results
                if j + 4 <= n {
                    v128_store(&mut c[i * n + j] as *mut f32 as *mut v128, acc);
                } else {
                    // Handle remaining columns
                    let results = [
                        f32x4_extract_lane::<0>(acc),
                        f32x4_extract_lane::<1>(acc),
                        f32x4_extract_lane::<2>(acc),
                        f32x4_extract_lane::<3>(acc),
                    ];
                    for col in 0..4 {
                        if j + col < n {
                            c[i * n + j + col] = results[col];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Scalar matrix multiplication fallback
    fn matmul_i2s_scalar(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BitNetError> {
        // Memory-efficient scalar implementation
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;

                // Unroll inner loop for better performance
                let mut l = 0;
                while l + 4 <= k {
                    let a_base = i * k + l;
                    let b_base = l * n + j;

                    sum += (a[a_base] as f32) * (b[b_base] as f32);
                    sum += (a[a_base + 1] as f32) * (b[b_base + n] as f32);
                    sum += (a[a_base + 2] as f32) * (b[b_base + 2 * n] as f32);
                    sum += (a[a_base + 3] as f32) * (b[b_base + 3 * n] as f32);

                    l += 4;
                }

                // Handle remaining elements
                while l < k {
                    sum += (a[i * k + l] as f32) * (b[l * n + j] as f32);
                    l += 1;
                }

                c[i * n + j] = sum;
            }
        }

        Ok(())
    }

    /// SIMD multiply helper for i8 and u8 vectors
    fn simd_multiply_i8_u8(&self, a: v128, b: v128) -> f32x4 {
        // Convert i8 to i16, then to i32, then to f32
        let a_low = i16x8_extend_low_i8x16(a);
        let a_high = i16x8_extend_high_i8x16(a);

        // Convert u8 to u16, then to u32, then to f32
        let b_low = u16x8_extend_low_u8x16(b);
        let b_high = u16x8_extend_high_u8x16(b);

        // Multiply and accumulate (simplified - real implementation would be more complex)
        let prod_low = i32x4_extend_low_i16x8(i16x8_mul(
            a_low,
            i16x8_narrow_i32x4(
                i32x4_extend_low_u16x8(b_low),
                i32x4_extend_high_u16x8(b_low),
            ),
        ));
        f32x4_convert_i32x4(prod_low)
    }

    /// Memory-efficient quantization for WASM
    pub fn quantize_wasm(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
        block_size: usize,
    ) -> Result<(), BitNetError> {
        match qtype {
            QuantizationType::I2S => self.quantize_i2s_wasm(input, output, scales, block_size),
            QuantizationType::TL1 => self.quantize_tl1_wasm(input, output, scales, block_size),
            QuantizationType::TL2 => self.quantize_tl2_wasm(input, output, scales, block_size),
        }
    }

    /// I2S quantization optimized for WASM
    fn quantize_i2s_wasm(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        block_size: usize,
    ) -> Result<(), BitNetError> {
        let num_blocks = (input.len() + block_size - 1) / block_size;

        if scales.len() < num_blocks {
            return Err(BitNetError::Quantization(
                "Insufficient scale buffer".into(),
            ));
        }

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(input.len());
            let block = &input[start..end];

            // Find scale (maximum absolute value)
            let mut scale = 0.0f32;
            for &val in block {
                scale = scale.max(val.abs());
            }

            if scale == 0.0 {
                scale = 1.0; // Avoid division by zero
            }

            scales[block_idx] = scale;
            let inv_scale = 1.0 / scale;

            // Quantize to 2-bit signed values and pack
            let output_start = block_idx * (block_size / 4); // 4 values per byte
            for (i, chunk) in block.chunks(4).enumerate() {
                let mut packed = 0u8;

                for (j, &val) in chunk.iter().enumerate() {
                    let quantized = ((val * inv_scale).clamp(-1.0, 1.0) * 1.5 + 1.5) as u8;
                    let quantized_2bit = quantized.min(3);
                    packed |= quantized_2bit << (j * 2);
                }

                if output_start + i < output.len() {
                    output[output_start + i] = packed;
                }
            }
        }

        Ok(())
    }

    /// TL1 quantization (simplified for WASM)
    fn quantize_tl1_wasm(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        block_size: usize,
    ) -> Result<(), BitNetError> {
        // Simplified TL1 implementation for WASM
        // In a full implementation, this would generate lookup tables
        self.quantize_i2s_wasm(input, output, scales, block_size)
    }

    /// TL2 quantization (simplified for WASM)
    fn quantize_tl2_wasm(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        block_size: usize,
    ) -> Result<(), BitNetError> {
        // Simplified TL2 implementation for WASM
        // In a full implementation, this would generate lookup tables
        self.quantize_i2s_wasm(input, output, scales, block_size)
    }

    /// Memory-efficient tensor operations
    pub fn tensor_add_inplace(&self, a: &mut [f32], b: &[f32]) -> Result<(), BitNetError> {
        if a.len() != b.len() {
            return Err(BitNetError::Kernel("Tensor dimension mismatch".into()));
        }

        if self.simd_supported {
            self.tensor_add_simd(a, b)
        } else {
            self.tensor_add_scalar(a, b)
        }
    }

    /// SIMD tensor addition
    fn tensor_add_simd(&self, a: &mut [f32], b: &[f32]) -> Result<(), BitNetError> {
        let mut i = 0;

        // Process 4 elements at a time with SIMD
        while i + 4 <= a.len() {
            let a_vec = v128_load(&a[i] as *const f32 as *const v128);
            let b_vec = v128_load(&b[i] as *const f32 as *const v128);
            let result = f32x4_add(a_vec, b_vec);
            v128_store(&mut a[i] as *mut f32 as *mut v128, result);
            i += 4;
        }

        // Handle remaining elements
        while i < a.len() {
            a[i] += b[i];
            i += 1;
        }

        Ok(())
    }

    /// Scalar tensor addition
    fn tensor_add_scalar(&self, a: &mut [f32], b: &[f32]) -> Result<(), BitNetError> {
        for (a_val, &b_val) in a.iter_mut().zip(b.iter()) {
            *a_val += b_val;
        }
        Ok(())
    }

    /// Get kernel performance characteristics
    pub fn get_performance_info(&self) -> KernelPerformanceInfo {
        KernelPerformanceInfo {
            simd_supported: self.simd_supported,
            bulk_memory_supported: self.bulk_memory_supported,
            estimated_gflops: if self.simd_supported { 2.0 } else { 0.5 },
            memory_bandwidth_gbps: 1.0, // Conservative estimate for WASM
        }
    }
}

impl Default for WasmKernelProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance information for WASM kernels
#[wasm_bindgen]
pub struct KernelPerformanceInfo {
    simd_supported: bool,
    bulk_memory_supported: bool,
    estimated_gflops: f64,
    memory_bandwidth_gbps: f64,
}

#[wasm_bindgen]
impl KernelPerformanceInfo {
    #[wasm_bindgen(getter)]
    pub fn simd_supported(&self) -> bool {
        self.simd_supported
    }

    #[wasm_bindgen(getter)]
    pub fn bulk_memory_supported(&self) -> bool {
        self.bulk_memory_supported
    }

    #[wasm_bindgen(getter)]
    pub fn estimated_gflops(&self) -> f64 {
        self.estimated_gflops
    }

    #[wasm_bindgen(getter)]
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        self.memory_bandwidth_gbps
    }
}

/// Benchmark utilities for WASM kernels
#[wasm_bindgen]
pub struct WasmBenchmark {
    kernel_provider: WasmKernelProvider,
}

#[wasm_bindgen]
impl WasmBenchmark {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmBenchmark {
        WasmBenchmark {
            kernel_provider: WasmKernelProvider::new(),
        }
    }

    /// Benchmark matrix multiplication performance
    #[wasm_bindgen]
    pub fn benchmark_matmul(&self, m: usize, n: usize, k: usize, iterations: usize) -> f64 {
        let a = vec![1i8; m * k];
        let b = vec![1u8; k * n];
        let mut c = vec![0.0f32; m * n];

        let start_time = web_sys::window()
            .and_then(|w| w.performance().ok())
            .map(|p| p.now())
            .unwrap_or_else(|| js_sys::Date::now());

        for _ in 0..iterations {
            let _ = self.kernel_provider.matmul_i2s(&a, &b, &mut c, m, n, k);
        }

        let end_time = web_sys::window()
            .and_then(|w| w.performance().ok())
            .map(|p| p.now())
            .unwrap_or_else(|| js_sys::Date::now());

        let elapsed_ms = end_time - start_time;
        let ops_per_iteration = 2.0 * (m * n * k) as f64; // Multiply-add operations
        let total_ops = ops_per_iteration * iterations as f64;
        let gflops = (total_ops / 1e9) / (elapsed_ms / 1000.0);

        web_sys::console::log_1(
            &format!(
                "Matrix multiplication benchmark: {:.2} GFLOPS ({} iterations, {:.2}ms total)",
                gflops, iterations, elapsed_ms
            )
            .into(),
        );

        gflops
    }

    /// Benchmark quantization performance
    #[wasm_bindgen]
    pub fn benchmark_quantization(&self, size: usize, iterations: usize) -> f64 {
        let input = vec![1.0f32; size];
        let mut output = vec![0u8; size / 4];
        let mut scales = vec![0.0f32; (size + 255) / 256]; // One scale per 256 elements

        let start_time = web_sys::window()
            .and_then(|w| w.performance().ok())
            .map(|p| p.now())
            .unwrap_or_else(|| js_sys::Date::now());

        for _ in 0..iterations {
            let _ = self.kernel_provider.quantize_wasm(
                &input,
                &mut output,
                &mut scales,
                QuantizationType::I2S,
                256,
            );
        }

        let end_time = web_sys::window()
            .and_then(|w| w.performance().ok())
            .map(|p| p.now())
            .unwrap_or_else(|| js_sys::Date::now());

        let elapsed_ms = end_time - start_time;
        let elements_per_second = (size * iterations) as f64 / (elapsed_ms / 1000.0);
        let throughput_gbps = (elements_per_second * 4.0) / 1e9; // 4 bytes per f32

        web_sys::console::log_1(
            &format!(
                "Quantization benchmark: {:.2} GB/s ({} iterations, {:.2}ms total)",
                throughput_gbps, iterations, elapsed_ms
            )
            .into(),
        );

        throughput_gbps
    }
}

impl Default for WasmBenchmark {
    fn default() -> Self {
        Self::new()
    }
}
