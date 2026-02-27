//! Backend abstraction for IQ2_S quantization to support both FFI and pure-Rust implementations
//!
//! This module provides a unified interface for IQ2_S quantization that can use either:
//! 1. **Pure Rust implementation** (default): Always available, handles partial blocks
//! 2. **GGML FFI implementation** (optional): Requires `iq2s-ffi` feature, may have stricter requirements
//!
//! ## Feature Configuration
//!
//! To enable IQ2_S FFI support, build with:
//! ```bash
//! cargo build --features iq2s-ffi
//! ```
//!
//! Or from the root workspace:
//! ```bash
//! cargo build --no-default-features --features iq2s-ffi
//! ```
//!
//! ## Backend Selection
//!
//! The backend is selected at runtime via the `BITNET_IQ2S_IMPL` environment variable:
//! - `BITNET_IQ2S_IMPL=rust` (default): Use pure Rust implementation
//! - `BITNET_IQ2S_IMPL=ffi`: Use GGML FFI implementation (if compiled with `iq2s-ffi` feature)
//!
//! ## Implementation Notes
//!
//! - The Rust implementation can handle partial blocks and arbitrary element counts
//! - The FFI implementation requires element counts to be multiples of QK_IQ2_S (256)
//! - Both implementations should produce similar results for compatible inputs
//! - Use [`Iq2sBackend::is_available()`] to check if a backend is available at runtime

use core::ffi::c_void;
use core::mem::{align_of, size_of};
use half::f16;

// --- Optional FFI shims: compile only with feature
#[cfg(feature = "iq2s-ffi")]
#[inline]
fn ffi_qk() -> usize {
    bitnet_ggml_ffi::iq2s_qk()
}

#[cfg(not(feature = "iq2s-ffi"))]
#[inline]
fn ffi_qk() -> usize {
    unreachable!("built without feature `iq2s-ffi`")
}

#[cfg(feature = "iq2s-ffi")]
#[inline]
fn ffi_block_bytes() -> usize {
    bitnet_ggml_ffi::iq2s_bytes_per_block()
}

#[cfg(not(feature = "iq2s-ffi"))]
#[inline]
fn ffi_block_bytes() -> usize {
    unreachable!("built without feature `iq2s-ffi`")
}

#[cfg(feature = "iq2s-ffi")]
#[inline]
fn ffi_dequant_row(src: *const c_void, dst: *mut f32, n: usize) {
    unsafe { bitnet_ggml_ffi::dequantize_row_iq2_s(src, dst, n) }
}

#[cfg(not(feature = "iq2s-ffi"))]
#[inline]
#[allow(dead_code)]
fn ffi_dequant_row(_src: *const c_void, _dst: *mut f32, _n: usize) {
    unreachable!("built without feature `iq2s-ffi`");
}

/// Rust representation of the C `block_iq2_s` used by the GGML backend.
///
/// The C struct has the following fields for `QK_IQ2_S = 256`:
/// - `d`: fp16 block scale (2 bytes)
/// - `qs`: 2-bit values packed four per byte (64 bytes)
/// - `qh`: unused high bits (8 bytes)
/// - `scales`: unused additional scales (8 bytes)
///
/// This yields a total size of 82 bytes with a required alignment of 2. The
/// layout must match exactly to safely share data between the Rust and FFI
/// implementations.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct BlockIq2S {
    d: u16,
    qs: [u8; 64],
    qh: [u8; 8],
    scales: [u8; 8],
}

impl Default for BlockIq2S {
    fn default() -> Self {
        Self { d: 0, qs: [0u8; 64], qh: [0u8; 8], scales: [0u8; 8] }
    }
}

// Compile-time checks to ensure parity with the C layout.
const _: () = assert!(size_of::<BlockIq2S>() == 82);
const _: () = assert!(align_of::<BlockIq2S>() == 2);

// --- Native Rust IQ2_S dequant (qk=256, block=82B: f16 scale + 64 bytes codes + 8 bytes qh + 8 bytes scales)
#[inline]
unsafe fn rust_dequant_row_iq2s(src: *const c_void, dst: *mut f32, n: usize) {
    const QK: usize = 256;
    const QMAP: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

    let mut in_ptr = src as *const u8;
    let out = unsafe { core::slice::from_raw_parts_mut(dst, n) };
    let mut produced = 0usize;

    while produced < n {
        let remain = n - produced;

        // f16 scale (2 bytes, little endian)
        let d_bits = unsafe { *(in_ptr as *const u16) };
        let d = f16::from_bits(u16::from_le(d_bits)).to_f32();
        in_ptr = unsafe { in_ptr.add(2) };

        // 64 bytes of packed 2-bit signed codes; 4 per byte
        let qs = unsafe { core::slice::from_raw_parts(in_ptr, 64) };
        in_ptr = unsafe { in_ptr.add(64) };

        // Skip unused qh and scales fields (8 bytes each)
        in_ptr = unsafe { in_ptr.add(16) };

        let take = QK.min(remain);
        let out_blk = &mut out[produced..produced + take];

        let mut o = 0usize;
        for &b in qs {
            if o >= take {
                break; // tail block
            }

            let q0 = (b & 0b11) as usize;
            if o < take {
                out_blk[o] = d * QMAP[q0];
                o += 1;
            }
            let q1 = ((b >> 2) & 0b11) as usize;
            if o < take {
                out_blk[o] = d * QMAP[q1];
                o += 1;
            }
            let q2 = ((b >> 4) & 0b11) as usize;
            if o < take {
                out_blk[o] = d * QMAP[q2];
                o += 1;
            }
            let q3 = ((b >> 6) & 0b11) as usize;
            if o < take {
                out_blk[o] = d * QMAP[q3];
                o += 1;
            }
        }

        produced += take;
        // Note: input pointer already advanced by full block (82B); nothing else for tail.
    }
}

/// Backend implementation for IQ2_S quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iq2sBackend {
    /// Pure Rust (default)
    Rust,
    /// Optional ggml FFI for parity/cross-validation
    Ffi,
}

impl Iq2sBackend {
    #[inline]
    fn checked_total_elements(dims: &[usize]) -> anyhow::Result<usize> {
        dims.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("IQ2_S element count overflow for dims {dims:?}"))
        })
    }

    /// Select the appropriate backend based on environment and features
    pub fn selected() -> Self {
        match std::env::var("BITNET_IQ2S_IMPL").ok().as_deref() {
            Some("ffi") => Iq2sBackend::Ffi,
            _ => Iq2sBackend::Rust,
        }
    }

    /// Check if the selected backend is available
    pub fn is_available(&self) -> bool {
        match self {
            Self::Rust => true, // Always available
            Self::Ffi => cfg!(feature = "iq2s-ffi"),
        }
    }

    /// Get the name of the backend
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Ffi => "ffi",
        }
    }

    /// Get the quantization constant QK for IQ2_S
    #[inline]
    pub fn qk(self) -> usize {
        match self {
            Iq2sBackend::Rust => 256,
            Iq2sBackend::Ffi => ffi_qk(),
        }
    }

    /// Get the block size in bytes for IQ2_S
    #[inline]
    pub fn block_bytes(self) -> usize {
        match self {
            Iq2sBackend::Rust => size_of::<BlockIq2S>(), // Match GGML's block_iq2_s layout
            Iq2sBackend::Ffi => ffi_block_bytes(),
        }
    }

    /// Dequantize a row of IQ2_S data
    ///
    /// # Safety
    ///
    /// - `src` must point to valid IQ2_S quantized data with at least `n` elements worth of data
    /// - `dst` must point to valid memory for at least `n` f32 elements
    /// - The caller must ensure both pointers remain valid for the duration of the call
    #[inline]
    pub unsafe fn dequantize_row(self, src: *const c_void, dst: *mut f32, n: usize) {
        match self {
            Iq2sBackend::Rust => unsafe { rust_dequant_row_iq2s(src, dst, n) },
            #[cfg(feature = "iq2s-ffi")]
            Iq2sBackend::Ffi => ffi_dequant_row(src, dst, n),
            #[cfg(not(feature = "iq2s-ffi"))]
            Iq2sBackend::Ffi => unreachable!("compiled without `iq2s-ffi`"),
        }
    }

    /// Dequantize IQ2_S data using the selected backend
    pub fn dequantize(&self, src_bytes: &[u8], dims: &[usize]) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Rust => {
                // Use the native Rust implementation
                let total_elements = Self::checked_total_elements(dims)?;
                let blocks = total_elements.div_ceil(self.qk());
                let required_bytes = blocks.checked_mul(self.block_bytes()).ok_or_else(|| {
                    anyhow::anyhow!("IQ2_S byte length overflow for dims {dims:?}")
                })?;

                anyhow::ensure!(
                    src_bytes.len() == required_bytes,
                    "IQ2_S byte length mismatch: got {} expected {} (dims={dims:?}, qk={}, block_bytes={})",
                    src_bytes.len(),
                    required_bytes,
                    self.qk(),
                    self.block_bytes()
                );

                let mut output = vec![0.0f32; total_elements];

                unsafe {
                    rust_dequant_row_iq2s(
                        src_bytes.as_ptr() as *const c_void,
                        output.as_mut_ptr(),
                        total_elements,
                    );
                }

                Ok(output)
            }
            Self::Ffi => {
                #[cfg(feature = "iq2s-ffi")]
                {
                    super::iq2s::dequantize_to_f32(src_bytes, dims)
                }
                #[cfg(not(feature = "iq2s-ffi"))]
                {
                    let _ = (src_bytes, dims); // Avoid unused param warnings
                    anyhow::bail!(
                        "IQ2_S FFI backend not available (compile with --features iq2s-ffi)"
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.get_or_init(|| Mutex::new(())).lock().expect("env guard poisoned")
    }

    #[test]
    fn iq2s_rust_dequant_basic() {
        let mut blk = [0u8; 82];
        let d = f16::from_f32(0.5).to_bits();
        blk[0..2].copy_from_slice(&u16::to_le_bytes(d));
        blk[2..66].fill(0b11_10_01_00);
        blk[66..74].fill(0xAA); // ensure qh ignored
        blk[74..82].fill(0x55); // ensure scales ignored
        let mut out = vec![0.0f32; 256];
        unsafe {
            Iq2sBackend::Rust.dequantize_row(blk.as_ptr() as *const c_void, out.as_mut_ptr(), 256);
        }
        let expect = [-1.0, -0.5, 0.5, 1.0];
        for i in 0..256 {
            assert!(
                (out[i] - expect[i % 4]).abs() < 1e-7,
                "mismatch at {i}: {} vs {}",
                out[i],
                expect[i % 4]
            );
        }
    }

    #[test]
    fn iq2s_rust_partial_tail() {
        let mut blk = [0u8; 82];
        let d = f16::from_f32(0.5).to_bits();
        blk[0..2].copy_from_slice(&u16::to_le_bytes(d));
        blk[2..66].fill(0b11_10_01_00);
        blk[66..74].fill(0xAA);
        blk[74..82].fill(0x55);
        let mut out = vec![0.0f32; 13];
        unsafe {
            Iq2sBackend::Rust.dequantize_row(blk.as_ptr() as *const c_void, out.as_mut_ptr(), 13);
        }
        let expect = [-1.0, -0.5, 0.5, 1.0, -1.0, -0.5, 0.5, 1.0, -1.0, -0.5, 0.5, 1.0, -1.0];
        for i in 0..13 {
            assert!((out[i] - expect[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn test_backend_selection() {
        let _g = env_guard();
        unsafe {
            std::env::remove_var("BITNET_IQ2S_IMPL");
        }

        let backend = Iq2sBackend::selected();
        assert_eq!(backend, Iq2sBackend::Rust); // Now defaults to Rust
    }

    #[test]
    fn test_backend_env_override() {
        let _g = env_guard();
        unsafe {
            std::env::set_var("BITNET_IQ2S_IMPL", "ffi");
        }
        let backend = Iq2sBackend::selected();
        assert_eq!(backend, Iq2sBackend::Ffi);

        unsafe {
            std::env::set_var("BITNET_IQ2S_IMPL", "rust");
        }
        let backend = Iq2sBackend::selected();
        assert_eq!(backend, Iq2sBackend::Rust);

        unsafe {
            std::env::remove_var("BITNET_IQ2S_IMPL");
        }
    }

    #[test]
    fn iq2s_rust_partial_blocks() {
        // Test that Rust backend can handle partial blocks
        let mut src = [0u8; 82 * 3]; // 3 blocks
        // Fill with a known pattern
        for blk in src.chunks_mut(82) {
            // Set scale to 0.5
            let d = f16::from_f32(0.5).to_bits();
            blk[0..2].copy_from_slice(&u16::to_le_bytes(d));
            // Set quantized values to pattern 0b11_10_01_00
            for slot in &mut blk[2..66] {
                *slot = 0b11_10_01_00; // Maps to [-1.0, -0.5, 0.5, 1.0] after scaling
            }
            blk[66..74].fill(0xAA);
            blk[74..82].fill(0x55);
        }

        // Test partial block handling
        let n = 3 * 256 - 17; // Include a partial tail
        let mut out = vec![0.0f32; n];
        unsafe {
            Iq2sBackend::Rust.dequantize_row(src.as_ptr() as *const c_void, out.as_mut_ptr(), n);
        }

        // Check expected pattern for first few elements
        let expected = [-1.0, -0.5, 0.5, 1.0]; // Scaled by 0.5
        for i in 0..std::cmp::min(n, 8) {
            assert!(
                (out[i] - expected[i % 4]).abs() < 1e-6,
                "mismatch at {i}: expected {}, got {}",
                expected[i % 4],
                out[i]
            );
        }
    }

    #[test]
    fn iq2s_rust_dequant_rejects_invalid_byte_length() {
        let dims = [257usize];
        let src = vec![0u8; 82]; // Needs 2 blocks for 257 elems

        let err = Iq2sBackend::Rust
            .dequantize(&src, &dims)
            .expect_err("expected length validation to fail");

        assert!(err.to_string().contains("byte length mismatch"), "unexpected error: {err}");
    }

    #[cfg(all(test, feature = "iq2s-ffi"))]
    #[test]
    fn iq2s_rust_matches_ffi() {
        // Two identical blocks with a deterministic pattern.
        let mut src = [BlockIq2S::default(); 2];
        for blk in &mut src {
            blk.d = f16::from_f32(0.5).to_bits();
            blk.qs.fill(0b11_10_01_00);
            blk.qh.fill(0xAA); // ensure unused fields don't affect output
            blk.scales.fill(0x55);
        }

        let n = 2 * 256; // Number of dequantized elements
        let mut rust_out = vec![0.0f32; n];
        let mut ffi_out = vec![0.0f32; n];
        unsafe {
            let ptr = src.as_ptr() as *const c_void;
            Iq2sBackend::Rust.dequantize_row(ptr, rust_out.as_mut_ptr(), n);
            Iq2sBackend::Ffi.dequantize_row(ptr, ffi_out.as_mut_ptr(), n);
        }

        // The FFI and Rust implementations should match element-for-element.
        for i in 0..n {
            assert!(
                (rust_out[i] - ffi_out[i]).abs() < 1e-6,
                "mismatch at {i}: rust {} ffi {}",
                rust_out[i],
                ffi_out[i]
            );
        }
    }
}
