//! Backend abstraction for IQ2_S quantization to support both FFI and pure-Rust implementations

use core::ffi::c_void;
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

// --- Native Rust IQ2_S dequant (qk=256, block=66B: f16 scale + 64 bytes codes)
// Note: The actual GGML block_iq2_s is 82B with additional unused fields,
// but the FFI reports 66B, so we match that for compatibility.
#[inline]
unsafe fn rust_dequant_row_iq2s(src: *const c_void, dst: *mut f32, n: usize) {
    const QK: usize = 256;

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

        let take = QK.min(remain);
        let out_blk = &mut out[produced..produced + take];

        let mut o = 0usize;
        for &b in qs {
            if o >= take {
                break;
            } // tail block
            // Bits: (1:0),(3:2),(5:4),(7:6) mapped to -2,-1,0,1
            let c0 = (b & 0b11) as i8 - 2;
            let c1 = ((b >> 2) & 0b11) as i8 - 2;
            let c2 = ((b >> 4) & 0b11) as i8 - 2;
            let c3 = ((b >> 6) & 0b11) as i8 - 2;

            if o < take {
                out_blk[o] = d * (c0 as f32);
                o += 1;
            }
            if o < take {
                out_blk[o] = d * (c1 as f32);
                o += 1;
            }
            if o < take {
                out_blk[o] = d * (c2 as f32);
                o += 1;
            }
            if o < take {
                out_blk[o] = d * (c3 as f32);
                o += 1;
            }
        }

        produced += take;
        // Note: input pointer already advanced by full block (66B); nothing else for tail.
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
            Iq2sBackend::Rust => 66, // Match FFI reported size
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
                let total_elements: usize = dims.iter().product();
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
        let mut blk = [0u8; 66];
        let d = f16::from_f32(0.5).to_bits();
        blk[0..2].copy_from_slice(&u16::to_le_bytes(d));
        blk[2..].fill(0b11_10_01_00);
        let mut out = vec![0.0f32; 256];
        unsafe {
            Iq2sBackend::Rust.dequantize_row(blk.as_ptr() as *const c_void, out.as_mut_ptr(), 256);
        }
        let expect = [-1.0, -0.5, 0.0, 0.5];
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
        let mut blk = [0u8; 66];
        let d = f16::from_f32(0.5).to_bits();
        blk[0..2].copy_from_slice(&u16::to_le_bytes(d));
        blk[2..].fill(0b11_10_01_00);
        let mut out = vec![0.0f32; 13];
        unsafe {
            Iq2sBackend::Rust.dequantize_row(blk.as_ptr() as *const c_void, out.as_mut_ptr(), 13);
        }
        let expect = [-1.0, -0.5, 0.0, 0.5, -1.0, -0.5, 0.0, 0.5, -1.0, -0.5, 0.0, 0.5, -1.0];
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

    #[cfg(all(test, feature = "iq2s-ffi"))]
    #[test]
    fn iq2s_rust_matches_ffi() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut src = vec![0u8; 66 * 3]; // 3 blocks
        for b in &mut src {
            *b = rng.sample(rand::distributions::Standard);
        }
        // Force a valid fp16 in each block's first two bytes
        for blk in src.chunks_mut(66) {
            let d = f16::from_f32(0.25).to_bits();
            blk[0..2].copy_from_slice(&u16::to_le_bytes(d));
        }
        let n = 3 * 256 - 17; // Include a partial tail
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        unsafe {
            Iq2sBackend::Rust.dequantize_row(src.as_ptr() as *const c_void, a.as_mut_ptr(), n);
            Iq2sBackend::Ffi.dequantize_row(src.as_ptr() as *const c_void, b.as_mut_ptr(), n);
        }
        for i in 0..n {
            assert!((a[i] - b[i]).abs() < 1e-6, "mismatch at {i}: {} vs {}", a[i], b[i]);
        }
    }
}
