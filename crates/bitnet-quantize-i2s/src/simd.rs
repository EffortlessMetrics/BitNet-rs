//! SIMD-accelerated helpers for 2-bit pack/unpack.
//!
//! Compile-time dispatch:
//! - `x86_64` with `avx2` target feature: AVX2 path
//! - `aarch64` with `neon` target feature: NEON path
//! - Otherwise: scalar fallback

// ---------------------------------------------------------------------------
// AVX2 implementation (x86_64)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    /// Pack 32 ternary i8 values (-1, 0, +1 mapped to codes 0, 1, 2) into 8
    /// bytes using AVX2. Caller must ensure AVX2 is available.
    ///
    /// # Safety
    ///
    /// Requires AVX2 support at runtime.
    #[target_feature(enable = "avx2")]
    pub unsafe fn pack_32_avx2(values: &[i8; 32]) -> [u8; 8] {
        unsafe {
            // Load 32 i8 values into a 256-bit register
            let v = _mm256_loadu_si256(values.as_ptr().cast());
            // Add 1 to shift {-1,0,1} to {0,1,2}
            let ones = _mm256_set1_epi8(1);
            let codes = _mm256_add_epi8(v, ones);
            // Extract to scalar and pack — AVX2 lacks a direct 2-bit pack.
            let mut tmp = [0u8; 32];
            _mm256_storeu_si256(tmp.as_mut_ptr().cast(), codes);
            let mut out = [0u8; 8];
            for (i, byte) in out.iter_mut().enumerate() {
                let base = i * 4;
                *byte = (tmp[base] & 0x3)
                    | ((tmp[base + 1] & 0x3) << 2)
                    | ((tmp[base + 2] & 0x3) << 4)
                    | ((tmp[base + 3] & 0x3) << 6);
            }
            out
        }
    }

    /// Unpack 8 packed bytes into 32 i8 ternary values using AVX2.
    ///
    /// # Safety
    ///
    /// Requires AVX2 support at runtime.
    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_32_avx2(packed: &[u8; 8]) -> [i8; 32] {
        unsafe {
            let mut expanded = [0u8; 32];
            for (i, &b) in packed.iter().enumerate() {
                expanded[i * 4] = b & 0x3;
                expanded[i * 4 + 1] = (b >> 2) & 0x3;
                expanded[i * 4 + 2] = (b >> 4) & 0x3;
                expanded[i * 4 + 3] = (b >> 6) & 0x3;
            }
            let codes = _mm256_loadu_si256(expanded.as_ptr().cast());
            let ones = _mm256_set1_epi8(1);
            let result = _mm256_sub_epi8(codes, ones);
            let mut out = [0i8; 32];
            _mm256_storeu_si256(out.as_mut_ptr().cast(), result);
            out
        }
    }
}

// ---------------------------------------------------------------------------
// NEON implementation (aarch64)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
pub mod neon {
    use std::arch::aarch64::*;

    /// Pack 16 ternary i8 values into 4 bytes using NEON.
    ///
    /// # Safety
    ///
    /// Requires NEON support (always present on `aarch64`).
    pub unsafe fn pack_16_neon(values: &[i8; 16]) -> [u8; 4] {
        unsafe {
            let v = vld1q_s8(values.as_ptr());
            let ones = vdupq_n_s8(1);
            let codes = vaddq_s8(v, ones);
            let mut tmp = [0u8; 16];
            vst1q_u8(tmp.as_mut_ptr(), vreinterpretq_u8_s8(codes));
            let mut out = [0u8; 4];
            for (i, byte) in out.iter_mut().enumerate() {
                let base = i * 4;
                *byte = (tmp[base] & 0x3)
                    | ((tmp[base + 1] & 0x3) << 2)
                    | ((tmp[base + 2] & 0x3) << 4)
                    | ((tmp[base + 3] & 0x3) << 6);
            }
            out
        }
    }

    /// Unpack 4 packed bytes into 16 i8 ternary values using NEON.
    ///
    /// # Safety
    ///
    /// Requires NEON support.
    pub unsafe fn unpack_16_neon(packed: &[u8; 4]) -> [i8; 16] {
        unsafe {
            let mut expanded = [0u8; 16];
            for (i, &b) in packed.iter().enumerate() {
                expanded[i * 4] = b & 0x3;
                expanded[i * 4 + 1] = (b >> 2) & 0x3;
                expanded[i * 4 + 2] = (b >> 4) & 0x3;
                expanded[i * 4 + 3] = (b >> 6) & 0x3;
            }
            let codes = vld1q_u8(expanded.as_ptr());
            let ones = vdupq_n_u8(1);
            let result = vsubq_u8(codes, ones);
            let mut out = [0i8; 16];
            vst1q_s8(out.as_mut_ptr(), vreinterpretq_s8_u8(result));
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatch helpers
// ---------------------------------------------------------------------------

/// Returns `true` if AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
#[must_use]
pub fn has_avx2() -> bool {
    false
}

/// Returns `true` if NEON is available at runtime (always true on `aarch64`).
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn has_neon() -> bool {
    true
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(dead_code)]
#[must_use]
pub const fn has_neon() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_detect_does_not_panic() {
        let _ = has_avx2();
        let _ = has_neon();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_pack_unpack_roundtrip() {
        if !has_avx2() {
            return;
        }
        let values: [i8; 32] = [
            -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0,
            1, -1, 0, 1, -1, 0,
        ];
        unsafe {
            let packed = avx2::pack_32_avx2(&values);
            let unpacked = avx2::unpack_32_avx2(&packed);
            assert_eq!(unpacked, values);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_all_minus_one() {
        if !has_avx2() {
            return;
        }
        let values = [-1i8; 32];
        unsafe {
            let packed = avx2::pack_32_avx2(&values);
            // code 0 for all -> all bytes should be 0
            assert!(packed.iter().all(|&b| b == 0));
            let unpacked = avx2::unpack_32_avx2(&packed);
            assert_eq!(unpacked, values);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_all_zero() {
        if !has_avx2() {
            return;
        }
        let values = [0i8; 32];
        unsafe {
            let packed = avx2::pack_32_avx2(&values);
            // code 1 for all -> each byte = 0b01_01_01_01 = 0x55
            assert!(packed.iter().all(|&b| b == 0x55));
            let unpacked = avx2::unpack_32_avx2(&packed);
            assert_eq!(unpacked, values);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_all_one() {
        if !has_avx2() {
            return;
        }
        let values = [1i8; 32];
        unsafe {
            let packed = avx2::pack_32_avx2(&values);
            // code 2 for all -> each byte = 0b10_10_10_10 = 0xAA
            assert!(packed.iter().all(|&b| b == 0xAA));
            let unpacked = avx2::unpack_32_avx2(&packed);
            assert_eq!(unpacked, values);
        }
    }
}
