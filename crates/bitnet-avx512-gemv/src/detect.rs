//! Runtime detection of AVX-512 support.

/// Returns `true` when the current CPU supports AVX-512F at runtime.
///
/// On non-x86 targets this always returns `false`.
///
/// # Example
///
/// ```
/// if bitnet_avx512_gemv::detect::avx512_available() {
///     println!("AVX-512 fast path available");
/// }
/// ```
#[must_use]
pub fn avx512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx512_detection_returns_bool() {
        // Just ensure the function doesn't panic.
        let _supported = avx512_available();
    }
}
