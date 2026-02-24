//! Compile-time library capabilities detected via symbol analysis in build.rs.
//!
//! The `build.rs` script uses `nm`/`objdump` to inspect found C++ libraries
//! and emits `cargo:rustc-cfg` flags. This module reads those flags.
//!
//! # Cfg flags
//!
//! - `bitnet_cpp_available` — at least one C++ library was found at build time
//! - `bitnet_cpp_has_cuda` — a CUDA-capable library was found (contains cuda/cublas symbols)
//! - `bitnet_cpp_has_bitnet_shim` — the BitNet C shim surface was found (bitnet_eval etc.)
//!
//! # MSRV
//!
//! These cfg flags are emitted by build.rs; reading them has no MSRV requirement.

/// Capabilities detected from C++ libraries at build time via symbol analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileTimeLibCapabilities {
    /// A C++ library was found in `BITNET_CPP_DIR` at build time.
    pub available: bool,
    /// The found library contains CUDA symbols (cuda/cublas).
    /// False if no library was found OR if no CUDA symbols were detected.
    pub has_cuda: bool,
    /// The found library contains the BitNet C shim surface (bitnet_eval, bitnet_init, etc.).
    pub has_bitnet_shim: bool,
}

impl CompileTimeLibCapabilities {
    /// Read capabilities from compile-time cfg flags emitted by build.rs.
    ///
    /// This is a zero-cost function: all branches are resolved at compile time.
    pub fn from_compile_time() -> Self {
        Self {
            available: cfg!(bitnet_cpp_available),
            has_cuda: cfg!(bitnet_cpp_has_cuda),
            has_bitnet_shim: cfg!(bitnet_cpp_has_bitnet_shim),
        }
    }

    /// Returns a human-readable summary suitable for logs and receipts.
    ///
    /// Example: `"cpp=available cuda=yes shim=yes"`
    pub fn summary(&self) -> String {
        format!(
            "cpp={} cuda={} shim={}",
            if self.available { "available" } else { "unavailable" },
            if self.has_cuda { "yes" } else { "no" },
            if self.has_bitnet_shim { "yes" } else { "no" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_compile_time_returns_consistent_caps() {
        let caps = CompileTimeLibCapabilities::from_compile_time();
        // Structural invariant: has_cuda and has_bitnet_shim imply available.
        if caps.has_cuda {
            assert!(caps.available, "has_cuda implies available");
        }
        if caps.has_bitnet_shim {
            assert!(caps.available, "has_bitnet_shim implies available");
        }
    }

    #[test]
    fn summary_format_is_stable() {
        let caps =
            CompileTimeLibCapabilities { available: true, has_cuda: false, has_bitnet_shim: true };
        let s = caps.summary();
        assert!(s.contains("cpp=available"), "got: {s}");
        assert!(s.contains("cuda=no"), "got: {s}");
        assert!(s.contains("shim=yes"), "got: {s}");
    }

    #[test]
    fn summary_unavailable_case() {
        let caps = CompileTimeLibCapabilities {
            available: false,
            has_cuda: false,
            has_bitnet_shim: false,
        };
        let s = caps.summary();
        assert!(s.contains("cpp=unavailable"), "got: {s}");
    }
}
