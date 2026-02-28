//! OpenCL mixed-precision support for Intel Arc GPUs.
//!
//! Selects optimal precision mode based on device capabilities.
//! Controlled via `BITNET_PRECISION` env var: fp16|int8|fp32|auto.

use std::fmt;

/// Precision mode for GPU compute kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenClPrecisionMode {
    /// Full FP32 precision (default, most compatible)
    FP32,
    /// Half precision FP16 (requires cl_khr_fp16)
    FP16,
    /// INT8 quantized (requires integer dot product support)
    INT8,
    /// Mixed: FP16 activations + ternary weights -> FP32 accumulate
    Mixed,
    /// Auto-detect best precision from device capabilities
    Auto,
}

impl fmt::Display for OpenClPrecisionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FP32 => write!(f, "fp32"),
            Self::FP16 => write!(f, "fp16"),
            Self::INT8 => write!(f, "int8"),
            Self::Mixed => write!(f, "mixed"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

impl OpenClPrecisionMode {
    /// Parse from string (env var value).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fp32" => Some(Self::FP32),
            "fp16" => Some(Self::FP16),
            "int8" => Some(Self::INT8),
            "mixed" => Some(Self::Mixed),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }

    /// Read from `BITNET_PRECISION` env var, defaulting to Auto.
    pub fn from_env() -> Self {
        std::env::var("BITNET_PRECISION")
            .ok()
            .and_then(|v| Self::from_str_loose(&v))
            .unwrap_or(Self::Auto)
    }
}

/// Device capabilities relevant to precision selection.
#[derive(Debug, Clone)]
pub struct OpenClDeviceCaps {
    /// Device supports cl_khr_fp16
    pub has_fp16: bool,
    /// Device supports cl_khr_int8
    pub has_int8: bool,
    /// Device name for logging
    pub device_name: String,
}

/// Select the best precision mode for the given device capabilities.
pub fn select_opencl_precision(
    caps: &OpenClDeviceCaps,
    requested: OpenClPrecisionMode,
) -> OpenClPrecisionMode {
    match requested {
        OpenClPrecisionMode::Auto => {
            if caps.has_fp16 {
                OpenClPrecisionMode::Mixed
            } else {
                OpenClPrecisionMode::FP32
            }
        }
        OpenClPrecisionMode::FP16 if !caps.has_fp16 => {
            log::warn!(
                "FP16 requested but cl_khr_fp16 not supported on {}; falling back to FP32",
                caps.device_name
            );
            OpenClPrecisionMode::FP32
        }
        OpenClPrecisionMode::INT8 if !caps.has_int8 => {
            log::warn!(
                "INT8 requested but not supported on {}; falling back to FP32",
                caps.device_name
            );
            OpenClPrecisionMode::FP32
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn arc_caps() -> OpenClDeviceCaps {
        OpenClDeviceCaps {
            has_fp16: true,
            has_int8: true,
            device_name: "Intel Arc A770".into(),
        }
    }

    fn basic_caps() -> OpenClDeviceCaps {
        OpenClDeviceCaps {
            has_fp16: false,
            has_int8: false,
            device_name: "Basic GPU".into(),
        }
    }

    #[test]
    fn test_auto_selects_mixed_when_fp16_available() {
        let mode = select_opencl_precision(&arc_caps(), OpenClPrecisionMode::Auto);
        assert_eq!(mode, OpenClPrecisionMode::Mixed);
    }

    #[test]
    fn test_auto_selects_fp32_when_no_fp16() {
        let mode = select_opencl_precision(&basic_caps(), OpenClPrecisionMode::Auto);
        assert_eq!(mode, OpenClPrecisionMode::FP32);
    }

    #[test]
    fn test_fp16_fallback_when_unsupported() {
        let mode = select_opencl_precision(&basic_caps(), OpenClPrecisionMode::FP16);
        assert_eq!(mode, OpenClPrecisionMode::FP32);
    }

    #[test]
    fn test_int8_fallback_when_unsupported() {
        let mode = select_opencl_precision(&basic_caps(), OpenClPrecisionMode::INT8);
        assert_eq!(mode, OpenClPrecisionMode::FP32);
    }

    #[test]
    fn test_explicit_fp32_always_works() {
        let mode = select_opencl_precision(&basic_caps(), OpenClPrecisionMode::FP32);
        assert_eq!(mode, OpenClPrecisionMode::FP32);
    }

    #[test]
    fn test_explicit_fp16_when_supported() {
        let mode = select_opencl_precision(&arc_caps(), OpenClPrecisionMode::FP16);
        assert_eq!(mode, OpenClPrecisionMode::FP16);
    }

    #[test]
    fn test_explicit_int8_when_supported() {
        let mode = select_opencl_precision(&arc_caps(), OpenClPrecisionMode::INT8);
        assert_eq!(mode, OpenClPrecisionMode::INT8);
    }

    #[test]
    fn test_parse_precision_mode() {
        assert_eq!(OpenClPrecisionMode::from_str_loose("fp16"), Some(OpenClPrecisionMode::FP16));
        assert_eq!(OpenClPrecisionMode::from_str_loose("FP32"), Some(OpenClPrecisionMode::FP32));
        assert_eq!(OpenClPrecisionMode::from_str_loose("INT8"), Some(OpenClPrecisionMode::INT8));
        assert_eq!(OpenClPrecisionMode::from_str_loose("auto"), Some(OpenClPrecisionMode::Auto));
        assert_eq!(OpenClPrecisionMode::from_str_loose("invalid"), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", OpenClPrecisionMode::FP16), "fp16");
        assert_eq!(format!("{}", OpenClPrecisionMode::Mixed), "mixed");
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_from_env_default_is_auto() {
        temp_env::with_var("BITNET_PRECISION", None::<&str>, || {
            assert_eq!(OpenClPrecisionMode::from_env(), OpenClPrecisionMode::Auto);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_from_env_fp16() {
        temp_env::with_var("BITNET_PRECISION", Some("fp16"), || {
            assert_eq!(OpenClPrecisionMode::from_env(), OpenClPrecisionMode::FP16);
        });
    }
}
