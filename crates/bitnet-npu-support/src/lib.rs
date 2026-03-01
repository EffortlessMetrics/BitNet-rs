//! SRP microcrate for NPU/HIP backend support policy.
//!
//! Centralizes behavior used by model loaders and device routing while
//! dedicated HIP/NPU backends are still being integrated.

use bitnet_common::Device;

/// Shared validation error text for model loading on unsupported accelerators.
pub const MODEL_LOADING_UNSUPPORTED_MSG: &str =
    "HIP/NPU devices are not yet supported for model loading";

/// Returns true when the provided device is currently unsupported for direct
/// model loading and should be routed through CPU fallback behavior.
#[must_use]
pub fn requires_cpu_fallback(device: &Device) -> bool {
    matches!(device, Device::Hip(_) | Device::Npu)
}

/// Returns the canonical model-loading validation message when loading is
/// requested on unsupported accelerators.
#[must_use]
pub fn model_loading_unsupported_message(device: &Device) -> Option<&'static str> {
    if requires_cpu_fallback(device) { Some(MODEL_LOADING_UNSUPPORTED_MSG) } else { None }
}

/// Returns a warning message for CPU fallback flows.
#[must_use]
pub fn fallback_warning_message(device: &Device) -> Option<&'static str> {
    if requires_cpu_fallback(device) {
        Some("HIP/NPU device requested but not supported, falling back to CPU")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_unsupported_accelerators() {
        assert!(requires_cpu_fallback(&Device::Hip(0)));
        assert!(requires_cpu_fallback(&Device::Npu));
        assert!(!requires_cpu_fallback(&Device::Cpu));
        assert!(!requires_cpu_fallback(&Device::Cuda(0)));
        assert!(!requires_cpu_fallback(&Device::OpenCL(0)));
    }

    #[test]
    fn returns_expected_model_loading_message() {
        assert_eq!(
            model_loading_unsupported_message(&Device::Npu),
            Some(MODEL_LOADING_UNSUPPORTED_MSG)
        );
        assert_eq!(model_loading_unsupported_message(&Device::Cpu), None);
    }

    #[test]
    fn returns_expected_fallback_warning() {
        assert_eq!(
            fallback_warning_message(&Device::Hip(1)),
            Some("HIP/NPU device requested but not supported, falling back to CPU")
        );
        assert_eq!(fallback_warning_message(&Device::Metal), None);
    }
}
