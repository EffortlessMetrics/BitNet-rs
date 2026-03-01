//! NPU integration utilities for inference.
//!
//! This module keeps the historical `bitnet_inference::npu` import path while
//! re-exporting Qualcomm-focused policy helpers from the dedicated SRP crate.

pub use bitnet_qualcomm::{BITNET_ENABLE_NPU, map_device_token, npu_requested};
