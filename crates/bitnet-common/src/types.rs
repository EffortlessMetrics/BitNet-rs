//! Common type definitions

use serde::{Deserialize, Serialize};

/// Quantization types supported by BitNet
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 2-bit signed quantization
    I2S,
    /// Table lookup 1 (ARM optimized)
    TL1,
    /// Table lookup 2 (x86 optimized)
    TL2,
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationType::I2S => write!(f, "I2_S"),
            QuantizationType::TL1 => write!(f, "TL1"),
            QuantizationType::TL2 => write!(f, "TL2"),
        }
    }
}

/// Device abstraction for computation
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(usize),
    Metal,
    OpenCL(usize),
    Vulkan(usize),
}

// Conservative conversion from Candle device - avoids assuming CUDA ordinal
impl From<&candle_core::Device> for Device {
    fn from(d: &candle_core::Device) -> Self {
        match d {
            candle_core::Device::Cpu => Device::Cpu,
            #[cfg(any(feature = "gpu", feature = "cuda"))]
            candle_core::Device::Cuda(_) => Device::Cuda(usize::MAX), // unknown ordinal
            #[cfg(all(feature = "metal", target_os = "macos"))]
            candle_core::Device::Metal(_) => Device::Metal,
            #[allow(unreachable_patterns)]
            _ => Device::Cpu,
        }
    }
}

impl Device {
    /// Create a new CUDA device with the specified index
    pub fn new_cuda(index: usize) -> anyhow::Result<Self> {
        Ok(Device::Cuda(index))
    }

    /// Check if this device is CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Check if this device is CUDA
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Create a new OpenCL device with the specified index
    pub fn new_opencl(index: usize) -> anyhow::Result<Self> {
        Ok(Device::OpenCL(index))
    }

    /// Check if this device is OpenCL
    pub fn is_opencl(&self) -> bool {
        matches!(self, Device::OpenCL(_))
    }

    /// Create a new Vulkan device with the specified index
    pub fn new_vulkan(index: usize) -> anyhow::Result<Self> {
        Ok(Device::Vulkan(index))
    }

    /// Check if this device is Vulkan
    pub fn is_vulkan(&self) -> bool {
        matches!(self, Device::Vulkan(_))
    }

    /// Convert our device preference to Candle's device.
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    pub fn to_candle(&self) -> anyhow::Result<candle_core::Device> {
        Ok(match *self {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(i) if i != usize::MAX => candle_core::Device::new_cuda(i)?,
            Device::Cuda(_) => candle_core::Device::Cpu, // fallback for unknown
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal => {
                use candle_core::backend::BackendDevice;
                candle_core::Device::Metal(candle_core::MetalDevice::new()?)
            }
            #[cfg(any(not(feature = "metal"), not(target_os = "macos")))]
            Device::Metal => candle_core::Device::Cpu,
            Device::OpenCL(_) => candle_core::Device::Cpu, // OpenCL uses its own buffer management
            Device::Vulkan(_) => candle_core::Device::Cpu, // Vulkan uses its own buffer management
        })
    }

    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    pub fn to_candle(&self) -> anyhow::Result<candle_core::Device> {
        Ok(match *self {
            Device::Cpu => candle_core::Device::Cpu,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal => {
                use candle_core::backend::BackendDevice;
                candle_core::Device::Metal(candle_core::MetalDevice::new()?)
            }
            #[cfg(any(not(feature = "metal"), not(target_os = "macos")))]
            _ => candle_core::Device::Cpu,
        })
    }
}

/// Generation configuration for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            do_sample: true,
            seed: None,
        }
    }
}

/// Model correction record (LayerNorm rescaling, I2_S dequant, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionRecord {
    /// Layer/tensor name (e.g., "model.layers.0.input_layernorm.weight")
    pub layer: String,
    /// Correction type (e.g., "ln_gamma_rescale_rms", "i2s_dequant_inversion")
    pub correction_type: String,
    /// RMS value before correction (for LayerNorm corrections)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_before: Option<f32>,
    /// RMS value after correction (for LayerNorm corrections)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_after: Option<f32>,
    /// Scaling factor applied (for LayerNorm corrections)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub factor: Option<f32>,
    /// Policy fingerprint (e.g., "BITNET_FIX_LN_SCALE=1", "policy.yml:sha256:abc123")
    pub policy_fingerprint: String,
    /// Additional correction metadata (JSON-serializable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub context_length: usize,
    pub quantization: Option<QuantizationType>,
    /// SHA256 fingerprint of the model file (format: "sha256-<hex>")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    /// Applied corrections (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corrections_applied: Option<Vec<CorrectionRecord>>,
}

/// Performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub tokens_per_second: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_opencl_variant_basic() {
        let dev = Device::OpenCL(0);
        assert!(dev.is_opencl());
        assert!(!dev.is_cpu());
        assert!(!dev.is_cuda());
    }

    #[test]
    fn device_opencl_new() {
        let dev = Device::new_opencl(0).unwrap();
        assert_eq!(dev, Device::OpenCL(0));
    }

    #[test]
    fn device_opencl_to_candle_falls_back_to_cpu() {
        let dev = Device::OpenCL(0);
        let candle_dev = dev.to_candle().unwrap();
        assert!(matches!(candle_dev, candle_core::Device::Cpu));
    }

    #[test]
    fn device_opencl_debug_format() {
        let dev = Device::OpenCL(2);
        let debug = format!("{:?}", dev);
        assert!(debug.contains("OpenCL"));
        assert!(debug.contains("2"));
    }

    #[test]
    fn device_opencl_serialization_roundtrip() {
        let dev = Device::OpenCL(1);
        let json = serde_json::to_string(&dev).unwrap();
        let deserialized: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, deserialized);
    }

    #[test]
    fn device_vulkan_variant_basic() {
        let dev = Device::Vulkan(0);
        assert!(dev.is_vulkan());
        assert!(!dev.is_cpu());
        assert!(!dev.is_cuda());
        assert!(!dev.is_opencl());
    }

    #[test]
    fn device_vulkan_new() {
        let dev = Device::new_vulkan(0).unwrap();
        assert_eq!(dev, Device::Vulkan(0));
    }

    #[test]
    fn device_vulkan_to_candle_falls_back_to_cpu() {
        let dev = Device::Vulkan(0);
        let candle_dev = dev.to_candle().unwrap();
        assert!(matches!(candle_dev, candle_core::Device::Cpu));
    }

    #[test]
    fn device_vulkan_debug_format() {
        let dev = Device::Vulkan(2);
        let debug = format!("{:?}", dev);
        assert!(debug.contains("Vulkan"));
        assert!(debug.contains("2"));
    }

    #[test]
    fn device_vulkan_serialization_roundtrip() {
        let dev = Device::Vulkan(1);
        let json = serde_json::to_string(&dev).unwrap();
        let deserialized: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, deserialized);
    }
}
