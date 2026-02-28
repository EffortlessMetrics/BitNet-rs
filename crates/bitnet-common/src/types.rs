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
    Hip(usize),
    Npu,
    Metal,
}

// Conservative conversion from Candle device - avoids assuming CUDA ordinal
impl From<&candle_core::Device> for Device {
    fn from(d: &candle_core::Device) -> Self {
        match d {
            candle_core::Device::Cpu => Device::Cpu,
            #[cfg(any(feature = "gpu", feature = "cuda", feature = "hip"))]
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

    /// Check if this device is HIP (ROCm)
    pub fn is_hip(&self) -> bool {
        matches!(self, Device::Hip(_))
    }

    /// Check if this device is NPU
    pub fn is_npu(&self) -> bool {
        matches!(self, Device::Npu)
    }

    /// Convert our device preference to Candle's device.
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "hip"))]
    pub fn to_candle(&self) -> anyhow::Result<candle_core::Device> {
        Ok(match *self {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(i) if i != usize::MAX => candle_core::Device::new_cuda(i)?,
            Device::Cuda(_) => candle_core::Device::Cpu, // fallback for unknown
            // HIP/NPU candle backends are not wired yet; fallback keeps API non-breaking.
            Device::Hip(_) | Device::Npu => candle_core::Device::Cpu,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal => {
                use candle_core::backend::BackendDevice;
                candle_core::Device::Metal(candle_core::MetalDevice::new()?)
            }
            #[cfg(any(not(feature = "metal"), not(target_os = "macos")))]
            Device::Metal => candle_core::Device::Cpu,
        })
    }

    #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "hip")))]
    pub fn to_candle(&self) -> anyhow::Result<candle_core::Device> {
        Ok(match *self {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Hip(_) | Device::Npu => candle_core::Device::Cpu,
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
