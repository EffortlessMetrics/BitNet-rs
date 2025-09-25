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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal,
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

// Conservative conversion from Candle device - avoids assuming CUDA ordinal
impl From<&candle_core::Device> for Device {
    fn from(d: &candle_core::Device) -> Self {
        match d {
            candle_core::Device::Cpu => Device::Cpu,
            #[cfg(feature = "cuda")]
            candle_core::Device::Cuda(_) => Device::Cuda(usize::MAX), // unknown ordinal
            #[cfg(feature = "metal")]
            candle_core::Device::Metal(_) => Device::Metal,
            #[allow(unreachable_patterns)]
            _ => Device::Cpu,
        }
    }
}

impl Device {
    /// Convert our device preference to Candle's device.
    #[cfg(feature = "cuda")]
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
        })
    }

    #[cfg(not(feature = "cuda"))]
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

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub context_length: usize,
    pub quantization: Option<QuantizationType>,
}

/// Performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub tokens_per_second: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization: Option<f64>,
}
