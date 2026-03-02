//! Model conversion pipeline.
//!
//! Orchestrates conversion between model formats (SafeTensors, GGUF, etc).

use std::collections::HashMap;

/// Source format for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormat {
    SafeTensors,
    Gguf,
    Onnx,
    PyTorch,
}

impl SourceFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Onnx => "onnx",
            Self::PyTorch => "pytorch",
        }
    }

    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "safetensors" => Some(Self::SafeTensors),
            "gguf" => Some(Self::Gguf),
            "onnx" => Some(Self::Onnx),
            "pt" | "bin" | "pth" => Some(Self::PyTorch),
            _ => None,
        }
    }
}

/// Target format for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetFormat {
    GgufF16,
    GgufF32,
    GgufQ4,
    GgufQ8,
    GgufI2S,
}

impl TargetFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GgufF16 => "gguf-f16",
            Self::GgufF32 => "gguf-f32",
            Self::GgufQ4 => "gguf-q4",
            Self::GgufQ8 => "gguf-q8",
            Self::GgufI2S => "gguf-i2s",
        }
    }

    pub fn quantized(&self) -> bool {
        matches!(self, Self::GgufQ4 | Self::GgufQ8 | Self::GgufI2S)
    }
}

/// Conversion step.
#[derive(Debug, Clone)]
pub struct ConversionStep {
    pub name: String,
    pub description: String,
    pub estimated_seconds: u64,
}

/// Conversion plan.
#[derive(Debug, Clone)]
pub struct ConversionPlan {
    pub source: SourceFormat,
    pub target: TargetFormat,
    pub steps: Vec<ConversionStep>,
    pub estimated_total_seconds: u64,
    pub requires_quantization: bool,
    pub memory_estimate_bytes: u64,
}

impl ConversionPlan {
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

/// Conversion progress.
#[derive(Debug, Clone)]
pub struct ConversionProgress {
    pub current_step: usize,
    pub total_steps: usize,
    pub step_name: String,
    pub tensors_processed: usize,
    pub total_tensors: usize,
    pub bytes_written: u64,
}

impl ConversionProgress {
    pub fn percentage(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        let step_pct = self.current_step as f64 / self.total_steps as f64;
        let tensor_pct = if self.total_tensors > 0 {
            self.tensors_processed as f64 / self.total_tensors as f64
        } else {
            0.0
        };
        (step_pct + tensor_pct / self.total_steps as f64) * 100.0
    }
}

/// Plan a conversion.
pub fn plan_conversion(
    source: SourceFormat,
    target: TargetFormat,
    model_size_bytes: u64,
    tensor_count: usize,
) -> ConversionPlan {
    let mut steps = Vec::new();

    // Step 1: Parse source
    let parse_time = (model_size_bytes / (1024 * 1024 * 100)).max(1); // ~100MB/s
    steps.push(ConversionStep {
        name: "parse".into(),
        description: format!("Parse {} source", source.as_str()),
        estimated_seconds: parse_time,
    });

    // Step 2: Validate tensors
    steps.push(ConversionStep {
        name: "validate".into(),
        description: format!("Validate {tensor_count} tensors"),
        estimated_seconds: (tensor_count as u64 / 100).max(1),
    });

    // Step 3: Convert types if needed
    let needs_convert = source != SourceFormat::Gguf;
    if needs_convert {
        steps.push(ConversionStep {
            name: "convert_types".into(),
            description: "Convert tensor types".into(),
            estimated_seconds: parse_time,
        });
    }

    // Step 4: Quantize if needed
    if target.quantized() {
        steps.push(ConversionStep {
            name: "quantize".into(),
            description: format!("Quantize to {}", target.as_str()),
            estimated_seconds: parse_time * 3,
        });
    }

    // Step 5: Write output
    steps.push(ConversionStep {
        name: "write".into(),
        description: format!("Write {}", target.as_str()),
        estimated_seconds: parse_time,
    });

    let estimated_total_seconds = steps.iter().map(|s| s.estimated_seconds).sum();
    let memory_estimate_bytes = model_size_bytes * 2; // source + working copy

    ConversionPlan {
        source,
        target,
        steps,
        estimated_total_seconds,
        requires_quantization: target.quantized(),
        memory_estimate_bytes,
    }
}

/// Supported conversion paths.
pub fn supported_conversions() -> HashMap<SourceFormat, Vec<TargetFormat>> {
    let mut map = HashMap::new();
    map.insert(
        SourceFormat::SafeTensors,
        vec![
            TargetFormat::GgufF16,
            TargetFormat::GgufF32,
            TargetFormat::GgufQ4,
            TargetFormat::GgufQ8,
            TargetFormat::GgufI2S,
        ],
    );
    map.insert(
        SourceFormat::Gguf,
        vec![
            TargetFormat::GgufF16,
            TargetFormat::GgufF32,
            TargetFormat::GgufQ4,
            TargetFormat::GgufQ8,
        ],
    );
    map.insert(SourceFormat::PyTorch, vec![TargetFormat::GgufF16, TargetFormat::GgufF32]);
    map
}

pub fn is_supported(source: SourceFormat, target: TargetFormat) -> bool {
    supported_conversions().get(&source).is_some_and(|targets| targets.contains(&target))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_from_ext() {
        assert_eq!(SourceFormat::from_extension("safetensors"), Some(SourceFormat::SafeTensors));
        assert_eq!(SourceFormat::from_extension("gguf"), Some(SourceFormat::Gguf));
        assert_eq!(SourceFormat::from_extension("pt"), Some(SourceFormat::PyTorch));
        assert!(SourceFormat::from_extension("txt").is_none());
    }

    #[test]
    fn test_target_quantized() {
        assert!(TargetFormat::GgufQ4.quantized());
        assert!(TargetFormat::GgufI2S.quantized());
        assert!(!TargetFormat::GgufF16.quantized());
    }

    #[test]
    fn test_plan_safetensors_f16() {
        let plan =
            plan_conversion(SourceFormat::SafeTensors, TargetFormat::GgufF16, 1_000_000_000, 200);
        assert_eq!(plan.source, SourceFormat::SafeTensors);
        assert!(!plan.requires_quantization);
        assert!(plan.step_count() >= 3);
    }

    #[test]
    fn test_plan_with_quantization() {
        let plan =
            plan_conversion(SourceFormat::SafeTensors, TargetFormat::GgufQ4, 1_000_000_000, 200);
        assert!(plan.requires_quantization);
        assert!(plan.steps.iter().any(|s| s.name == "quantize"));
    }

    #[test]
    fn test_plan_gguf_to_gguf() {
        let plan = plan_conversion(SourceFormat::Gguf, TargetFormat::GgufF16, 500_000_000, 100);
        // No convert_types step for GGUF→GGUF
        assert!(!plan.steps.iter().any(|s| s.name == "convert_types"));
    }

    #[test]
    fn test_memory_estimate() {
        let plan = plan_conversion(SourceFormat::SafeTensors, TargetFormat::GgufF16, 1_000_000, 10);
        assert_eq!(plan.memory_estimate_bytes, 2_000_000);
    }

    #[test]
    fn test_supported() {
        assert!(is_supported(SourceFormat::SafeTensors, TargetFormat::GgufF16));
        assert!(is_supported(SourceFormat::SafeTensors, TargetFormat::GgufI2S));
        assert!(!is_supported(SourceFormat::Onnx, TargetFormat::GgufF16));
    }

    #[test]
    fn test_progress_percentage() {
        let p = ConversionProgress {
            current_step: 1,
            total_steps: 4,
            step_name: "validate".into(),
            tensors_processed: 50,
            total_tensors: 100,
            bytes_written: 0,
        };
        let pct = p.percentage();
        assert!(pct > 0.0 && pct < 100.0);
    }

    #[test]
    fn test_progress_empty() {
        let p = ConversionProgress {
            current_step: 0,
            total_steps: 0,
            step_name: "".into(),
            tensors_processed: 0,
            total_tensors: 0,
            bytes_written: 0,
        };
        assert_eq!(p.percentage(), 0.0);
    }

    #[test]
    fn test_source_as_str() {
        assert_eq!(SourceFormat::SafeTensors.as_str(), "safetensors");
        assert_eq!(SourceFormat::PyTorch.as_str(), "pytorch");
    }

    #[test]
    fn test_target_as_str() {
        assert_eq!(TargetFormat::GgufF16.as_str(), "gguf-f16");
        assert_eq!(TargetFormat::GgufI2S.as_str(), "gguf-i2s");
    }

    #[test]
    fn test_supported_conversions_count() {
        let s = supported_conversions();
        assert!(s.len() >= 3);
        assert!(s[&SourceFormat::SafeTensors].len() >= 5);
    }
}
