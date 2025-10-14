//! Mock quantized model fixtures for testing strict quantization guards
//!
//! This module provides mock implementations of quantized layers and models
//! for testing the three-tier validation strategy without requiring actual
//! model files or GPU hardware.

use bitnet_common::device::Device;
use bitnet_common::error::BitNetError;
use bitnet_quantization::QuantizationType;
use anyhow::Result;

/// Mock quantized linear layer for testing
///
/// **Implementation Status:** ❌ NOT IMPLEMENTED
/// This is test scaffolding - actual implementation required for AC1/AC3 tests.
pub struct MockQuantizedLinear {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub quantization_type: QuantizationType,
    pub device: Device,
    pub kernel_available: bool,
}

impl MockQuantizedLinear {
    /// Create a new mock quantized linear layer
    pub fn new(
        name: impl Into<String>,
        in_features: usize,
        out_features: usize,
        qtype: QuantizationType,
        device: Device,
    ) -> Self {
        Self {
            name: name.into(),
            in_features,
            out_features,
            quantization_type: qtype,
            device,
            kernel_available: true, // Default: kernel available
        }
    }

    /// Force kernel unavailability for testing fallback scenarios
    pub fn with_unavailable_kernel(mut self) -> Self {
        self.kernel_available = false;
        self
    }

    /// Check if native quantized kernel is available
    pub fn has_native_quantized_kernel(&self) -> bool {
        // TODO: Implement actual kernel availability check
        // For now, return mock value
        self.kernel_available
    }

    /// Forward pass with strict mode validation
    ///
    /// **Implementation Status:** ❌ NOT IMPLEMENTED
    /// This is where AC1 debug assertions and AC3 strict mode checks will be added.
    pub async fn forward(&self, _input: &MockTensor) -> Result<MockTensor> {
        // TODO: Implement forward pass with:
        // 1. Debug assertions (AC1)
        // 2. Strict mode checks (AC3)
        // 3. Kernel availability validation
        // 4. Fallback path (if strict mode disabled)

        if !self.has_native_quantized_kernel() {
            #[cfg(debug_assertions)]
            panic!(
                "fallback to FP32 in debug mode: layer={}, qtype={:?}, device={:?}",
                self.name, self.quantization_type, self.device
            );

            // Strict mode check would go here
            return Err(BitNetError::StrictMode(format!(
                "FP32 fallback rejected - qtype={:?}, device={:?}, layer_dims=[{}, {}], reason=kernel_unavailable",
                self.quantization_type, self.device, self.in_features, self.out_features
            )).into());
        }

        // Successful forward pass (quantized path)
        Ok(MockTensor::new(vec![self.out_features]))
    }
}

/// Mock attention layer for testing Q/K/V/O projection validation
///
/// **Implementation Status:** ❌ NOT IMPLEMENTED
/// This is test scaffolding for AC2/AC4 tests.
pub struct MockBitNetAttention {
    pub q_proj: MockQuantizedLinear,
    pub k_proj: MockQuantizedLinear,
    pub v_proj: MockQuantizedLinear,
    pub o_proj: MockQuantizedLinear,
}

impl MockBitNetAttention {
    /// Create a new mock attention layer
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        qtype: QuantizationType,
        device: Device,
    ) -> Self {
        let head_dim = hidden_size / num_heads;

        Self {
            q_proj: MockQuantizedLinear::new("q_proj", hidden_size, hidden_size, qtype, device),
            k_proj: MockQuantizedLinear::new("k_proj", hidden_size, hidden_size, qtype, device),
            v_proj: MockQuantizedLinear::new("v_proj", hidden_size, hidden_size, qtype, device),
            o_proj: MockQuantizedLinear::new("o_proj", hidden_size, hidden_size, qtype, device),
        }
    }

    /// Force one projection to have unavailable kernel for testing
    pub fn with_unavailable_projection(mut self, projection: &str) -> Self {
        match projection {
            "q" => self.q_proj.kernel_available = false,
            "k" => self.k_proj.kernel_available = false,
            "v" => self.v_proj.kernel_available = false,
            "o" => self.o_proj.kernel_available = false,
            _ => panic!("Invalid projection name: {}", projection),
        }
        self
    }

    /// Validate all projections have quantized kernels (AC2, AC4)
    ///
    /// **Implementation Status:** ❌ NOT IMPLEMENTED
    pub fn validate_projections_quantized(&self) -> Result<()> {
        let projections = [
            ("Q", &self.q_proj),
            ("K", &self.k_proj),
            ("V", &self.v_proj),
            ("O", &self.o_proj),
        ];

        for (name, proj) in &projections {
            let has_native_kernel = proj.has_native_quantized_kernel();

            if !has_native_kernel {
                #[cfg(debug_assertions)]
                panic!(
                    "fallback to FP32 in debug mode: {} projection would fall back",
                    name
                );

                // Strict mode check would go here
                return Err(BitNetError::StrictMode(format!(
                    "{} projection would fall back to FP32 - qtype={:?}, device={:?}",
                    name, proj.quantization_type, proj.device
                )).into());
            }
        }

        Ok(())
    }

    /// Forward pass with strict mode validation
    ///
    /// **Implementation Status:** ❌ NOT IMPLEMENTED
    pub async fn forward(&self, _hidden_states: &MockTensor) -> Result<MockTensor> {
        // Validate projections before forward pass
        self.validate_projections_quantized()?;

        // TODO: Implement actual attention computation
        // For now, return mock output
        Ok(MockTensor::new(vec![_hidden_states.shape[0]]))
    }
}

/// Mock tensor for testing (placeholder)
///
/// **Implementation Status:** ❌ NOT IMPLEMENTED
#[derive(Debug, Clone)]
pub struct MockTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl MockTensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    pub fn with_data(shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self { shape, data }
    }
}

/// Mock model for integration testing
///
/// **Implementation Status:** ❌ NOT IMPLEMENTED
pub struct MockBitNetModel {
    pub layers: Vec<MockBitNetAttention>,
    pub device: Device,
}

impl MockBitNetModel {
    /// Create a new mock model with specified number of layers
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        qtype: QuantizationType,
        device: Device,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| MockBitNetAttention::new(hidden_size, num_heads, qtype, device))
            .collect();

        Self { layers, device }
    }

    /// Run forward pass through all layers
    ///
    /// **Implementation Status:** ❌ NOT IMPLEMENTED
    pub async fn forward(&self, input: &MockTensor) -> Result<MockTensor> {
        let mut hidden_states = input.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states).await?;
        }

        Ok(hidden_states)
    }
}

/// Mock tokenizer for integration testing
///
/// **Implementation Status:** ❌ NOT IMPLEMENTED
pub struct MockTokenizer {
    pub vocab_size: usize,
}

impl MockTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Simple mock: return token IDs based on text length
        vec![1; text.len().min(10)]
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        // Simple mock: return placeholder text
        format!("decoded_{}_tokens", tokens.len())
    }
}

/// Mock inference result for testing
///
/// **Implementation Status:** ❌ NOT IMPLEMENTED
#[derive(Debug)]
pub struct MockInferenceResult {
    pub tokens_generated: usize,
    pub output_text: String,
    pub receipt: MockReceipt,
}

/// Mock receipt for testing (schema v1.1.0)
///
/// **Implementation Status:** ❌ NOT IMPLEMENTED
#[derive(Debug, Clone)]
pub struct MockReceipt {
    pub schema_version: String,
    pub compute_path: String,
    pub backend: String,
    pub kernel_path: Option<String>,
    pub kernels: Vec<String>,
    pub tokens_per_second: f64,
}

impl MockReceipt {
    /// Create a receipt with native quantized kernels
    pub fn with_native_quantized(backend: &str) -> Self {
        Self {
            schema_version: "1.1.0".to_string(),
            compute_path: "real".to_string(),
            backend: backend.to_string(),
            kernel_path: Some("native_quantized".to_string()),
            kernels: vec![
                "gemm_fp16".to_string(),
                "i2s_gpu_quantize".to_string(),
                "wmma_matmul".to_string(),
            ],
            tokens_per_second: 87.5,
        }
    }

    /// Create a receipt with FP32 fallback
    pub fn with_fp32_fallback(backend: &str) -> Self {
        Self {
            schema_version: "1.1.0".to_string(),
            compute_path: "real".to_string(),
            backend: backend.to_string(),
            kernel_path: Some("fp32_fallback".to_string()),
            kernels: vec![
                "dequant_i2s".to_string(),
                "fp32_matmul".to_string(),
                "cuda_sync".to_string(),
            ],
            tokens_per_second: 35.0,
        }
    }
}

/// Run mock inference for integration testing
///
/// **Implementation Status:** ❌ NOT IMPLEMENTED
pub async fn run_mock_inference(
    model: &MockBitNetModel,
    tokenizer: &MockTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> Result<MockInferenceResult> {
    // TODO: Implement mock inference loop
    // 1. Encode prompt
    // 2. Run forward pass for max_tokens iterations
    // 3. Track kernel usage for receipt
    // 4. Return result with receipt

    let tokens = tokenizer.encode(prompt);
    let input = MockTensor::new(vec![tokens.len()]);
    let _output = model.forward(&input).await?;

    Ok(MockInferenceResult {
        tokens_generated: max_tokens,
        output_text: tokenizer.decode(&tokens),
        receipt: MockReceipt::with_native_quantized("cpu"),
    })
}

// =============================================================================
// Test Helpers
// =============================================================================

/// Create a test model with forced fallback scenarios
pub fn create_test_model_with_fallback() -> MockBitNetModel {
    let mut model = MockBitNetModel::new(
        2,
        768,
        12,
        QuantizationType::I2S,
        Device::Cpu,
    );

    // Force first layer's Q projection to have unavailable kernel
    model.layers[0].q_proj.kernel_available = false;

    model
}

/// Create a test model with all quantized kernels available
pub fn create_test_model_quantized() -> MockBitNetModel {
    MockBitNetModel::new(
        2,
        768,
        12,
        QuantizationType::I2S,
        Device::Cpu,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_quantized_linear_creation() {
        let layer = MockQuantizedLinear::new(
            "test_layer",
            2048,
            2048,
            QuantizationType::I2S,
            Device::Cpu,
        );

        assert_eq!(layer.name, "test_layer");
        assert_eq!(layer.in_features, 2048);
        assert_eq!(layer.out_features, 2048);
        assert!(layer.has_native_quantized_kernel());
    }

    #[test]
    fn test_mock_attention_creation() {
        let attention = MockBitNetAttention::new(
            768,
            12,
            QuantizationType::I2S,
            Device::Cpu,
        );

        assert_eq!(attention.q_proj.in_features, 768);
        assert!(attention.q_proj.has_native_quantized_kernel());
    }

    #[test]
    fn test_mock_receipt_creation() {
        let receipt = MockReceipt::with_native_quantized("cuda");
        assert_eq!(receipt.kernel_path, Some("native_quantized".to_string()));
        assert_eq!(receipt.kernels.len(), 3);
    }
}
