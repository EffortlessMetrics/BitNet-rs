//! Issue #261 GGUF Model Test Fixtures
//!
//! Minimal GGUF model structures for testing tensor alignment, quantization metadata,
//! and weight mapper compatibility. Supports AC5 (QLinear layer replacement tests).
//!
//! GGUF Specification: 32-byte tensor alignment, I2S/TL1/TL2 quantization support.

#![allow(dead_code)]

/// GGUF model test fixture with metadata and tensor information
#[derive(Debug, Clone)]
pub struct GgufModelFixture {
    pub model_id: &'static str,
    pub metadata: GgufMetadata,
    pub tensors: Vec<GgufTensorInfo>,
    pub alignment: u64,
    pub quantization_type: GgufQuantizationType,
    pub weight_mapper_compatible: bool,
    pub validation_flags: ValidationFlags,
    pub description: &'static str,
}

/// GGUF metadata structure
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub magic: u32,   // 0x46554747 ("GGUF")
    pub version: u32, // GGUF version (3)
    pub tensor_count: u64,
    pub kv_count: u64,
    pub architecture: &'static str,
    pub vocab_size: u32,
    pub embedding_dim: u32,
    pub hidden_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub max_seq_len: u32,
}

/// GGUF tensor information
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: &'static str,
    pub shape: Vec<u64>,
    pub qtype: GgufQuantizationType,
    pub offset: u64,
    pub size_bytes: u64,
    pub aligned: bool,
}

/// GGUF quantization type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufQuantizationType {
    F32,
    F16,
    I2S,  // BitNet I2S quantization
    TL1,  // BitNet TL1 quantization
    TL2,  // BitNet TL2 quantization
    IQ2S, // GGML IQ2_S quantization (via FFI)
}

/// Validation flags for GGUF model testing
#[derive(Debug, Clone)]
pub struct ValidationFlags {
    pub check_tensor_alignment: bool,
    pub check_metadata_integrity: bool,
    pub check_quantization_format: bool,
    pub check_weight_mapper_compatible: bool,
    pub check_32byte_alignment: bool,
}

// ============================================================================
// Valid GGUF Model Fixtures
// ============================================================================

/// Load valid GGUF model fixtures with I2S quantization
pub fn load_valid_i2s_model_fixtures() -> Vec<GgufModelFixture> {
    vec![
        // Minimal valid I2S model
        GgufModelFixture {
            model_id: "minimal_i2s_model",
            metadata: GgufMetadata {
                magic: 0x46554747,
                version: 3,
                tensor_count: 4,
                kv_count: 10,
                architecture: "bitnet",
                vocab_size: 32000,
                embedding_dim: 768,
                hidden_dim: 2048,
                num_layers: 2,
                num_heads: 8,
                max_seq_len: 512,
            },
            tensors: vec![
                GgufTensorInfo {
                    name: "token_embd.weight",
                    shape: vec![768, 32000],
                    qtype: GgufQuantizationType::I2S,
                    offset: 0,
                    size_bytes: calculate_i2s_size(768 * 32000),
                    aligned: true,
                },
                GgufTensorInfo {
                    name: "layers.0.attn_q.weight",
                    shape: vec![768, 768],
                    qtype: GgufQuantizationType::I2S,
                    offset: calculate_aligned_offset(calculate_i2s_size(768 * 32000), 32),
                    size_bytes: calculate_i2s_size(768 * 768),
                    aligned: true,
                },
                GgufTensorInfo {
                    name: "layers.0.attn_k.weight",
                    shape: vec![768, 768],
                    qtype: GgufQuantizationType::I2S,
                    offset: calculate_aligned_offset(
                        calculate_i2s_size(768 * 32000) + calculate_i2s_size(768 * 768),
                        32,
                    ),
                    size_bytes: calculate_i2s_size(768 * 768),
                    aligned: true,
                },
                GgufTensorInfo {
                    name: "output.weight",
                    shape: vec![32000, 768],
                    qtype: GgufQuantizationType::I2S,
                    offset: calculate_aligned_offset(
                        calculate_i2s_size(768 * 32000) + 2 * calculate_i2s_size(768 * 768),
                        32,
                    ),
                    size_bytes: calculate_i2s_size(32000 * 768),
                    aligned: true,
                },
            ],
            alignment: 32,
            quantization_type: GgufQuantizationType::I2S,
            weight_mapper_compatible: true,
            validation_flags: ValidationFlags {
                check_tensor_alignment: true,
                check_metadata_integrity: true,
                check_quantization_format: true,
                check_weight_mapper_compatible: true,
                check_32byte_alignment: true,
            },
            description: "Minimal valid GGUF model with I2S quantization and 32-byte alignment",
        },
        // Medium I2S model with more layers
        GgufModelFixture {
            model_id: "medium_i2s_model",
            metadata: GgufMetadata {
                magic: 0x46554747,
                version: 3,
                tensor_count: 8,
                kv_count: 12,
                architecture: "bitnet",
                vocab_size: 50000,
                embedding_dim: 1024,
                hidden_dim: 4096,
                num_layers: 4,
                num_heads: 16,
                max_seq_len: 1024,
            },
            tensors: generate_medium_tensors(),
            alignment: 32,
            quantization_type: GgufQuantizationType::I2S,
            weight_mapper_compatible: true,
            validation_flags: ValidationFlags {
                check_tensor_alignment: true,
                check_metadata_integrity: true,
                check_quantization_format: true,
                check_weight_mapper_compatible: true,
                check_32byte_alignment: true,
            },
            description: "Medium GGUF model with I2S quantization for comprehensive testing",
        },
    ]
}

/// Load valid GGUF model fixtures with TL1/TL2 quantization
pub fn load_valid_tl_model_fixtures() -> Vec<GgufModelFixture> {
    vec![
        // TL1 model (ARM NEON optimized)
        GgufModelFixture {
            model_id: "minimal_tl1_model",
            metadata: GgufMetadata {
                magic: 0x46554747,
                version: 3,
                tensor_count: 4,
                kv_count: 10,
                architecture: "bitnet",
                vocab_size: 32000,
                embedding_dim: 768,
                hidden_dim: 2048,
                num_layers: 2,
                num_heads: 8,
                max_seq_len: 512,
            },
            tensors: generate_tl_tensors(GgufQuantizationType::TL1),
            alignment: 32,
            quantization_type: GgufQuantizationType::TL1,
            weight_mapper_compatible: true,
            validation_flags: ValidationFlags {
                check_tensor_alignment: true,
                check_metadata_integrity: true,
                check_quantization_format: true,
                check_weight_mapper_compatible: true,
                check_32byte_alignment: true,
            },
            description: "TL1 quantized model for ARM NEON optimization testing",
        },
        // TL2 model (x86 AVX2/AVX-512 optimized)
        GgufModelFixture {
            model_id: "minimal_tl2_model",
            metadata: GgufMetadata {
                magic: 0x46554747,
                version: 3,
                tensor_count: 4,
                kv_count: 10,
                architecture: "bitnet",
                vocab_size: 32000,
                embedding_dim: 768,
                hidden_dim: 2048,
                num_layers: 2,
                num_heads: 8,
                max_seq_len: 512,
            },
            tensors: generate_tl_tensors(GgufQuantizationType::TL2),
            alignment: 32,
            quantization_type: GgufQuantizationType::TL2,
            weight_mapper_compatible: true,
            validation_flags: ValidationFlags {
                check_tensor_alignment: true,
                check_metadata_integrity: true,
                check_quantization_format: true,
                check_weight_mapper_compatible: true,
                check_32byte_alignment: true,
            },
            description: "TL2 quantized model for x86 AVX2/AVX-512 optimization testing",
        },
    ]
}

// ============================================================================
// Invalid/Edge Case GGUF Model Fixtures
// ============================================================================

/// Load corrupted GGUF model fixtures for error handling validation
pub fn load_corrupted_model_fixtures() -> Vec<GgufModelFixture> {
    vec![
        // Invalid magic number
        GgufModelFixture {
            model_id: "invalid_magic",
            metadata: GgufMetadata {
                magic: 0xDEADBEEF, // Invalid magic
                version: 3,
                tensor_count: 2,
                kv_count: 5,
                architecture: "bitnet",
                vocab_size: 1000,
                embedding_dim: 128,
                hidden_dim: 512,
                num_layers: 1,
                num_heads: 4,
                max_seq_len: 256,
            },
            tensors: vec![],
            alignment: 32,
            quantization_type: GgufQuantizationType::I2S,
            weight_mapper_compatible: false,
            validation_flags: ValidationFlags {
                check_tensor_alignment: false,
                check_metadata_integrity: true,
                check_quantization_format: false,
                check_weight_mapper_compatible: false,
                check_32byte_alignment: false,
            },
            description: "Corrupted GGUF with invalid magic number",
        },
        // Misaligned tensors
        GgufModelFixture {
            model_id: "misaligned_tensors",
            metadata: GgufMetadata {
                magic: 0x46554747,
                version: 3,
                tensor_count: 2,
                kv_count: 5,
                architecture: "bitnet",
                vocab_size: 1000,
                embedding_dim: 128,
                hidden_dim: 512,
                num_layers: 1,
                num_heads: 4,
                max_seq_len: 256,
            },
            tensors: vec![GgufTensorInfo {
                name: "test.weight",
                shape: vec![128, 128],
                qtype: GgufQuantizationType::I2S,
                offset: 17, // Not 32-byte aligned
                size_bytes: calculate_i2s_size(128 * 128),
                aligned: false,
            }],
            alignment: 32,
            quantization_type: GgufQuantizationType::I2S,
            weight_mapper_compatible: false,
            validation_flags: ValidationFlags {
                check_tensor_alignment: true,
                check_metadata_integrity: true,
                check_quantization_format: true,
                check_weight_mapper_compatible: false,
                check_32byte_alignment: true,
            },
            description: "GGUF model with misaligned tensors (not 32-byte aligned)",
        },
        // Mixed quantization types (unsupported)
        GgufModelFixture {
            model_id: "mixed_quantization",
            metadata: GgufMetadata {
                magic: 0x46554747,
                version: 3,
                tensor_count: 3,
                kv_count: 5,
                architecture: "bitnet",
                vocab_size: 1000,
                embedding_dim: 128,
                hidden_dim: 512,
                num_layers: 1,
                num_heads: 4,
                max_seq_len: 256,
            },
            tensors: vec![
                GgufTensorInfo {
                    name: "weight1",
                    shape: vec![128, 128],
                    qtype: GgufQuantizationType::I2S,
                    offset: 0,
                    size_bytes: calculate_i2s_size(128 * 128),
                    aligned: true,
                },
                GgufTensorInfo {
                    name: "weight2",
                    shape: vec![128, 128],
                    qtype: GgufQuantizationType::TL1,
                    offset: calculate_aligned_offset(calculate_i2s_size(128 * 128), 32),
                    size_bytes: calculate_i2s_size(128 * 128),
                    aligned: true,
                },
            ],
            alignment: 32,
            quantization_type: GgufQuantizationType::I2S, // Inconsistent
            weight_mapper_compatible: false,
            validation_flags: ValidationFlags {
                check_tensor_alignment: true,
                check_metadata_integrity: true,
                check_quantization_format: true,
                check_weight_mapper_compatible: false,
                check_32byte_alignment: true,
            },
            description: "GGUF model with mixed quantization types (error condition)",
        },
    ]
}

// ============================================================================
// Tensor Alignment Test Fixtures
// ============================================================================

/// Load tensor alignment test fixtures
pub fn load_tensor_alignment_fixtures() -> Vec<TensorAlignmentFixture> {
    vec![
        TensorAlignmentFixture {
            test_id: "valid_32byte_alignment",
            offset: 0,
            size: 1024,
            alignment_requirement: 32,
            expected_aligned: true,
            description: "Tensor with valid 32-byte alignment (offset 0)",
        },
        TensorAlignmentFixture {
            test_id: "valid_64byte_alignment",
            offset: 64,
            size: 2048,
            alignment_requirement: 32,
            expected_aligned: true,
            description: "Tensor with valid alignment (offset 64, 32-byte requirement)",
        },
        TensorAlignmentFixture {
            test_id: "invalid_alignment_17",
            offset: 17,
            size: 1024,
            alignment_requirement: 32,
            expected_aligned: false,
            description: "Tensor with invalid alignment (offset 17, not 32-byte aligned)",
        },
        TensorAlignmentFixture {
            test_id: "invalid_alignment_31",
            offset: 31,
            size: 1024,
            alignment_requirement: 32,
            expected_aligned: false,
            description: "Tensor with invalid alignment (offset 31, off by 1)",
        },
        TensorAlignmentFixture {
            test_id: "valid_128byte_alignment",
            offset: 128,
            size: 4096,
            alignment_requirement: 32,
            expected_aligned: true,
            description: "Tensor with valid alignment (offset 128, exceeds requirement)",
        },
    ]
}

/// Tensor alignment test fixture
#[derive(Debug, Clone)]
pub struct TensorAlignmentFixture {
    pub test_id: &'static str,
    pub offset: u64,
    pub size: u64,
    pub alignment_requirement: u64,
    pub expected_aligned: bool,
    pub description: &'static str,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate I2S quantized tensor size in bytes
/// I2S uses 2 bits per weight + scale factors
fn calculate_i2s_size(num_elements: u64) -> u64 {
    const BLOCK_SIZE: u64 = 32;
    let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
    let quantized_bytes = (num_elements * 2 + 7) / 8; // 2 bits per element
    let scale_bytes = num_blocks * 4; // f32 per block
    quantized_bytes + scale_bytes
}

/// Calculate aligned offset
fn calculate_aligned_offset(current_offset: u64, alignment: u64) -> u64 {
    ((current_offset + alignment - 1) / alignment) * alignment
}

/// Generate medium-sized tensor list
fn generate_medium_tensors() -> Vec<GgufTensorInfo> {
    let mut offset = 0u64;

    let tensors = vec![
        ("token_embd.weight", vec![1024, 50000]),
        ("layers.0.attn_q.weight", vec![1024, 1024]),
        ("layers.0.attn_k.weight", vec![1024, 1024]),
        ("layers.0.attn_v.weight", vec![1024, 1024]),
        ("layers.0.mlp_up.weight", vec![4096, 1024]),
        ("layers.0.mlp_down.weight", vec![1024, 4096]),
        ("layers.1.attn_q.weight", vec![1024, 1024]),
        ("output.weight", vec![50000, 1024]),
    ];

    tensors
        .into_iter()
        .map(|(name, shape)| {
            let num_elements = shape.iter().product::<u64>();
            let size = calculate_i2s_size(num_elements);
            let tensor = GgufTensorInfo {
                name,
                shape,
                qtype: GgufQuantizationType::I2S,
                offset,
                size_bytes: size,
                aligned: offset % 32 == 0,
            };
            offset = calculate_aligned_offset(offset + size, 32);
            tensor
        })
        .collect()
}

/// Generate TL quantization tensors
fn generate_tl_tensors(qtype: GgufQuantizationType) -> Vec<GgufTensorInfo> {
    let mut offset = 0u64;

    let tensors = vec![
        ("token_embd.weight", vec![768, 32000]),
        ("layers.0.attn_q.weight", vec![768, 768]),
        ("layers.0.attn_k.weight", vec![768, 768]),
        ("output.weight", vec![32000, 768]),
    ];

    tensors
        .into_iter()
        .map(|(name, shape)| {
            let num_elements = shape.iter().product::<u64>();
            let size = calculate_i2s_size(num_elements); // TL uses similar size calculation
            let tensor = GgufTensorInfo {
                name,
                shape,
                qtype,
                offset,
                size_bytes: size,
                aligned: offset % 32 == 0,
            };
            offset = calculate_aligned_offset(offset + size, 32);
            tensor
        })
        .collect()
}

/// Validate GGUF model fixture integrity
pub fn validate_gguf_fixture(fixture: &GgufModelFixture) -> Result<(), String> {
    // Check magic number
    if fixture.metadata.magic != 0x46554747 && fixture.model_id != "invalid_magic" {
        return Err("Invalid GGUF magic number".to_string());
    }

    // Check tensor alignment if required
    if fixture.validation_flags.check_32byte_alignment {
        for tensor in &fixture.tensors {
            if tensor.aligned && tensor.offset % fixture.alignment != 0 {
                return Err(format!(
                    "Tensor {} at offset {} not {}-byte aligned",
                    tensor.name, tensor.offset, fixture.alignment
                ));
            }
        }
    }

    // Check tensor count matches metadata
    if fixture.tensors.len() as u64 != fixture.metadata.tensor_count {
        return Err(format!(
            "Tensor count mismatch: metadata says {}, found {}",
            fixture.metadata.tensor_count,
            fixture.tensors.len()
        ));
    }

    Ok(())
}

/// Get fixture by model ID
pub fn get_gguf_fixture_by_id(model_id: &str) -> Option<GgufModelFixture> {
    let mut all_fixtures = load_valid_i2s_model_fixtures();
    all_fixtures.extend(load_valid_tl_model_fixtures());
    all_fixtures.extend(load_corrupted_model_fixtures());

    all_fixtures.into_iter().find(|f| f.model_id == model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i2s_size_calculation() {
        let size = calculate_i2s_size(1024);
        assert!(size > 0, "I2S size should be positive");
        // 1024 elements / 32 block size = 32 blocks = 32 * 4 = 128 bytes for scales
        // 1024 * 2 bits = 2048 bits = 256 bytes for quantized data
        // Total ≈ 384 bytes
        assert!(size >= 256 && size <= 512, "I2S size should be reasonable");
    }

    #[test]
    fn test_aligned_offset_calculation() {
        assert_eq!(calculate_aligned_offset(0, 32), 0);
        assert_eq!(calculate_aligned_offset(17, 32), 32);
        assert_eq!(calculate_aligned_offset(32, 32), 32);
        assert_eq!(calculate_aligned_offset(33, 32), 64);
    }

    #[test]
    fn test_valid_fixture_validation() {
        let fixtures = load_valid_i2s_model_fixtures();
        for fixture in fixtures {
            validate_gguf_fixture(&fixture).expect("Valid fixture should pass validation");
        }
    }

    #[test]
    fn test_tensor_alignment_validation() {
        let fixtures = load_tensor_alignment_fixtures();
        for fixture in fixtures {
            let is_aligned = fixture.offset % fixture.alignment_requirement == 0;
            assert_eq!(
                is_aligned, fixture.expected_aligned,
                "Alignment validation mismatch for {}",
                fixture.test_id
            );
        }
    }
}
