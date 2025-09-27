//! GGUF Test Model Generator for BitNet.rs Neural Network Testing
//!
//! This module provides realistic GGUF model fixture generation for comprehensive
//! BitNet quantization testing (I2_S, TL1, TL2), tensor alignment validation,
//! and device-aware operations.

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// GGUF fixture configuration for BitNet testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufFixtureConfig {
    pub name: String,
    pub model_type: ModelType,
    pub quantization_type: QuantizationType,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub tensor_alignment: u64,
    pub generate_invalid: bool,
    pub seed: u64,
}

/// Supported model types for BitNet testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    BitNet158_1B,
    BitNet158_3B,
    BitNetB1_58_2B,
    Minimal, // For fast CI testing
}

/// BitNet quantization types supported in fixtures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    I2S,  // 2-bit signed quantization
    TL1,  // Table lookup 1
    TL2,  // Table lookup 2
    IQ2S, // GGML-compatible 2-bit
    FP32, // Unquantized baseline
}

/// Generated GGUF fixture metadata
#[derive(Debug, Clone)]
pub struct GgufFixture {
    pub path: PathBuf,
    pub config: GgufFixtureConfig,
    pub tensors: Vec<TensorInfo>,
    pub file_size: u64,
    pub checksum: String,
}

/// Tensor information for validation
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: DataType,
    pub offset: u64,
    pub size: u64,
    pub quantized: bool,
}

/// GGUF data types
#[derive(Debug, Clone, Copy)]
pub enum DataType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    I2S = 50,  // Custom BitNet 2-bit signed
    TL1 = 51,  // Custom BitNet table lookup 1
    TL2 = 52,  // Custom BitNet table lookup 2
    IQ2S = 53, // GGML compatible 2-bit
}

/// GGUF fixture generator for BitNet testing
pub struct GgufFixtureGenerator {
    output_dir: PathBuf,
    seed: u64,
}

impl GgufFixtureGenerator {
    pub fn new(output_dir: PathBuf, seed: u64) -> Self {
        Self { output_dir, seed }
    }

    /// Generate a complete set of BitNet GGUF test fixtures
    pub fn generate_bitnet_fixture_set(&self) -> Result<Vec<GgufFixture>> {
        let mut fixtures = Vec::new();

        // Valid fixtures for different quantization types
        for quant_type in &[
            QuantizationType::I2S,
            QuantizationType::TL1,
            QuantizationType::TL2,
            QuantizationType::IQ2S,
        ] {
            fixtures.push(self.generate_fixture(&GgufFixtureConfig {
                name: format!("bitnet_minimal_{:?}", quant_type).to_lowercase(),
                model_type: ModelType::Minimal,
                quantization_type: quant_type.clone(),
                vocab_size: 1000,
                hidden_size: 128,
                num_layers: 2,
                tensor_alignment: 32,
                generate_invalid: false,
                seed: self.seed,
            })?);
        }

        // Realistic BitNet 1.58B model fixtures
        fixtures.push(self.generate_fixture(&GgufFixtureConfig {
            name: "bitnet_158_1b_i2s".to_string(),
            model_type: ModelType::BitNet158_1B,
            quantization_type: QuantizationType::I2S,
            vocab_size: 32000,
            hidden_size: 2048,
            num_layers: 24,
            tensor_alignment: 32,
            generate_invalid: false,
            seed: self.seed,
        })?);

        // Test alignment scenarios
        fixtures.push(self.generate_fixture(&GgufFixtureConfig {
            name: "bitnet_alignment_test".to_string(),
            model_type: ModelType::Minimal,
            quantization_type: QuantizationType::I2S,
            vocab_size: 256,
            hidden_size: 64,
            num_layers: 1,
            tensor_alignment: 64, // Non-standard alignment
            generate_invalid: false,
            seed: self.seed,
        })?);

        // Invalid fixtures for error testing
        fixtures.push(self.generate_fixture(&GgufFixtureConfig {
            name: "bitnet_invalid_header".to_string(),
            model_type: ModelType::Minimal,
            quantization_type: QuantizationType::I2S,
            vocab_size: 100,
            hidden_size: 32,
            num_layers: 1,
            tensor_alignment: 32,
            generate_invalid: true,
            seed: self.seed,
        })?);

        Ok(fixtures)
    }

    /// Generate a single GGUF fixture
    pub fn generate_fixture(&self, config: &GgufFixtureConfig) -> Result<GgufFixture> {
        let file_path = self.output_dir.join(format!("{}.gguf", config.name));

        // Ensure output directory exists
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut writer = BufWriter::new(File::create(&file_path)?);
        let mut tensors = Vec::new();
        let mut current_offset = 0u64;

        // Write GGUF header
        current_offset += self.write_gguf_header(&mut writer, config)?;

        // Write metadata
        current_offset += self.write_metadata(&mut writer, config)?;

        // Generate and write tensors
        let tensor_configs =
            self.get_tensor_configs_for_model(&config.model_type, &config.quantization_type);

        for tensor_config in tensor_configs {
            let tensor_info =
                self.write_tensor(&mut writer, &tensor_config, config, current_offset)?;
            current_offset += tensor_info.size;
            tensors.push(tensor_info);
        }

        writer.flush()?;
        drop(writer);

        let file_size = std::fs::metadata(&file_path)?.len();
        let checksum = self.calculate_checksum(&file_path)?;

        Ok(GgufFixture { path: file_path, config: config.clone(), tensors, file_size, checksum })
    }

    /// Write GGUF header with magic and version
    fn write_gguf_header(
        &self,
        writer: &mut BufWriter<File>,
        config: &GgufFixtureConfig,
    ) -> Result<u64> {
        if config.generate_invalid {
            // Write invalid magic for error testing
            writer.write_all(b"FAKE")?;
        } else {
            writer.write_all(b"GGUF")?;
        }

        WriteBytesExt::write_u32::<LittleEndian>(writer, 3)?; // Version 3

        Ok(8) // 4 bytes magic + 4 bytes version
    }

    /// Write GGUF metadata section
    fn write_metadata(
        &self,
        writer: &mut BufWriter<File>,
        config: &GgufFixtureConfig,
    ) -> Result<u64> {
        let mut bytes_written = 0u64;

        // Number of tensor info entries
        let tensor_configs =
            self.get_tensor_configs_for_model(&config.model_type, &config.quantization_type);
        WriteBytesExt::write_u64::<LittleEndian>(writer, tensor_configs.len() as u64)?;
        bytes_written += 8;

        // Number of KV metadata entries
        WriteBytesExt::write_u64::<LittleEndian>(writer, 5)?; // model.type, vocab_size, hidden_size, layers, quantization
        bytes_written += 8;

        // Write metadata KV pairs
        bytes_written +=
            self.write_kv_metadata(writer, "model.type", &format!("{:?}", config.model_type))?;
        bytes_written +=
            self.write_kv_metadata(writer, "bitnet.vocab_size", &config.vocab_size.to_string())?;
        bytes_written +=
            self.write_kv_metadata(writer, "bitnet.hidden_size", &config.hidden_size.to_string())?;
        bytes_written +=
            self.write_kv_metadata(writer, "bitnet.num_layers", &config.num_layers.to_string())?;
        bytes_written += self.write_kv_metadata(
            writer,
            "bitnet.quantization",
            &format!("{:?}", config.quantization_type),
        )?;

        // Write tensor info entries
        for tensor_config in &tensor_configs {
            bytes_written += self.write_tensor_info(writer, tensor_config)?;
        }

        Ok(bytes_written)
    }

    /// Write a key-value metadata pair
    fn write_kv_metadata(
        &self,
        writer: &mut BufWriter<File>,
        key: &str,
        value: &str,
    ) -> Result<u64> {
        let mut bytes_written = 0u64;

        // Key length and data
        WriteBytesExt::write_u64::<LittleEndian>(writer, key.len() as u64)?;
        writer.write_all(key.as_bytes())?;
        bytes_written += 8 + key.len() as u64;

        // Value type (string = 8)
        WriteBytesExt::write_u32::<LittleEndian>(writer, 8)?;
        bytes_written += 4;

        // Value length and data
        WriteBytesExt::write_u64::<LittleEndian>(writer, value.len() as u64)?;
        writer.write_all(value.as_bytes())?;
        bytes_written += 8 + value.len() as u64;

        Ok(bytes_written)
    }

    /// Write tensor information entry
    fn write_tensor_info(
        &self,
        writer: &mut BufWriter<File>,
        config: &TensorConfig,
    ) -> Result<u64> {
        let mut bytes_written = 0u64;

        // Tensor name
        WriteBytesExt::write_u64::<LittleEndian>(writer, config.name.len() as u64)?;
        writer.write_all(config.name.as_bytes())?;
        bytes_written += 8 + config.name.len() as u64;

        // Number of dimensions
        WriteBytesExt::write_u32::<LittleEndian>(writer, config.shape.len() as u32)?;
        bytes_written += 4;

        // Shape dimensions
        for &dim in &config.shape {
            WriteBytesExt::write_u64::<LittleEndian>(writer, dim as u64)?;
            bytes_written += 8;
        }

        // Data type
        WriteBytesExt::write_u32::<LittleEndian>(writer, config.data_type as u32)?;
        bytes_written += 4;

        // Tensor data offset (will be calculated later)
        WriteBytesExt::write_u64::<LittleEndian>(writer, 0)?; // Placeholder
        bytes_written += 8;

        Ok(bytes_written)
    }

    /// Write actual tensor data
    fn write_tensor(
        &self,
        writer: &mut BufWriter<File>,
        config: &TensorConfig,
        fixture_config: &GgufFixtureConfig,
        offset: u64,
    ) -> Result<TensorInfo> {
        let elements: usize = config.shape.iter().product();
        let mut bytes_written = 0u64;

        // Generate tensor data based on quantization type
        match config.data_type {
            DataType::F32 => {
                for i in 0..elements {
                    let value = self.generate_f32_value(i, fixture_config.seed);
                    WriteBytesExt::write_f32::<LittleEndian>(writer, value)?;
                    bytes_written += 4;
                }
            }
            DataType::I2S => {
                // I2S quantized data: 2 bits per weight + scale factors
                let block_size = 32;
                let num_blocks = (elements + block_size - 1) / block_size;

                for block in 0..num_blocks {
                    // Write scale factor (FP32)
                    let scale = self.generate_f32_value(block, fixture_config.seed + 1000);
                    WriteBytesExt::write_f32::<LittleEndian>(writer, scale)?;
                    bytes_written += 4;

                    // Write quantized weights (2 bits each, packed)
                    let weights_in_block = std::cmp::min(block_size, elements - block * block_size);
                    let bytes_needed = (weights_in_block + 3) / 4; // 4 weights per byte

                    for byte_idx in 0..bytes_needed {
                        let mut packed_byte = 0u8;
                        for bit_pair in 0..4 {
                            let weight_idx = block * block_size + byte_idx * 4 + bit_pair;
                            if weight_idx < elements {
                                let weight_value = (self
                                    .generate_f32_value(weight_idx, fixture_config.seed + 2000)
                                    * 2.0) as i8;
                                let quantized =
                                    std::cmp::max(-2, std::cmp::min(1, weight_value)) + 1; // Map to 0-3
                                packed_byte |= (quantized as u8) << (bit_pair * 2);
                            }
                        }
                        WriteBytesExt::write_u8(writer, packed_byte)?;
                        bytes_written += 1;
                    }
                }
            }
            DataType::TL1 | DataType::TL2 => {
                // Table lookup quantization: lookup table + indices
                let table_size = if matches!(config.data_type, DataType::TL1) { 16 } else { 256 };

                // Write lookup table
                for i in 0..table_size {
                    let value = self.generate_f32_value(i, fixture_config.seed + 3000);
                    WriteBytesExt::write_f32::<LittleEndian>(writer, value)?;
                    bytes_written += 4;
                }

                // Write indices
                let bits_per_index = if matches!(config.data_type, DataType::TL1) { 4 } else { 8 };

                if bits_per_index == 4 {
                    // TL1: 4 bits per index, pack 2 per byte
                    for i in (0..elements).step_by(2) {
                        let idx1 = (self.generate_f32_value(i, fixture_config.seed + 4000)
                            * table_size as f32) as u8
                            % table_size as u8;
                        let idx2 = if i + 1 < elements {
                            (self.generate_f32_value(i + 1, fixture_config.seed + 4000)
                                * table_size as f32) as u8
                                % table_size as u8
                        } else {
                            0
                        };
                        WriteBytesExt::write_u8(writer, (idx1 & 0xF) | ((idx2 & 0xF) << 4))?;
                        bytes_written += 1;
                    }
                } else {
                    // TL2: 8 bits per index
                    for i in 0..elements {
                        let idx = (self.generate_f32_value(i, fixture_config.seed + 4000)
                            * table_size as f32) as u8
                            % table_size as u8;
                        WriteBytesExt::write_u8(writer, idx)?;
                        bytes_written += 1;
                    }
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported data type for fixture generation: {:?}",
                    config.data_type
                ));
            }
        }

        // Align to tensor_alignment boundary
        let alignment_padding = (fixture_config.tensor_alignment
            - (bytes_written % fixture_config.tensor_alignment))
            % fixture_config.tensor_alignment;
        for _ in 0..alignment_padding {
            WriteBytesExt::write_u8(writer, 0)?;
            bytes_written += 1;
        }

        Ok(TensorInfo {
            name: config.name.clone(),
            shape: config.shape.clone(),
            data_type: config.data_type,
            offset,
            size: bytes_written,
            quantized: !matches!(config.data_type, DataType::F32),
        })
    }

    /// Generate deterministic F32 values for testing
    fn generate_f32_value(&self, index: usize, seed: u64) -> f32 {
        let mut state = seed.wrapping_add(index as u64);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        // Convert to -1.0 to 1.0 range for neural network weights
        ((state as f32) / (u64::MAX as f32)) * 2.0 - 1.0
    }

    /// Get tensor configurations for different model types
    fn get_tensor_configs_for_model(
        &self,
        model_type: &ModelType,
        quant_type: &QuantizationType,
    ) -> Vec<TensorConfig> {
        let data_type = match quant_type {
            QuantizationType::I2S => DataType::I2S,
            QuantizationType::TL1 => DataType::TL1,
            QuantizationType::TL2 => DataType::TL2,
            QuantizationType::IQ2S => DataType::IQ2S,
            QuantizationType::FP32 => DataType::F32,
        };

        match model_type {
            ModelType::Minimal => vec![
                TensorConfig {
                    name: "token_embd.weight".to_string(),
                    shape: vec![1000, 128],
                    data_type,
                },
                TensorConfig {
                    name: "blk.0.attn_norm.weight".to_string(),
                    shape: vec![128],
                    data_type: DataType::F32,
                },
                TensorConfig {
                    name: "blk.0.attn_q.weight".to_string(),
                    shape: vec![128, 128],
                    data_type,
                },
                TensorConfig {
                    name: "blk.0.attn_k.weight".to_string(),
                    shape: vec![128, 128],
                    data_type,
                },
                TensorConfig {
                    name: "blk.0.attn_v.weight".to_string(),
                    shape: vec![128, 128],
                    data_type,
                },
                TensorConfig {
                    name: "blk.0.attn_output.weight".to_string(),
                    shape: vec![128, 128],
                    data_type,
                },
                TensorConfig {
                    name: "blk.0.ffn_norm.weight".to_string(),
                    shape: vec![128],
                    data_type: DataType::F32,
                },
                TensorConfig {
                    name: "blk.0.ffn_gate.weight".to_string(),
                    shape: vec![128, 256],
                    data_type,
                },
                TensorConfig {
                    name: "blk.0.ffn_up.weight".to_string(),
                    shape: vec![128, 256],
                    data_type,
                },
                TensorConfig {
                    name: "blk.0.ffn_down.weight".to_string(),
                    shape: vec![256, 128],
                    data_type,
                },
                TensorConfig {
                    name: "output_norm.weight".to_string(),
                    shape: vec![128],
                    data_type: DataType::F32,
                },
                TensorConfig {
                    name: "output.weight".to_string(),
                    shape: vec![128, 1000],
                    data_type,
                },
            ],
            ModelType::BitNet158_1B => {
                let mut tensors = Vec::new();
                let hidden_size = 2048;
                let vocab_size = 32000;
                let num_layers = 24;

                // Token embeddings
                tensors.push(TensorConfig {
                    name: "token_embd.weight".to_string(),
                    shape: vec![vocab_size, hidden_size],
                    data_type,
                });

                // Transformer layers
                for layer in 0..num_layers {
                    tensors.extend(vec![
                        TensorConfig {
                            name: format!("blk.{}.attn_norm.weight", layer),
                            shape: vec![hidden_size],
                            data_type: DataType::F32,
                        },
                        TensorConfig {
                            name: format!("blk.{}.attn_q.weight", layer),
                            shape: vec![hidden_size, hidden_size],
                            data_type,
                        },
                        TensorConfig {
                            name: format!("blk.{}.attn_k.weight", layer),
                            shape: vec![hidden_size, hidden_size],
                            data_type,
                        },
                        TensorConfig {
                            name: format!("blk.{}.attn_v.weight", layer),
                            shape: vec![hidden_size, hidden_size],
                            data_type,
                        },
                        TensorConfig {
                            name: format!("blk.{}.attn_output.weight", layer),
                            shape: vec![hidden_size, hidden_size],
                            data_type,
                        },
                        TensorConfig {
                            name: format!("blk.{}.ffn_norm.weight", layer),
                            shape: vec![hidden_size],
                            data_type: DataType::F32,
                        },
                        TensorConfig {
                            name: format!("blk.{}.ffn_gate.weight", layer),
                            shape: vec![hidden_size, hidden_size * 4],
                            data_type,
                        },
                        TensorConfig {
                            name: format!("blk.{}.ffn_up.weight", layer),
                            shape: vec![hidden_size, hidden_size * 4],
                            data_type,
                        },
                        TensorConfig {
                            name: format!("blk.{}.ffn_down.weight", layer),
                            shape: vec![hidden_size * 4, hidden_size],
                            data_type,
                        },
                    ]);
                }

                // Output layers
                tensors.extend(vec![
                    TensorConfig {
                        name: "output_norm.weight".to_string(),
                        shape: vec![hidden_size],
                        data_type: DataType::F32,
                    },
                    TensorConfig {
                        name: "output.weight".to_string(),
                        shape: vec![hidden_size, vocab_size],
                        data_type,
                    },
                ]);

                tensors
            }
            _ => {
                // Default to minimal for other model types
                self.get_tensor_configs_for_model(&ModelType::Minimal, quant_type)
            }
        }
    }

    /// Calculate file checksum for validation
    fn calculate_checksum(&self, path: &Path) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let content = std::fs::read(path)?;
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        Ok(format!("{:x}", hasher.finish()))
    }
}

/// Tensor configuration for generation
#[derive(Debug, Clone)]
struct TensorConfig {
    name: String,
    shape: Vec<usize>,
    data_type: DataType,
}

/// Create all BitNet GGUF test fixtures
pub fn create_bitnet_gguf_fixtures(output_dir: PathBuf) -> Result<Vec<GgufFixture>> {
    let generator = GgufFixtureGenerator::new(output_dir, 42); // Deterministic seed
    generator.generate_bitnet_fixture_set()
}

/// Create a specific GGUF fixture for testing
pub fn create_gguf_fixture(output_dir: PathBuf, config: GgufFixtureConfig) -> Result<GgufFixture> {
    let generator = GgufFixtureGenerator::new(output_dir, config.seed);
    generator.generate_fixture(&config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_minimal_gguf_generation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = GgufFixtureConfig {
            name: "test_minimal".to_string(),
            model_type: ModelType::Minimal,
            quantization_type: QuantizationType::I2S,
            vocab_size: 100,
            hidden_size: 32,
            num_layers: 1,
            tensor_alignment: 32,
            generate_invalid: false,
            seed: 42,
        };

        let fixture = create_gguf_fixture(temp_dir.path().to_path_buf(), config)?;

        // Validate fixture was created
        assert!(fixture.path.exists());
        assert!(fixture.file_size > 0);
        assert!(!fixture.tensors.is_empty());

        Ok(())
    }

    #[test]
    fn test_invalid_gguf_generation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let config = GgufFixtureConfig {
            name: "test_invalid".to_string(),
            model_type: ModelType::Minimal,
            quantization_type: QuantizationType::I2S,
            vocab_size: 100,
            hidden_size: 32,
            num_layers: 1,
            tensor_alignment: 32,
            generate_invalid: true,
            seed: 42,
        };

        let fixture = create_gguf_fixture(temp_dir.path().to_path_buf(), config)?;

        // Validate invalid fixture characteristics
        assert!(fixture.path.exists());
        assert!(fixture.file_size > 0);

        Ok(())
    }
}
