//! GGUF model metadata and parity validation utilities

use anyhow::{Result, bail};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

/// GGUF metadata that we validate for parity
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub tokenizer_type: Option<String>,
    pub model_type: String,
}

/// Read GGUF metadata for parity validation
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read GGUF magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("Not a GGUF file: invalid magic bytes");
    }

    // Read version
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != 3 {
        bail!("Unsupported GGUF version: {} (expected 3)", version);
    }

    // Read tensor count and metadata count
    let mut tensor_count_bytes = [0u8; 8];
    reader.read_exact(&mut tensor_count_bytes)?;
    let _tensor_count = u64::from_le_bytes(tensor_count_bytes);

    let mut metadata_count_bytes = [0u8; 8];
    reader.read_exact(&mut metadata_count_bytes)?;
    let metadata_count = u64::from_le_bytes(metadata_count_bytes);

    // Read metadata key-value pairs
    let mut metadata = HashMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut reader)?;
        let value = read_metadata_value(&mut reader)?;
        metadata.insert(key, value);
    }

    // Extract required fields
    let vocab_size = metadata
        .get("llama.vocab_size")
        .or_else(|| metadata.get("tokenizer.ggml.tokens"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50257); // GPT-2 default

    let hidden_size = metadata
        .get("llama.embedding_length")
        .or_else(|| metadata.get("llama.hidden_size"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(768);

    let num_layers = metadata
        .get("llama.block_count")
        .or_else(|| metadata.get("llama.layer_count"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);

    let num_heads = metadata
        .get("llama.attention.head_count")
        .or_else(|| metadata.get("llama.head_count"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);

    let tokenizer_type = metadata
        .get("tokenizer.ggml.model")
        .or_else(|| metadata.get("general.architecture"))
        .cloned();

    let model_type =
        metadata.get("general.architecture").cloned().unwrap_or_else(|| "bitnet".to_string());

    Ok(GgufMetadata { vocab_size, hidden_size, num_layers, num_heads, tokenizer_type, model_type })
}

/// Validate vocab size against expected value
pub fn validate_vocab_size(metadata: &GgufMetadata, expected: usize) -> Result<()> {
    if metadata.vocab_size != expected {
        bail!("Vocab size mismatch: model has {} but expected {}", metadata.vocab_size, expected);
    }
    Ok(())
}

/// Validate model compatibility
pub fn validate_model_compatibility(metadata: &GgufMetadata) -> Result<()> {
    // Check if it's a BitNet model
    if !metadata.model_type.contains("bitnet") && !metadata.model_type.contains("llama") {
        bail!("Incompatible model type: '{}' (expected bitnet or llama)", metadata.model_type);
    }

    // Validate reasonable dimensions
    if metadata.vocab_size < 100 || metadata.vocab_size > 500000 {
        bail!("Unreasonable vocab size: {}", metadata.vocab_size);
    }

    if metadata.hidden_size < 64 || metadata.hidden_size > 16384 {
        bail!("Unreasonable hidden size: {}", metadata.hidden_size);
    }

    if metadata.num_layers < 1 || metadata.num_layers > 200 {
        bail!("Unreasonable layer count: {}", metadata.num_layers);
    }

    Ok(())
}

// Helper functions for reading GGUF format

fn read_string(reader: &mut BufReader<File>) -> Result<String> {
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes) as usize;

    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;

    String::from_utf8(bytes).map_err(Into::into)
}

fn read_metadata_value(reader: &mut BufReader<File>) -> Result<String> {
    let mut type_bytes = [0u8; 4];
    reader.read_exact(&mut type_bytes)?;
    let value_type = u32::from_le_bytes(type_bytes);

    match value_type {
        4 => {
            // UINT32
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            Ok(u32::from_le_bytes(bytes).to_string())
        }
        5 => {
            // INT32
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            Ok(i32::from_le_bytes(bytes).to_string())
        }
        6 => {
            // FLOAT32
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            Ok(f32::from_le_bytes(bytes).to_string())
        }
        7 => {
            // BOOL
            let mut byte = [0u8; 1];
            reader.read_exact(&mut byte)?;
            Ok((byte[0] != 0).to_string())
        }
        8 => {
            // STRING
            read_string(reader)
        }
        9 => {
            // ARRAY
            // For now, just return a placeholder
            skip_array(reader)?;
            Ok("[array]".to_string())
        }
        10 => {
            // UINT64
            let mut bytes = [0u8; 8];
            reader.read_exact(&mut bytes)?;
            Ok(u64::from_le_bytes(bytes).to_string())
        }
        11 => {
            // INT64
            let mut bytes = [0u8; 8];
            reader.read_exact(&mut bytes)?;
            Ok(i64::from_le_bytes(bytes).to_string())
        }
        12 => {
            // FLOAT64
            let mut bytes = [0u8; 8];
            reader.read_exact(&mut bytes)?;
            Ok(f64::from_le_bytes(bytes).to_string())
        }
        _ => bail!("Unknown metadata value type: {}", value_type),
    }
}

fn skip_array(reader: &mut BufReader<File>) -> Result<()> {
    // Read array type and length
    let mut type_bytes = [0u8; 4];
    reader.read_exact(&mut type_bytes)?;
    let elem_type = u32::from_le_bytes(type_bytes);

    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes) as usize;

    // Skip array elements based on type
    let elem_size = match elem_type {
        4..=6 => 4,   // UINT32, INT32, FLOAT32
        7 => 1,       // BOOL
        10..=12 => 8, // UINT64, INT64, FLOAT64
        8 => {
            // STRING array - need to read each string
            for _ in 0..len {
                read_string(reader)?;
            }
            return Ok(());
        }
        _ => bail!("Unknown array element type: {}", elem_type),
    };

    // Skip the array data
    let skip_bytes = len * elem_size;
    reader.seek(std::io::SeekFrom::Current(skip_bytes as i64))?;

    Ok(())
}
