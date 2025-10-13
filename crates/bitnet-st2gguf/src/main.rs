//! SafeTensors to GGUF Converter
//!
//! This binary converts SafeTensors model checkpoints to GGUF format with
//! LayerNorm preservation. It ensures that LayerNorm tensors are always
//! stored as F16 (never quantized) to maintain numerical stability.
//!
//! # Usage
//!
//! ## Installed binary
//!
//! ```bash
//! # Convert a SafeTensors file
//! st2gguf --input model.safetensors --output model.gguf
//!
//! # Convert from a directory (auto-discovers *.safetensors and config.json)
//! st2gguf --input model_dir/ --output model.gguf
//!
//! # Specify config and tokenizer explicitly
//! st2gguf --input model.safetensors --output model.gguf \
//!         --config config.json --tokenizer tokenizer.json
//!
//! # Enable strict validation (fail if LN tensors aren't F16)
//! st2gguf --input model.safetensors --output model.gguf --strict
//! ```
//!
//! ## Via cargo
//!
//! ```bash
//! cargo run -p bitnet-st2gguf -- --input model.safetensors --output model.gguf
//! ```
//!
//! # Features
//!
//! - **LayerNorm Preservation**: Automatically detects and enforces F16 format for LayerNorm tensors
//! - **Auto-Discovery**: Finds SafeTensors and config.json files in directories
//! - **Metadata Extraction**: Extracts model configuration from config.json
//! - **Strict Validation**: Optional strict mode to validate LayerNorm preservation
//! - **Sidecar Metadata**: Generates .meta.json file with conversion details

mod layernorm;
mod writer;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use half::f16;
use safetensors::SafeTensors;
use serde_json::{Value as Json, json};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use writer::{GgufWriter, MetadataValue, TensorDType, TensorEntry};

/// Minimal set we require in strict mode
const REQUIRED_KEYS: &[&str] = &[
    "general.architecture",
    "bitnet.hidden_size",
    "bitnet.num_layers",
    "bitnet.num_heads",
    "bitnet.vocab_size",
    "bitnet.context_length",
    "general.file_type",
];

/// SafeTensors to GGUF converter with LayerNorm preservation
#[derive(Parser, Debug)]
#[command(name = "st2gguf")]
#[command(about = "Convert SafeTensors models to GGUF format with LayerNorm preservation")]
#[command(version)]
struct Args {
    /// Input SafeTensors file or directory
    ///
    /// If a directory is provided, will auto-discover the first *.safetensors file
    #[arg(short, long)]
    input: PathBuf,

    /// Output GGUF file path
    #[arg(short, long)]
    output: PathBuf,

    /// Optional config.json path
    ///
    /// If not specified, will try to find config.json in the input directory
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Optional tokenizer path (for parity, not embedded in GGUF)
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,

    /// Enable strict validation
    ///
    /// Fails if LayerNorm tensors are not F16 after conversion
    #[arg(short, long)]
    strict: bool,
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    // Resolve input path
    let (safetensors_path, input_dir) = resolve_input(&args.input)?;

    // Load config.json if available
    let config = if let Some(config_path) = args.config {
        load_config(&config_path)?
    } else if let Some(dir) = &input_dir {
        load_config(&dir.join("config.json"))?
    } else {
        None
    };

    // Log configuration
    tracing::info!("Input: {}", safetensors_path.display());
    tracing::info!("Output: {}", args.output.display());
    if let Some(cfg) = &config {
        tracing::info!("Config: loaded {} keys", cfg.as_object().map(|o| o.len()).unwrap_or(0));
    }
    if args.strict {
        tracing::info!("Strict mode: enabled");
    }

    // Convert SafeTensors to GGUF
    let conversion_result =
        convert_safetensors_to_gguf(&safetensors_path, &args.output, config.as_ref(), args.strict)?;

    // Write sidecar metadata
    write_sidecar_metadata(&args.output, &safetensors_path, &conversion_result)?;

    tracing::info!("Conversion complete!");
    tracing::info!(
        "  Tensors: {} (LayerNorm: {})",
        conversion_result.total_tensors,
        conversion_result.layernorm_tensors_enforced
    );
    tracing::info!("  Output: {}", args.output.display());
    tracing::info!("  Metadata: {}.meta.json", args.output.display());

    Ok(())
}

/// Resolve input path to SafeTensors file and optional directory
fn resolve_input(input: &Path) -> Result<(PathBuf, Option<PathBuf>)> {
    if input.is_file() {
        // Input is a file - use directly
        Ok((input.to_path_buf(), input.parent().map(|p| p.to_path_buf())))
    } else if input.is_dir() {
        // Input is a directory - find first *.safetensors
        let safetensors_path = find_safetensors(input)?;
        Ok((safetensors_path, Some(input.to_path_buf())))
    } else {
        bail!("Input path does not exist: {}", input.display())
    }
}

/// Find the first *.safetensors file in a directory
fn find_safetensors(dir: &Path) -> Result<PathBuf> {
    let entries = fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && let Some(ext) = path.extension()
            && ext == "safetensors"
        {
            return Ok(path);
        }
    }

    bail!("No *.safetensors file found in directory: {}", dir.display())
}

/// Load config.json if it exists
fn load_config(path: &Path) -> Result<Option<Json>> {
    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;

    let config: Json = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse config: {}", path.display()))?;

    Ok(Some(config))
}

/// Conversion result metadata
struct ConversionResult {
    total_tensors: usize,
    layernorm_tensors_enforced: usize,
    metadata_entries: usize,
}

/// Validate that required metadata is present (strict mode)
fn validate_required_metadata_strict(metadata: &[(String, MetadataValue)]) -> Result<()> {
    // Build a set of metadata keys for efficient lookup
    let keys: HashMap<&str, ()> = metadata.iter().map(|(k, _)| (k.as_str(), ())).collect();

    // Check each required key
    for &required in REQUIRED_KEYS {
        ensure!(
            keys.contains_key(required),
            "strict mode: missing required metadata key `{}`",
            required
        );
    }

    Ok(())
}

/// Convert SafeTensors to GGUF format
fn convert_safetensors_to_gguf(
    input_path: &Path,
    output_path: &Path,
    config: Option<&Json>,
    strict: bool,
) -> Result<ConversionResult> {
    // Load SafeTensors file
    tracing::info!("Loading SafeTensors file: {}", input_path.display());
    let mut file = fs::File::open(input_path)
        .with_context(|| format!("Failed to open SafeTensors file: {}", input_path.display()))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .with_context(|| format!("Failed to read SafeTensors file: {}", input_path.display()))?;

    let safetensors =
        SafeTensors::deserialize(&buffer).context("Failed to deserialize SafeTensors")?;

    // Create GGUF writer
    let mut writer = GgufWriter::new();

    // Extract and add metadata from config.json
    let metadata_count = if let Some(cfg) = config {
        add_metadata_from_config(&mut writer, cfg)?
    } else {
        tracing::warn!("No config.json found - using minimal metadata");
        add_minimal_metadata(&mut writer);
        3 // bitnet.quantization_type, bitnet.weight_scale, general.file_type
    };

    // Strict mode: validate required metadata before proceeding
    if strict {
        validate_required_metadata_strict(&writer.metadata)
            .context("strict metadata validation failed")?;
        tracing::info!("Strict metadata validation passed");
    }

    // Convert tensors
    tracing::info!("Converting tensors...");
    let tensor_names: Vec<&str> = safetensors.names();
    let total_tensors = tensor_names.len();

    // Count LayerNorm tensors
    let layernorm_count = layernorm::count_layernorm_tensors(tensor_names.iter().copied());
    tracing::info!(
        "Found {} LayerNorm tensors out of {} total tensors",
        layernorm_count,
        total_tensors
    );

    let mut layernorm_enforced = 0;

    for name in &tensor_names {
        let view =
            safetensors.tensor(name).with_context(|| format!("Failed to get tensor: {}", name))?;

        let is_layernorm = layernorm::is_layernorm_tensor(name);

        if is_layernorm {
            layernorm_enforced += 1;
            tracing::debug!("LayerNorm tensor: {} (enforcing F16)", name);
        }

        // Convert to F16 (always F16, never F32 or quantized)
        let f16_data = convert_to_f16(&view)
            .with_context(|| format!("Failed to convert tensor to F16: {}", name))?;

        // Strict validation for LayerNorm tensors
        if strict && is_layernorm {
            validate_f16_conversion(&f16_data, name)?;
        }

        // Pack F16 as bytes
        let data_bytes = bytemuck::cast_slice(&f16_data).to_vec();

        // Convert shape from usize to u64
        let shape: Vec<u64> = view.shape().iter().map(|&s| s as u64).collect();

        // Create tensor entry
        let tensor = TensorEntry::new(name.to_string(), shape, TensorDType::F16, data_bytes);

        writer.add_tensor(tensor);
    }

    // Write GGUF file
    tracing::info!("Writing GGUF file: {}", output_path.display());
    writer.write_to_file(output_path).context("Failed to write GGUF file")?;

    Ok(ConversionResult {
        total_tensors,
        layernorm_tensors_enforced: layernorm_enforced,
        metadata_entries: metadata_count,
    })
}

/// Add metadata from config.json
fn add_metadata_from_config(writer: &mut GgufWriter, config: &Json) -> Result<usize> {
    let mut count = 0;

    // Extract standard metadata
    if let Some(obj) = config.as_object() {
        // Model architecture
        if let Some(model_type) = obj.get("model_type").and_then(|v| v.as_str()) {
            writer.add_metadata(
                "general.architecture",
                MetadataValue::String(model_type.to_string()),
            );
            count += 1;
        }

        // Hidden size
        if let Some(hidden_size) = obj.get("hidden_size").and_then(|v| v.as_u64()) {
            writer.add_metadata("bitnet.hidden_size", MetadataValue::U32(hidden_size as u32));
            count += 1;
        }

        // Number of layers
        if let Some(num_layers) = obj.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            writer.add_metadata("bitnet.num_layers", MetadataValue::U32(num_layers as u32));
            count += 1;
        }

        // Number of attention heads
        if let Some(num_heads) = obj.get("num_attention_heads").and_then(|v| v.as_u64()) {
            writer.add_metadata("bitnet.num_heads", MetadataValue::U32(num_heads as u32));
            count += 1;
        }

        // Vocabulary size
        if let Some(vocab_size) = obj.get("vocab_size").and_then(|v| v.as_u64()) {
            writer.add_metadata("bitnet.vocab_size", MetadataValue::U32(vocab_size as u32));
            count += 1;
        }

        // Context length
        if let Some(max_pos) = obj.get("max_position_embeddings").and_then(|v| v.as_u64()) {
            writer.add_metadata("bitnet.context_length", MetadataValue::U32(max_pos as u32));
            count += 1;
        }
    }

    // Add BitNet-specific metadata
    add_minimal_metadata(writer);
    count += 3;

    Ok(count)
}

/// Add minimal BitNet metadata (always included)
fn add_minimal_metadata(writer: &mut GgufWriter) {
    // Quantization type (f16 for clean GGUF)
    writer.add_metadata("bitnet.quantization_type", MetadataValue::String("f16".to_string()));

    // Weight scale (1.0 for unscaled F16)
    writer.add_metadata("bitnet.weight_scale", MetadataValue::F32(1.0));

    // File type (1 = F16)
    writer.add_metadata("general.file_type", MetadataValue::U32(1));
}

/// Convert SafeTensors tensor to F16
fn convert_to_f16(view: &safetensors::tensor::TensorView) -> Result<Vec<f16>> {
    use safetensors::Dtype;

    let data = view.data();
    let dtype = view.dtype();

    match dtype {
        Dtype::F32 => {
            // F32 -> F16 conversion
            // Handle potential alignment issues by reading bytes manually
            let mut result = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                let f32_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                result.push(f16::from_f32(f32_val));
            }
            Ok(result)
        }
        Dtype::F16 => {
            // F16 -> F16 (passthrough)
            // Handle potential alignment issues by reading bytes manually
            let mut result = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                result.push(f16::from_le_bytes([chunk[0], chunk[1]]));
            }
            Ok(result)
        }
        Dtype::BF16 => {
            // BF16 -> F16 conversion (via F32)
            let mut result = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let bf16_val = half::bf16::from_le_bytes([chunk[0], chunk[1]]);
                result.push(f16::from_f32(bf16_val.to_f32()));
            }
            Ok(result)
        }
        Dtype::I8 => {
            // I8 -> F16 cast
            let i8_data = bytemuck::cast_slice::<u8, i8>(data);
            Ok(i8_data.iter().map(|&i| f16::from_f32(i as f32)).collect())
        }
        Dtype::I16 => {
            // I16 -> F16 cast
            let mut result = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let i16_val = i16::from_le_bytes([chunk[0], chunk[1]]);
                result.push(f16::from_f32(i16_val as f32));
            }
            Ok(result)
        }
        Dtype::I32 => {
            // I32 -> F16 cast
            let mut result = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                let i32_val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                result.push(f16::from_f32(i32_val as f32));
            }
            Ok(result)
        }
        Dtype::U8 => {
            // U8 -> F16 cast
            Ok(data.iter().map(|&u| f16::from_f32(u as f32)).collect())
        }
        _ => {
            bail!("Unsupported dtype for conversion: {:?}", dtype)
        }
    }
}

/// Validate that F16 conversion preserved LayerNorm properties
fn validate_f16_conversion(data: &[f16], name: &str) -> Result<()> {
    // Check for NaN or Inf values
    let has_invalid = data.iter().any(|&x| {
        let f = x.to_f32();
        f.is_nan() || f.is_infinite()
    });

    if has_invalid {
        bail!(
            "Strict validation failed for LayerNorm tensor '{}': contains NaN or Inf after F16 conversion",
            name
        );
    }

    // Check for reasonable range (LayerNorm gammas typically around 1.0, betas around 0.0)
    let values: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
    let max_abs = values.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

    // Warn if values are suspiciously large (might indicate quantization artifacts)
    if max_abs > 100.0 {
        tracing::warn!(
            "LayerNorm tensor '{}' has large absolute values (max: {:.2}) - may indicate quantization artifacts",
            name,
            max_abs
        );
    }

    Ok(())
}

/// Write sidecar metadata file
fn write_sidecar_metadata(
    output_path: &Path,
    source_path: &Path,
    result: &ConversionResult,
) -> Result<()> {
    let metadata = json!({
        "source": source_path.display().to_string(),
        "format": "gguf",
        "version": 3,
        "tensors": result.total_tensors,
        "metadata_entries": result.metadata_entries,
        "layernorm_tensors_enforced": result.layernorm_tensors_enforced,
        "conversion_tool": "bitnet-st2gguf",
        "conversion_version": env!("CARGO_PKG_VERSION"),
    });

    let meta_path = output_path.with_extension("gguf.meta.json");
    let meta_json =
        serde_json::to_string_pretty(&metadata).context("Failed to serialize metadata")?;

    fs::write(&meta_path, meta_json)
        .with_context(|| format!("Failed to write metadata file: {}", meta_path.display()))?;

    tracing::debug!("Wrote sidecar metadata: {}", meta_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_find_safetensors_in_directory() {
        let temp_dir = TempDir::new().unwrap();
        let safetensors_path = temp_dir.path().join("model.safetensors");
        fs::write(&safetensors_path, b"dummy").unwrap();

        let found = find_safetensors(temp_dir.path()).unwrap();
        assert_eq!(found, safetensors_path);
    }

    #[test]
    fn test_find_safetensors_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let result = find_safetensors(temp_dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_config_missing() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let result = load_config(&config_path).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_load_config_valid() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        fs::write(&config_path, r#"{"model_type": "bitnet"}"#).unwrap();

        let result = load_config(&config_path).unwrap();
        assert!(result.is_some());
        let config = result.unwrap();
        assert_eq!(config["model_type"], "bitnet");
    }

    #[test]
    fn test_convert_to_f16_from_f32() {
        // Create a mock F32 tensor view
        let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        // Create a minimal SafeTensors buffer with proper alignment
        // SafeTensors format: header_size (u64 le) | header_json | tensor_data
        // The data_offsets in JSON are byte offsets from the start of the data section
        let json = r#"{"test":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let header_size = json.len() as u64;

        let mut buffer = Vec::new();
        buffer.extend_from_slice(&header_size.to_le_bytes());
        buffer.extend_from_slice(json.as_bytes());

        // Convert f32 data to bytes manually to avoid alignment issues
        for &val in &f32_data {
            buffer.extend_from_slice(&val.to_le_bytes());
        }

        let st = SafeTensors::deserialize(&buffer).unwrap();
        let view = st.tensor("test").unwrap();

        let result = convert_to_f16(&view).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].to_f32(), 1.0);
        assert_eq!(result[3].to_f32(), 4.0);
    }

    #[test]
    fn test_validate_f16_conversion_valid() {
        let data = vec![f16::from_f32(1.0), f16::from_f32(0.5), f16::from_f32(2.0)];

        let result = validate_f16_conversion(&data, "test.weight");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_f16_conversion_invalid() {
        let data = vec![f16::from_f32(1.0), f16::from_f32(f32::NAN), f16::from_f32(2.0)];

        let result = validate_f16_conversion(&data, "test.weight");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_input_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("model.safetensors");
        fs::write(&file_path, b"dummy").unwrap();

        let (resolved, dir) = resolve_input(&file_path).unwrap();
        assert_eq!(resolved, file_path);
        assert!(dir.is_some());
    }

    #[test]
    fn test_resolve_input_directory() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("model.safetensors");
        fs::write(&file_path, b"dummy").unwrap();

        let (resolved, dir) = resolve_input(temp_dir.path()).unwrap();
        assert_eq!(resolved, file_path);
        assert_eq!(dir.unwrap(), temp_dir.path());
    }
}
