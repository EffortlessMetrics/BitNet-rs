//! Model inspection commands for diagnostics and debugging

use anyhow::{Context, Result};
use bitnet_common::BitNetError;
use bitnet_models::formats::gguf::{GgufReader, GgufTensorType};
use candle_core::{DType, Tensor};
use clap::Args;
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::PathBuf;
use tracing::debug;

/// Inspect command arguments
#[derive(Args)]
pub struct InspectCommand {
    /// Model file path
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Compute and display LayerNorm gamma statistics
    #[arg(long)]
    pub ln_stats: bool,

    /// Output format as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

impl InspectCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.ln_stats {
            self.check_ln_gamma_stats().await
        } else {
            anyhow::bail!(
                "No inspection mode specified. Use --ln-stats to check LayerNorm gamma statistics."
            );
        }
    }

    /// Check if a tensor name is a LayerNorm weight
    fn is_layernorm_weight(name: &str) -> bool {
        // LLaMA/HF-style names
        name.ends_with(".attention_norm.weight")
            || name.ends_with(".ffn_norm.weight")
            || name.ends_with(".input_layernorm.weight")
            || name.ends_with(".post_attention_layernorm.weight")
            // Microsoft BitNet-style names
            || name.ends_with(".attn_norm.weight")
            || name.ends_with(".ffn_norm.weight")
            // Root-level norms
            || name.ends_with(".final_norm.weight")
            || name == "final_norm.weight"
            // Generic catch-all (last, most permissive)
            || name.ends_with(".norm.weight")
    }

    /// Check LayerNorm gamma statistics
    async fn check_ln_gamma_stats(&self) -> Result<()> {
        // Compute model SHA256
        let file = File::open(&self.model)
            .with_context(|| format!("Failed to open model: {}", self.model.display()))?;

        let mut hasher = Sha256::new();
        std::io::copy(&mut std::io::BufReader::new(&file), &mut hasher)?;
        let hash = hasher.finalize();
        let model_sha256 = format!("{:x}", hash);

        // Memory-map the file for tensor reading
        let file = File::open(&self.model)
            .with_context(|| format!("Failed to open model: {}", self.model.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let reader = GgufReader::new(&mmap)?;

        let tensor_count = reader.tensor_count() as usize;
        debug!("Inspecting {} tensors for LayerNorm gamma statistics", tensor_count);

        let mut ln_stats = Vec::new();
        let mut suspicious_count = 0;
        let mut total_ln_count = 0;

        // Scan all tensors for LayerNorm weights
        for i in 0..tensor_count {
            let info = reader.get_tensor_info(i)?;

            // Check if this is a LayerNorm gamma tensor
            if !Self::is_layernorm_weight(&info.name) {
                continue;
            }

            total_ln_count += 1;

            // Load tensor data and compute RMS
            let tensor_data = reader.get_tensor_data(i)?;
            let tensor =
                Self::decode_tensor(&info.name, &info.shape, info.tensor_type, tensor_data)?;

            // Compute RMS
            let rms = Self::compute_rms(&tensor)?;

            // Check if RMS is in acceptable envelope [0.5, 2.0]
            let is_ok = (0.5..=2.0).contains(&rms) && rms.is_finite();

            if !is_ok {
                suspicious_count += 1;
            }

            ln_stats.push(LayerNormStat { name: info.name.clone(), rms, is_ok });
        }

        // Output results
        if self.json {
            self.output_json(&model_sha256, &ln_stats, suspicious_count, total_ln_count)?;
        } else {
            self.output_text(&model_sha256, &ln_stats, suspicious_count, total_ln_count)?;
        }

        // Determine exit code based on strict mode
        let strict_mode = std::env::var("BITNET_STRICT_MODE")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        if suspicious_count > 0 && strict_mode {
            std::process::exit(crate::exit::EXIT_LN_SUSPICIOUS);
        }

        Ok(())
    }

    /// Decode tensor from raw bytes
    fn decode_tensor(
        name: &str,
        shape: &[usize],
        tensor_type: GgufTensorType,
        data: &[u8],
    ) -> Result<Tensor> {
        let tensor = match tensor_type {
            GgufTensorType::F32 => {
                let float_data = bytemuck::cast_slice::<u8, f32>(data);
                Tensor::from_slice(float_data, shape, &candle_core::Device::Cpu)
                    .map_err(|e| anyhow::anyhow!("Failed to create F32 tensor '{}': {}", name, e))?
            }
            GgufTensorType::F16 => {
                let half_data = bytemuck::cast_slice::<u8, u16>(data);
                let float_data: Vec<f32> =
                    half_data.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
                Tensor::from_slice(&float_data, shape, &candle_core::Device::Cpu)
                    .map_err(|e| anyhow::anyhow!("Failed to create F16 tensor '{}': {}", name, e))?
            }
            _ => {
                // For quantized types, we need to dequantize first
                return Err(anyhow::anyhow!(
                    "LayerNorm tensor '{}' has quantized type {:?}, expected float (F32/F16)",
                    name,
                    tensor_type
                ));
            }
        };

        Ok(tensor)
    }

    /// Compute RMS (root mean square) of a tensor
    fn compute_rms(tensor: &Tensor) -> Result<f32> {
        // Convert to F32 for reliable statistics
        let t32 =
            tensor.to_dtype(DType::F32).map_err(|e| BitNetError::Validation(e.to_string()))?;

        let mean_sq = t32
            .sqr()
            .map_err(|e| BitNetError::Validation(e.to_string()))?
            .mean_all()
            .map_err(|e| BitNetError::Validation(e.to_string()))?
            .to_scalar::<f32>()
            .map_err(|e| BitNetError::Validation(e.to_string()))?;

        Ok(mean_sq.sqrt())
    }

    /// Output results as JSON
    fn output_json(
        &self,
        model_sha256: &str,
        stats: &[LayerNormStat],
        suspicious_count: usize,
        total_count: usize,
    ) -> Result<()> {
        use serde_json::json;

        let strict_mode = std::env::var("BITNET_STRICT_MODE")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        let layers: Vec<_> = stats
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "rms": format!("{:.4}", s.rms),
                    "status": if s.is_ok { "ok" } else { "suspicious" }
                })
            })
            .collect();

        let output = json!({
            "model_sha256": model_sha256,
            "total_ln_layers": total_count,
            "suspicious_count": suspicious_count,
            "strict_mode": strict_mode,
            "layers": layers,
            "status": if suspicious_count > 0 {
                if strict_mode { "failed" } else { "warning" }
            } else {
                "ok"
            }
        });

        println!("{}", serde_json::to_string_pretty(&output)?);
        Ok(())
    }

    /// Output results as human-readable text
    fn output_text(
        &self,
        model_sha256: &str,
        stats: &[LayerNormStat],
        suspicious_count: usize,
        total_count: usize,
    ) -> Result<()> {
        println!("model_sha256: {}", model_sha256);
        println!();

        for stat in stats {
            let status_icon = if stat.is_ok { "✅" } else { "❌" };
            println!("{:<40} rms={:<8} {}", stat.name, format!("{:.4}", stat.rms), status_icon);
        }

        println!();

        let strict_mode = std::env::var("BITNET_STRICT_MODE")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        if suspicious_count > 0 {
            if strict_mode {
                println!(
                    "STRICT: suspicious LayerNorm gamma detected ({}/{} layers)",
                    suspicious_count, total_count
                );
            } else {
                println!(
                    "WARNING: suspicious LayerNorm gamma detected ({}/{} layers)",
                    suspicious_count, total_count
                );
            }
        } else {
            println!("All LayerNorm gamma tensors within acceptable envelope [0.5, 2.0]");
        }

        Ok(())
    }
}

/// LayerNorm statistics for a single layer
#[derive(Debug)]
struct LayerNormStat {
    name: String,
    rms: f32,
    is_ok: bool,
}
