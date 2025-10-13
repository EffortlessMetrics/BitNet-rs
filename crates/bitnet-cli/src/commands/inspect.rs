//! Model inspection commands for diagnostics and debugging

use anyhow::{Context, Result};
use bitnet_common::BitNetError;
use bitnet_models::formats::gguf::{GgufReader, GgufTensorType};
use bitnet_models::names::{is_layernorm_weight, is_projection_weight};
use candle_core::{DType, Tensor};
use clap::Args;
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::PathBuf;
use tracing::debug;

use crate::ln_rules::{Ruleset, detect_rules, load_policy};

/// Inspect command arguments
#[derive(Args)]
pub struct InspectCommand {
    /// Model file path
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Compute and display LayerNorm gamma statistics
    #[arg(long)]
    pub ln_stats: bool,

    /// Gate behavior: none|auto|policy
    #[arg(long, default_value = "auto")]
    pub gate: String,

    /// Policy file (YAML) for custom validation rules
    #[arg(long)]
    pub policy: Option<PathBuf>,

    /// Policy key (architecture ID) for rules lookup
    #[arg(long)]
    pub policy_key: Option<String>,

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

    /// Check LayerNorm gamma statistics with architecture-aware validation
    async fn check_ln_gamma_stats(&self) -> Result<()> {
        // Open once, mmap once, hash from slice
        let file = File::open(&self.model)
            .with_context(|| format!("Failed to open model: {}", self.model.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Compute SHA256 from mmap
        let mut hasher = Sha256::new();
        hasher.update(&mmap);
        let hash = hasher.finalize();
        let model_sha256 = format!("{:x}", hash);

        // Create reader from existing mmap
        let reader = GgufReader::new(&mmap)?;

        // 1) Select validation rules based on gate mode
        let arch = reader.get_string_metadata("general.architecture").unwrap_or_else(|| {
            debug!("'general.architecture' metadata not found, using 'unknown'");
            "unknown".to_string()
        });
        let arch = arch.as_str();
        debug!("Architecture: {}", arch);
        let file_type = reader.get_u32_metadata("general.file_type").unwrap_or(0);
        debug!("File type: {}", file_type);

        // Compute strict_mode once (DRY)
        let strict_mode = std::env::var("BITNET_STRICT_MODE")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);

        // Gate selection with explicit validation
        let rules: Ruleset = match self.gate.as_str() {
            "none" => crate::ln_rules::rules_generic(),
            "auto" => detect_rules(arch, file_type),
            "policy" => {
                let pol = self
                    .policy
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--policy required for gate=policy"))?;
                let key = self.policy_key.as_deref().unwrap_or(arch);
                load_policy(pol, key)?
            }
            other => {
                return Err(anyhow::anyhow!(
                    "Invalid gate mode '{}'. Must be one of: none, auto, policy.",
                    other
                ));
            }
        };

        tracing::info!(
            "LN gate ruleset: {} (architecture: {}, file_type: {})",
            rules.name,
            arch,
            file_type
        );

        let tensor_count = reader.tensor_count() as usize;
        debug!("Inspecting {} tensors for LayerNorm gamma statistics", tensor_count);

        let mut ln_stats = Vec::new();
        let mut ln_bad_count = 0;
        let mut ln_total_count = 0;

        let mut proj_stats = Vec::new();
        let mut proj_bad_count = 0;
        let mut proj_total_count = 0;

        // 2) Single-pass scan: route to LayerNorm or Projection validation
        for i in 0..tensor_count {
            let info = reader.get_tensor_info(i)?;

            // Route by tensor type
            if is_layernorm_weight(&info.name) {
                debug!("Processing LayerNorm tensor: {} (type: {:?})", info.name, info.tensor_type);
                ln_total_count += 1;

                // Load tensor data and compute RMS
                let tensor_data = reader.get_tensor_data(i)?;
                let tensor = Self::decode_tensor(
                    &info.name,
                    &info.shape,
                    info.tensor_type,
                    tensor_data,
                    TensorKind::LayerNorm,
                )?;

                let rms = Self::compute_rms(&tensor)?;
                let is_ok = rules.check_ln(&info.name, rms);

                if !is_ok {
                    ln_bad_count += 1;
                }

                ln_stats.push(TensorStat {
                    name: info.name.clone(),
                    rms,
                    is_ok,
                    kind: TensorKind::LayerNorm,
                });
            } else if is_projection_weight(&info.name) {
                // Only validate RMS for float tensors (F32/F16)
                // Quantized projection weights (I2_S, etc.) are expected and don't need RMS validation
                if !matches!(info.tensor_type, GgufTensorType::F32 | GgufTensorType::F16) {
                    debug!(
                        "Skipping RMS validation for quantized projection tensor: {} (type: {:?})",
                        info.name, info.tensor_type
                    );
                    continue;
                }

                proj_total_count += 1;

                // Load tensor data and compute RMS
                let tensor_data = reader.get_tensor_data(i)?;
                let tensor = Self::decode_tensor(
                    &info.name,
                    &info.shape,
                    info.tensor_type,
                    tensor_data,
                    TensorKind::Projection,
                )?;

                let rms = Self::compute_rms(&tensor)?;
                let is_ok = rules.check_proj_rms(rms);

                if !is_ok {
                    proj_bad_count += 1;
                }

                proj_stats.push(TensorStat {
                    name: info.name.clone(),
                    rms,
                    is_ok,
                    kind: TensorKind::Projection,
                });
            }
        }

        // Combine stats for output
        let mut all_stats = ln_stats;
        all_stats.extend(proj_stats);

        // Output results
        if self.json {
            self.output_json(
                &model_sha256,
                &all_stats,
                ln_bad_count,
                ln_total_count,
                proj_bad_count,
                proj_total_count,
                &rules.name,
                strict_mode,
            )?;
        } else {
            self.output_text(
                &model_sha256,
                &all_stats,
                ln_bad_count,
                ln_total_count,
                proj_bad_count,
                proj_total_count,
                &rules.name,
                strict_mode,
            )?;
        }

        // Determine exit code based on strict mode
        let total_bad = ln_bad_count + proj_bad_count;

        if total_bad > 0 && strict_mode {
            std::process::exit(crate::exit::EXIT_LN_SUSPICIOUS);
        }

        Ok(())
    }

    /// Decode tensor from raw bytes
    ///
    /// # Arguments
    /// * `name` - Tensor name for error messages
    /// * `shape` - Tensor shape
    /// * `tensor_type` - GGUF tensor type
    /// * `data` - Raw tensor data
    /// * `tensor_kind` - What kind of tensor this is (for error messages)
    fn decode_tensor(
        name: &str,
        shape: &[usize],
        tensor_type: GgufTensorType,
        data: &[u8],
        tensor_kind: TensorKind,
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
                let kind_str = match tensor_kind {
                    TensorKind::LayerNorm => "LayerNorm",
                    TensorKind::Projection => "Projection",
                };
                return Err(anyhow::anyhow!(
                    "{} tensor '{}' has quantized type {:?}, expected float (F32/F16) for RMS validation",
                    kind_str,
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
    #[allow(clippy::too_many_arguments)]
    fn output_json(
        &self,
        model_sha256: &str,
        stats: &[TensorStat],
        ln_bad_count: usize,
        ln_total_count: usize,
        proj_bad_count: usize,
        proj_total_count: usize,
        ruleset_name: &str,
        strict_mode: bool,
    ) -> Result<()> {
        use serde_json::json;

        let tensors: Vec<_> = stats
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "kind": match s.kind {
                        TensorKind::LayerNorm => "layernorm",
                        TensorKind::Projection => "projection",
                    },
                    "rms": format!("{:.4}", s.rms),
                    "status": if s.is_ok { "ok" } else { "suspicious" }
                })
            })
            .collect();

        let total_bad = ln_bad_count + proj_bad_count;

        let output = json!({
            "model_sha256": model_sha256,
            "ruleset": ruleset_name,
            "layernorm": {
                "total": ln_total_count,
                "suspicious": ln_bad_count,
            },
            "projection": {
                "total": proj_total_count,
                "suspicious": proj_bad_count,
            },
            "strict_mode": strict_mode,
            "tensors": tensors,
            "status": if total_bad > 0 {
                if strict_mode { "failed" } else { "warning" }
            } else {
                "ok"
            }
        });

        println!("{}", serde_json::to_string_pretty(&output)?);
        Ok(())
    }

    /// Output results as human-readable text
    #[allow(clippy::too_many_arguments)]
    fn output_text(
        &self,
        model_sha256: &str,
        stats: &[TensorStat],
        ln_bad_count: usize,
        ln_total_count: usize,
        proj_bad_count: usize,
        proj_total_count: usize,
        ruleset_name: &str,
        strict_mode: bool,
    ) -> Result<()> {
        println!("model_sha256: {}", model_sha256);
        println!("ruleset: {}", ruleset_name);
        println!();

        for stat in stats {
            let status_icon = if stat.is_ok { "✅" } else { "❌" };
            let kind_str = match stat.kind {
                TensorKind::LayerNorm => "[LN]",
                TensorKind::Projection => "[PROJ]",
            };
            println!(
                "{:<64} {:<8} rms={:<8} {}",
                stat.name,
                kind_str,
                format!("{:.4}", stat.rms),
                status_icon
            );
        }

        println!();

        let total_bad = ln_bad_count + proj_bad_count;

        if ln_bad_count > 0 {
            if strict_mode {
                println!(
                    "❌ LN RMS gate failed: {}/{} out of envelope ({})",
                    ln_bad_count, ln_total_count, ruleset_name
                );
            } else {
                println!(
                    "⚠️  WARNING: suspicious LayerNorm gamma detected ({}/{} layers)",
                    ln_bad_count, ln_total_count
                );
            }
        } else if ln_total_count > 0 {
            println!("✅ LN RMS gate passed ({})", ruleset_name);
        }

        if proj_bad_count > 0 {
            if strict_mode {
                println!(
                    "❌ Projection RMS gate failed: {}/{} out of envelope ({})",
                    proj_bad_count, proj_total_count, ruleset_name
                );
            } else {
                println!(
                    "⚠️  WARNING: suspicious projection weights detected ({}/{} tensors)",
                    proj_bad_count, proj_total_count
                );
            }
        } else if proj_total_count > 0 {
            println!("✅ Projection RMS gate passed ({})", ruleset_name);
        }

        if total_bad > 0 && strict_mode {
            println!();
            println!("❌ STRICT MODE: Validation failed");
        }

        Ok(())
    }
}

/// Tensor statistics for validation
#[derive(Debug)]
struct TensorStat {
    name: String,
    rms: f32,
    is_ok: bool,
    kind: TensorKind,
}

/// Type of tensor being validated
#[derive(Debug, Clone, Copy)]
enum TensorKind {
    LayerNorm,
    Projection,
}
