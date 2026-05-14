//! Model inspection commands for diagnostics and debugging

use anyhow::{Context, Result};
use bitnet_common::{BitNetConfig, BitNetError};
use bitnet_models::formats::gguf::{GgufReader, GgufTensorType};
use bitnet_models::names::{is_layernorm_weight, is_projection_weight};
use candle_core::{DType, Tensor};
use clap::Args;
use memmap2::Mmap;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::PathBuf;
use tracing::debug;

use crate::ln_rules::{Ruleset, detect_rules, load_policy};

/// Inspect command arguments
#[derive(Args)]
pub struct InspectCommand {
    /// Model file path
    #[arg(value_name = "MODEL", required_unless_present = "model")]
    pub model_positional: Option<PathBuf>,

    /// Model file path
    #[arg(long = "model", value_name = "MODEL", conflicts_with = "model_positional")]
    pub model: Option<PathBuf>,

    /// Compute and display LayerNorm gamma statistics
    #[arg(long)]
    pub ln_stats: bool,

    /// Emit a QK256 layout report for real GGUF tensors
    #[arg(long)]
    pub qk256_layout_report: bool,

    /// Emit effective runtime contract metadata for the GGUF model
    #[arg(long)]
    pub runtime_contract_report: bool,

    /// Explicit tokenizer path to compare against GGUF tokenizer metadata
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

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

    /// Write JSON report to this path
    #[arg(long)]
    pub json_out: Option<PathBuf>,
}

impl InspectCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.qk256_layout_report {
            self.write_qk256_layout_report().await
        } else if self.runtime_contract_report {
            self.write_runtime_contract_report().await
        } else if self.ln_stats {
            self.check_ln_gamma_stats().await
        } else {
            anyhow::bail!(
                "No inspection mode specified. Use --ln-stats, --qk256-layout-report, or --runtime-contract-report."
            );
        }
    }

    fn model_path(&self) -> Result<&PathBuf> {
        self.model
            .as_ref()
            .or(self.model_positional.as_ref())
            .ok_or_else(|| anyhow::anyhow!("model path is required"))
    }

    async fn write_qk256_layout_report(&self) -> Result<()> {
        let report = self.qk256_layout_report()?;
        let json = serde_json::to_string_pretty(&report)?;
        self.write_json(json)?;
        Ok(())
    }

    fn write_json(&self, json: String) -> Result<()> {
        if let Some(path) = &self.json_out {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            std::fs::write(path, format!("{json}\n"))
                .with_context(|| format!("failed to write {}", path.display()))?;
        } else {
            println!("{json}");
        }
        Ok(())
    }

    fn qk256_layout_report(&self) -> Result<Qk256LayoutReport> {
        let model_path = self.model_path()?;
        let file = File::open(model_path)
            .with_context(|| format!("Failed to open model: {}", model_path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut hasher = Sha256::new();
        hasher.update(&mmap);
        let model_sha256 = format!("{:x}", hasher.finalize());

        let reader = GgufReader::new(&mmap)?;
        let mut tensors = Vec::new();

        for i in 0..reader.tensor_count() as usize {
            let info = reader.get_tensor_info(i)?;
            if info.tensor_type != GgufTensorType::I2_S || info.shape.len() != 2 {
                continue;
            }

            let gguf_cols = info.shape[0];
            let gguf_rows = info.shape[1];
            let nelems = gguf_rows.checked_mul(gguf_cols).ok_or_else(|| {
                anyhow::anyhow!("QK256 tensor '{}' element count overflow", info.name)
            })?;
            let row_stride_bytes = qk256_logical_packed_bytes(gguf_cols);
            let logical_packed_bytes =
                gguf_rows.checked_mul(row_stride_bytes).ok_or_else(|| {
                    anyhow::anyhow!("QK256 tensor '{}' byte count overflow", info.name)
                })?;
            let data = reader.get_tensor_data_by_info(info)?;
            if data.len() < logical_packed_bytes {
                continue;
            }

            let qk_bytes = &data[..logical_packed_bytes];
            let trailer_scale = if data.len() >= logical_packed_bytes + 4 {
                Some(f32::from_le_bytes([
                    data[logical_packed_bytes],
                    data[logical_packed_bytes + 1],
                    data[logical_packed_bytes + 2],
                    data[logical_packed_bytes + 3],
                ]))
            } else {
                None
            };
            let trailer_scale_bytes = if trailer_scale.is_some() { 4 } else { 0 };
            let padding_bytes =
                data.len().saturating_sub(logical_packed_bytes + trailer_scale_bytes);

            let hist = qk256_code_histogram_act_parallel_rows(
                qk_bytes,
                gguf_rows,
                gguf_cols,
                row_stride_bytes,
            );
            let code_3_frequency = if nelems == 0 { 0.0 } else { hist[3] as f64 / nelems as f64 };

            let kernel_shape_from_gguf_dims = [gguf_rows, gguf_cols];
            let first_row_bytes = &qk_bytes[..row_stride_bytes.min(qk_bytes.len())];
            let sample_cols = gguf_cols.min(256);
            let act_sample = unpack_act_parallel_codes(first_row_bytes, sample_cols);
            let contiguous_sample = unpack_contiguous_codes(first_row_bytes, sample_cols);

            tensors.push(Qk256TensorLayoutReport {
                name: info.name.clone(),
                gguf_shape: info.shape.clone(),
                kernel_shape_from_gguf_dims,
                tensor_type: format!("{:?}", info.tensor_type),
                nelems,
                row_stride_bytes,
                logical_packed_bytes,
                actual_bytes: data.len(),
                trailer_scale,
                trailer_scale_bytes,
                padding_bytes,
                packing_mode_detected: if code_3_frequency == 0.0 {
                    "qk256_act_parallel_128_ternary_like".to_string()
                } else {
                    "qk256_act_parallel_128_code3_present".to_string()
                },
                code_histogram: hist,
                code_3_count: hist[3],
                code_3_frequency,
                first_row_sample_len: sample_cols,
                first_row_act_parallel_hash: sha256_hex_bytes(&act_sample),
                first_row_contiguous_hash: sha256_hex_bytes(&contiguous_sample),
                first_row_hashes_match: act_sample == contiguous_sample,
                first_row_act_parallel_codes_32: act_sample.iter().take(32).copied().collect(),
                first_row_contiguous_codes_32: contiguous_sample.iter().take(32).copied().collect(),
            });
        }

        Ok(Qk256LayoutReport {
            schema_version: 1,
            diagnostic: "bitnet_qk256_layout_report".to_string(),
            diagnostic_only: true,
            promotion_allowed: false,
            proof_receipts_written: false,
            manifest_updated: false,
            model: model_path.display().to_string(),
            model_sha256,
            tensor_count: tensors.len(),
            not_claims: critical_qk256_report_not_claims()
                .into_iter()
                .map(str::to_string)
                .collect(),
            tensors,
        })
    }

    async fn write_runtime_contract_report(&self) -> Result<()> {
        let report = self.runtime_contract_report()?;
        self.write_json(serde_json::to_string_pretty(&report)?)?;
        Ok(())
    }

    fn runtime_contract_report(&self) -> Result<RuntimeContractReport> {
        let model_path = self.model_path()?;
        let file = File::open(model_path)
            .with_context(|| format!("Failed to open model: {}", model_path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut hasher = Sha256::new();
        hasher.update(&mmap);
        let model_sha256 = format!("{:x}", hasher.finalize());

        let reader = GgufReader::new(&mmap)?;
        let architecture = reader.get_string_metadata("general.architecture");
        let file_type = reader.get_u32_metadata("general.file_type");
        let tokenizer_model = reader.get_string_metadata("tokenizer.ggml.model");
        let tokenizer_tokens_len =
            reader.get_string_array_metadata("tokenizer.ggml.tokens").map(|tokens| tokens.len());

        let mut config = BitNetConfig::default();
        if let Some(architecture) = &architecture {
            config.model.apply_architecture_defaults(architecture);
        }
        fill_runtime_contract_config(&reader, &mut config);

        let gguf_special_tokens = RuntimeSpecialTokens {
            bos_token_id: u32_any(
                &reader,
                &[
                    "bitnet-b1.58.tokenizer.bos_token_id",
                    "llama.tokenizer.bos_token_id",
                    "tokenizer.ggml.bos_token_id",
                    "general.bos_token_id",
                ],
            ),
            eos_token_id: u32_any(
                &reader,
                &[
                    "bitnet-b1.58.tokenizer.eos_token_id",
                    "llama.tokenizer.eos_token_id",
                    "tokenizer.ggml.eos_token_id",
                    "general.eos_token_id",
                ],
            ),
            eot_token_id: reader.get_u32_metadata("tokenizer.ggml.eot_token_id"),
            pad_token_id: u32_any(
                &reader,
                &[
                    "bitnet-b1.58.tokenizer.pad_token_id",
                    "llama.tokenizer.pad_token_id",
                    "tokenizer.ggml.padding_token_id",
                    "general.pad_token_id",
                ],
            ),
        };

        let external_tokenizer =
            self.tokenizer.as_ref().map(|path| external_tokenizer_contract(path)).transpose()?;

        let tokenizer_agreement = external_tokenizer
            .as_ref()
            .map(|external| tokenizer_agreement(&gguf_special_tokens, external));

        Ok(RuntimeContractReport {
            schema_version: 1,
            diagnostic: "bitnet_runtime_contract_report".to_string(),
            diagnostic_only: true,
            promotion_allowed: false,
            proof_receipts_written: false,
            manifest_updated: false,
            model: model_path.display().to_string(),
            model_sha256,
            gguf_metadata: RuntimeGgufMetadata {
                architecture,
                file_type,
                tokenizer_model,
                tokenizer_tokens_len,
                llama_vocab_size: reader.get_u32_metadata("llama.vocab_size"),
                bitnet_vocab_size: reader.get_u32_metadata("bitnet-b1.58.vocab_size"),
                context_length: u32_any(
                    &reader,
                    &["bitnet-b1.58.context_length", "llama.context_length"],
                ),
            },
            effective_config: RuntimeEffectiveConfig {
                norm_type: format!("{:?}", config.model.norm_type),
                activation_type: format!("{:?}", config.model.activation_type),
                vocab_size: config.model.vocab_size,
                hidden_size: config.model.hidden_size,
                num_layers: config.model.num_layers,
                num_heads: config.model.num_heads,
                num_key_value_heads: config.model.num_key_value_heads,
                intermediate_size: config.model.intermediate_size,
                max_position_embeddings: config.model.max_position_embeddings,
                rope_theta: config.model.rope_theta,
                rms_norm_eps: config.model.rms_norm_eps,
                add_bos: config.inference.add_bos,
                append_eos: config.inference.append_eos,
                mask_pad: config.inference.mask_pad,
            },
            gguf_special_tokens,
            external_tokenizer,
            tokenizer_agreement,
            not_claims: critical_qk256_report_not_claims()
                .into_iter()
                .chain([
                    "runtime_reference_parity",
                    "semantic_quality",
                    "tokenizer_template_authority",
                ])
                .map(str::to_string)
                .collect(),
        })
    }

    /// Check LayerNorm gamma statistics with architecture-aware validation
    async fn check_ln_gamma_stats(&self) -> Result<()> {
        // Open once, mmap once, hash from slice
        let model_path = self.model_path()?;
        let file = File::open(model_path)
            .with_context(|| format!("Failed to open model: {}", model_path.display()))?;
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

#[derive(Debug, Serialize)]
struct Qk256LayoutReport {
    schema_version: u32,
    diagnostic: String,
    diagnostic_only: bool,
    promotion_allowed: bool,
    proof_receipts_written: bool,
    manifest_updated: bool,
    model: String,
    model_sha256: String,
    tensor_count: usize,
    not_claims: Vec<String>,
    tensors: Vec<Qk256TensorLayoutReport>,
}

#[derive(Debug, Serialize)]
struct Qk256TensorLayoutReport {
    name: String,
    gguf_shape: Vec<usize>,
    kernel_shape_from_gguf_dims: [usize; 2],
    tensor_type: String,
    nelems: usize,
    row_stride_bytes: usize,
    logical_packed_bytes: usize,
    actual_bytes: usize,
    trailer_scale: Option<f32>,
    trailer_scale_bytes: usize,
    padding_bytes: usize,
    packing_mode_detected: String,
    code_histogram: [usize; 4],
    code_3_count: usize,
    code_3_frequency: f64,
    first_row_sample_len: usize,
    first_row_act_parallel_hash: String,
    first_row_contiguous_hash: String,
    first_row_hashes_match: bool,
    first_row_act_parallel_codes_32: Vec<u8>,
    first_row_contiguous_codes_32: Vec<u8>,
}

#[derive(Debug, Serialize)]
struct RuntimeContractReport {
    schema_version: u32,
    diagnostic: String,
    diagnostic_only: bool,
    promotion_allowed: bool,
    proof_receipts_written: bool,
    manifest_updated: bool,
    model: String,
    model_sha256: String,
    gguf_metadata: RuntimeGgufMetadata,
    effective_config: RuntimeEffectiveConfig,
    gguf_special_tokens: RuntimeSpecialTokens,
    external_tokenizer: Option<RuntimeExternalTokenizer>,
    tokenizer_agreement: Option<RuntimeTokenizerAgreement>,
    not_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RuntimeGgufMetadata {
    architecture: Option<String>,
    file_type: Option<u32>,
    tokenizer_model: Option<String>,
    tokenizer_tokens_len: Option<usize>,
    llama_vocab_size: Option<u32>,
    bitnet_vocab_size: Option<u32>,
    context_length: Option<u32>,
}

#[derive(Debug, Serialize)]
struct RuntimeEffectiveConfig {
    norm_type: String,
    activation_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    rope_theta: Option<f32>,
    rms_norm_eps: Option<f32>,
    add_bos: bool,
    append_eos: bool,
    mask_pad: bool,
}

#[derive(Debug, Serialize)]
struct RuntimeSpecialTokens {
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    eot_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

#[derive(Debug, Serialize)]
struct RuntimeExternalTokenizer {
    path: String,
    sha256: String,
    vocab_size: usize,
    real_vocab_size: usize,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    begin_of_text_id: Option<u32>,
    end_of_text_id: Option<u32>,
    eot_id: Option<u32>,
    start_header_id: Option<u32>,
    end_header_id: Option<u32>,
}

#[derive(Debug, Serialize)]
struct RuntimeTokenizerAgreement {
    gguf_bos_matches_tokenizer_bos: Option<bool>,
    gguf_eos_matches_tokenizer_eos: Option<bool>,
    gguf_pad_matches_tokenizer_pad: Option<bool>,
    gguf_bos_matches_begin_of_text: Option<bool>,
    gguf_eos_matches_end_of_text: Option<bool>,
    gguf_eos_matches_eot: Option<bool>,
    gguf_eot_matches_tokenizer_eot: Option<bool>,
    all_checked_specials_match: bool,
    checked_count: usize,
    mismatch_count: usize,
}

fn fill_runtime_contract_config(reader: &GgufReader, config: &mut BitNetConfig) {
    if let Some(vocab_size) = reader
        .get_string_array_metadata("tokenizer.ggml.tokens")
        .map(|tokens| tokens.len() as u32)
        .or_else(|| u32_any(reader, &["llama.vocab_size", "bitnet-b1.58.vocab_size"]))
    {
        config.model.vocab_size = vocab_size as usize;
    }

    if let Some(num_layers) =
        u32_any(reader, &["llama.block_count", "bitnet-b1.58.block_count", "n_layer"])
    {
        config.model.num_layers = num_layers as usize;
    }

    if let Some(hidden_size) = u32_any(
        reader,
        &["llama.embedding_length", "bitnet-b1.58.embedding_length", "n_embd", "hidden_size"],
    ) {
        config.model.hidden_size = hidden_size as usize;
    }

    if let Some(num_heads) = u32_any(
        reader,
        &[
            "llama.attention.head_count",
            "bitnet-b1.58.attention.head_count",
            "n_head",
            "attn.n_heads",
            "num_attention_heads",
        ],
    ) {
        config.model.num_heads = num_heads as usize;
    }

    config.model.num_key_value_heads = u32_any(
        reader,
        &[
            "llama.attention.head_count_kv",
            "bitnet-b1.58.attention.head_count_kv",
            "n_head_kv",
            "n_kv_heads",
            "attn.n_kv_heads",
            "attn_n_kv_heads",
            "num_key_value_heads",
        ],
    )
    .map(|v| v as usize)
    .unwrap_or(0);
    if config.model.num_key_value_heads == 0 {
        config.model.num_key_value_heads = config.model.num_heads;
    }

    if let Some(intermediate_size) =
        u32_any(reader, &["llama.feed_forward_length", "bitnet-b1.58.feed_forward_length", "n_ff"])
    {
        config.model.intermediate_size = intermediate_size as usize;
    }

    if let Some(context_length) =
        u32_any(reader, &["llama.context_length", "bitnet-b1.58.context_length"])
    {
        config.model.max_position_embeddings = context_length as usize;
    }

    config.model.rope_theta =
        f32_any(reader, &["bitnet-b1.58.rope.freq_base", "llama.rope.freq_base", "rope.freq_base"]);

    config.model.rms_norm_eps = f32_any(
        reader,
        &[
            "bitnet-b1.58.attention.layer_norm_rms_epsilon",
            "llama.attention.layer_norm_rms_epsilon",
            "llama.attention.layer_norm_epsilon",
            "general.layer_norm_epsilon",
        ],
    );

    if let Some(bos) = u32_any(
        reader,
        &[
            "bitnet-b1.58.tokenizer.bos_token_id",
            "llama.tokenizer.bos_token_id",
            "tokenizer.ggml.bos_token_id",
            "general.bos_token_id",
        ],
    ) {
        config.model.tokenizer.bos_id = Some(bos as i32);
    }

    if let Some(eos) = u32_any(
        reader,
        &[
            "bitnet-b1.58.tokenizer.eos_token_id",
            "llama.tokenizer.eos_token_id",
            "tokenizer.ggml.eos_token_id",
            "general.eos_token_id",
        ],
    ) {
        config.model.tokenizer.eos_id = Some(eos as i32);
    }

    if let Some(pad) = u32_any(
        reader,
        &[
            "bitnet-b1.58.tokenizer.padding_token_id",
            "llama.tokenizer.padding_token_id",
            "tokenizer.ggml.padding_token_id",
            "general.padding_token_id",
        ],
    ) {
        config.model.tokenizer.pad_id = Some(pad as i32);
    }

    if let Some(add_bos) = bool_any(
        reader,
        &[
            "bitnet-b1.58.tokenizer.add_bos",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_bos",
            "general.add_bos",
        ],
    ) {
        config.inference.add_bos = add_bos;
    }

    if let Some(append_eos) = bool_any(
        reader,
        &[
            "bitnet-b1.58.tokenizer.append_eos",
            "tokenizer.ggml.add_eos_token",
            "tokenizer.ggml.append_eos",
            "general.append_eos",
        ],
    ) {
        config.inference.append_eos = append_eos;
    }

    if let Some(mask_pad) = bool_any(
        reader,
        &["bitnet-b1.58.tokenizer.mask_pad", "tokenizer.ggml.mask_pad", "general.mask_pad"],
    ) {
        config.inference.mask_pad = mask_pad;
    }
}

fn u32_any(reader: &GgufReader, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|key| {
        reader
            .get_u32_metadata(key)
            .or_else(|| reader.get_i32_metadata(key).and_then(|v| (v >= 0).then_some(v as u32)))
    })
}

fn f32_any(reader: &GgufReader, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| reader.get_f32_metadata(key))
}

fn bool_any(reader: &GgufReader, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| reader.get_bool_metadata(key))
}

fn external_tokenizer_contract(path: &PathBuf) -> Result<RuntimeExternalTokenizer> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read tokenizer {}", path.display()))?;
    let sha256 = sha256_hex_bytes(&bytes);
    let tokenizer = bitnet_tokenizers::load_tokenizer(path)
        .with_context(|| format!("failed to load tokenizer {}", path.display()))?;

    Ok(RuntimeExternalTokenizer {
        path: path.display().to_string(),
        sha256,
        vocab_size: tokenizer.vocab_size(),
        real_vocab_size: tokenizer.real_vocab_size(),
        bos_token_id: tokenizer.bos_token_id(),
        eos_token_id: tokenizer.eos_token_id(),
        pad_token_id: tokenizer.pad_token_id(),
        begin_of_text_id: tokenizer.token_to_id("<|begin_of_text|>"),
        end_of_text_id: tokenizer.token_to_id("<|end_of_text|>"),
        eot_id: tokenizer.token_to_id("<|eot_id|>"),
        start_header_id: tokenizer.token_to_id("<|start_header_id|>"),
        end_header_id: tokenizer.token_to_id("<|end_header_id|>"),
    })
}

fn tokenizer_agreement(
    gguf: &RuntimeSpecialTokens,
    external: &RuntimeExternalTokenizer,
) -> RuntimeTokenizerAgreement {
    let checks = [
        eq_if_both(gguf.bos_token_id, external.bos_token_id),
        eq_if_both(gguf.eos_token_id, external.eos_token_id),
        eq_if_both(gguf.pad_token_id, external.pad_token_id),
        eq_if_both(gguf.bos_token_id, external.begin_of_text_id),
        eq_if_both(gguf.eos_token_id, external.end_of_text_id),
        eq_if_both(gguf.eos_token_id, external.eot_id),
        eq_if_both(gguf.eot_token_id, external.eot_id),
    ];
    let checked_count = checks.iter().filter(|check| check.is_some()).count();
    let mismatch_count = checks.iter().filter(|check| **check == Some(false)).count();

    RuntimeTokenizerAgreement {
        gguf_bos_matches_tokenizer_bos: checks[0],
        gguf_eos_matches_tokenizer_eos: checks[1],
        gguf_pad_matches_tokenizer_pad: checks[2],
        gguf_bos_matches_begin_of_text: checks[3],
        gguf_eos_matches_end_of_text: checks[4],
        gguf_eos_matches_eot: checks[5],
        gguf_eot_matches_tokenizer_eot: checks[6],
        all_checked_specials_match: mismatch_count == 0,
        checked_count,
        mismatch_count,
    }
}

fn eq_if_both(left: Option<u32>, right: Option<u32>) -> Option<bool> {
    Some(left? == right?)
}

fn qk256_logical_packed_bytes(nelems: usize) -> usize {
    nelems.div_ceil(256) * 64
}

fn qk256_act_parallel_code_at(qk_bytes: &[u8], col: usize) -> u8 {
    let group128 = col / 128;
    let within = col % 128;
    let lane = within / 32;
    let pos = within % 32;
    let byte = qk_bytes[group128 * 32 + pos];
    (byte >> (6 - lane * 2)) & 0x03
}

fn qk256_contiguous_code_at(qk_bytes: &[u8], col: usize) -> u8 {
    let byte = qk_bytes[col / 4];
    (byte >> ((col % 4) * 2)) & 0x03
}

fn qk256_code_histogram_act_parallel_rows(
    qk_bytes: &[u8],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> [usize; 4] {
    let mut hist = [0usize; 4];
    for row in 0..rows {
        let start = row * row_stride_bytes;
        let end = start + row_stride_bytes;
        let row_bytes = &qk_bytes[start..end];
        for col in 0..cols {
            let code = qk256_act_parallel_code_at(row_bytes, col);
            hist[code as usize] += 1;
        }
    }
    hist
}

fn unpack_act_parallel_codes(qk_bytes: &[u8], cols: usize) -> Vec<u8> {
    (0..cols).map(|col| qk256_act_parallel_code_at(qk_bytes, col)).collect()
}

fn unpack_contiguous_codes(qk_bytes: &[u8], cols: usize) -> Vec<u8> {
    (0..cols).map(|col| qk256_contiguous_code_at(qk_bytes, col)).collect()
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn critical_qk256_report_not_claims() -> Vec<&'static str> {
    vec![
        "selected_attention_residency",
        "resident_kv_decode",
        "attention_scores_residency",
        "softmax_residency",
        "attention_value_mix_residency",
        "full_support_op_residency",
        "full_device_residency",
        "completion",
    ]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act_parallel_code_at_decodes_128_value_lane_layout() {
        let mut bytes = vec![0u8; 32];
        bytes[0] = (0 << 6) | (1 << 4) | (2 << 2) | 3;
        bytes[31] = (3 << 6) | (2 << 4) | (1 << 2);

        assert_eq!(qk256_act_parallel_code_at(&bytes, 0), 0);
        assert_eq!(qk256_act_parallel_code_at(&bytes, 32), 1);
        assert_eq!(qk256_act_parallel_code_at(&bytes, 64), 2);
        assert_eq!(qk256_act_parallel_code_at(&bytes, 96), 3);
        assert_eq!(qk256_act_parallel_code_at(&bytes, 31), 3);
        assert_eq!(qk256_act_parallel_code_at(&bytes, 63), 2);
        assert_eq!(qk256_act_parallel_code_at(&bytes, 95), 1);
        assert_eq!(qk256_act_parallel_code_at(&bytes, 127), 0);
    }

    #[test]
    fn contiguous_and_act_parallel_samples_differ_for_lane_packed_bytes() {
        let mut bytes = vec![0u8; 64];
        bytes[0] = (0 << 6) | (1 << 4) | (2 << 2) | 3;

        let act = unpack_act_parallel_codes(&bytes, 128);
        let contiguous = unpack_contiguous_codes(&bytes, 128);

        assert_ne!(act, contiguous);
        assert_eq!(&act[..4], &[0, 0, 0, 0]);
        assert_eq!(act[32], 1);
        assert_eq!(act[64], 2);
        assert_eq!(act[96], 3);
        assert_eq!(&contiguous[..4], &[3, 2, 1, 0]);
    }
}
