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

    /// Emit an I2_S/QK256 matmul contract report against the reference policy
    #[arg(long)]
    pub i2s_matmul_contract_report: bool,

    /// Emit a RoPE tensor/factor contract report for real GGUF tensors
    #[arg(long)]
    pub rope_contract_report: bool,

    /// Emit effective runtime contract metadata for the GGUF model
    #[arg(long)]
    pub runtime_contract_report: bool,

    /// Emit embedding/final-norm/logits tensor contract metadata for the GGUF model
    #[arg(long)]
    pub logits_contract_report: bool,

    /// Token IDs to probe in the logits/embedding contract report
    #[arg(long = "logits-token-id", value_name = "TOKEN_ID")]
    pub logits_token_ids: Vec<u32>,

    /// Emit BitNet-b1.58 graph-order/subnorm contract metadata for the GGUF model
    #[arg(long)]
    pub bitnet_graph_contract_report: bool,

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
        } else if self.i2s_matmul_contract_report {
            self.write_i2s_matmul_contract_report().await
        } else if self.rope_contract_report {
            self.write_rope_contract_report().await
        } else if self.runtime_contract_report {
            self.write_runtime_contract_report().await
        } else if self.logits_contract_report {
            self.write_logits_contract_report().await
        } else if self.bitnet_graph_contract_report {
            self.write_bitnet_graph_contract_report().await
        } else if self.ln_stats {
            self.check_ln_gamma_stats().await
        } else {
            anyhow::bail!(
                "No inspection mode specified. Use --ln-stats, --qk256-layout-report, --i2s-matmul-contract-report, --rope-contract-report, --runtime-contract-report, --logits-contract-report, or --bitnet-graph-contract-report."
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

    async fn write_i2s_matmul_contract_report(&self) -> Result<()> {
        let report = self.i2s_matmul_contract_report()?;
        self.write_json(serde_json::to_string_pretty(&report)?)?;
        Ok(())
    }

    fn i2s_matmul_contract_report(&self) -> Result<I2sMatmulContractReport> {
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

        let qk256_inventory = qk256_contract_inventory(&reader)?;
        let rust_uses_reference_activation_quantization = true;
        let summary = i2s_matmul_contract_summary(
            &qk256_inventory,
            rust_uses_reference_activation_quantization,
        );

        Ok(I2sMatmulContractReport {
            schema_version: 1,
            diagnostic: "bitnet_i2s_matmul_contract_report".to_string(),
            diagnostic_only: true,
            promotion_allowed: false,
            proof_receipts_written: false,
            manifest_updated: false,
            model: model_path.display().to_string(),
            model_sha256,
            gguf_metadata: I2sMatmulGgufMetadata { architecture, file_type },
            qk256_inventory,
            rust_policy: I2sRustMatmulPolicy {
                activation_quantization: "quantize_row_i8_s".to_string(),
                dot_formula: "(integer_dot - act_sum) / act_scale * trailer_scale".to_string(),
                code3_value: 2.0,
                uses_reference_activation_quantization: rust_uses_reference_activation_quantization,
            },
            reference_policy: I2sReferenceMatmulPolicy {
                source: "ggml.c I2_S matmul + quantize_row_i8_s + ggml_vec_dot_i2_i8_s".to_string(),
                activation_quantization: "quantize_row_i8_s".to_string(),
                correction_formula: "(integer_dot - act_sum) / act_scale * trailer_scale"
                    .to_string(),
                matmul_effective_code3_value: 2.0,
                dequantize_row_i2_s_code3_value: 0.0,
            },
            summary,
            not_claims: critical_qk256_report_not_claims()
                .into_iter()
                .chain(["runtime_reference_parity", "semantic_quality", "a770_semantic_quality"])
                .map(str::to_string)
                .collect(),
        })
    }

    async fn write_rope_contract_report(&self) -> Result<()> {
        let report = self.rope_contract_report()?;
        self.write_json(serde_json::to_string_pretty(&report)?)?;
        Ok(())
    }

    fn rope_contract_report(&self) -> Result<RopeContractReport> {
        let model_path = self.model_path()?;
        let file = File::open(model_path)
            .with_context(|| format!("Failed to open model: {}", model_path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut hasher = Sha256::new();
        hasher.update(&mmap);
        let model_sha256 = format!("{:x}", hasher.finalize());

        let reader = GgufReader::new(&mmap)?;
        let architecture = reader.get_string_metadata("general.architecture");

        let mut config = BitNetConfig::default();
        if let Some(architecture) = &architecture {
            config.model.apply_architecture_defaults(architecture);
        }
        fill_runtime_contract_config(&reader, &mut config);

        let mut rope_freqs = Vec::new();
        for i in 0..reader.tensor_count() as usize {
            let info = reader.get_tensor_info(i)?;
            if !is_rope_freqs_tensor_name(&info.name) {
                continue;
            }

            let data = reader.get_tensor_data_by_info(info)?;
            let values = decode_rope_freq_values(&info.name, info.tensor_type, data)?;
            let stats = rope_value_stats(&values);

            rope_freqs.push(RopeFreqsTensorReport {
                name: info.name.clone(),
                shape: info.shape.clone(),
                tensor_type: format!("{:?}", info.tensor_type),
                element_count: values.len(),
                actual_bytes: data.len(),
                raw_sha256: sha256_hex_bytes(data),
                sample_len: values.len().min(8),
                sample_values_first_8: values.iter().take(8).copied().collect(),
                min: stats.min,
                max: stats.max,
                mean: stats.mean,
            });
        }

        let rust_uses_gguf_rope_freqs = false;
        let summary = rope_contract_summary(rope_freqs.len(), rust_uses_gguf_rope_freqs);
        let head_dim = if config.model.num_heads == 0 {
            None
        } else {
            Some(config.model.hidden_size / config.model.num_heads)
        };

        Ok(RopeContractReport {
            schema_version: 1,
            diagnostic: "bitnet_rope_contract_report".to_string(),
            diagnostic_only: true,
            promotion_allowed: false,
            proof_receipts_written: false,
            manifest_updated: false,
            model: model_path.display().to_string(),
            model_sha256,
            gguf_metadata: RopeGgufMetadata {
                architecture,
                context_length: u32_any(
                    &reader,
                    &["bitnet-b1.58.context_length", "llama.context_length"],
                ),
                rope_freq_base: f32_any(
                    &reader,
                    &["bitnet-b1.58.rope.freq_base", "llama.rope.freq_base", "rope.freq_base"],
                ),
            },
            effective_rust_policy: RopeRustPolicy {
                policy: "base_theta_sincos_tables_without_gguf_rope_freqs".to_string(),
                uses_gguf_rope_freqs: rust_uses_gguf_rope_freqs,
                rope_layout: "neox_offset_by_half".to_string(),
                rope_layout_source: "RotaryEmbedding::apply split-half pairing".to_string(),
                ggml_rope_type_id: 2,
                rope_theta: config.model.rope_theta,
                num_heads: config.model.num_heads,
                num_key_value_heads: config.model.num_key_value_heads,
                head_dim,
                max_position_embeddings: config.model.max_position_embeddings,
            },
            reference_policy: RopeReferencePolicy {
                source: "llama.cpp build_bitnet_158/build_rope_factors/ggml_rope_ext".to_string(),
                expects_optional_rope_freqs_tensor: true,
                expected_tensor_suffix: "rope_freqs.weight".to_string(),
                rope_type: "GGML_ROPE_TYPE_NEOX".to_string(),
                rope_type_id: 2,
                rope_layout: "neox_offset_by_half".to_string(),
                rope_layout_source: "llama_rope_type(LLM_ARCH_BITNET_B158)".to_string(),
            },
            summary,
            rope_freqs,
            not_claims: critical_qk256_report_not_claims()
                .into_iter()
                .chain([
                    "runtime_reference_parity",
                    "semantic_quality",
                    "rope_factor_implementation",
                    "a770_semantic_quality",
                ])
                .map(str::to_string)
                .collect(),
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

    async fn write_logits_contract_report(&self) -> Result<()> {
        let report = self.logits_contract_report()?;
        self.write_json(serde_json::to_string_pretty(&report)?)?;
        Ok(())
    }

    fn logits_contract_report(&self) -> Result<LogitsContractReport> {
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

        let mut config = BitNetConfig::default();
        if let Some(architecture) = &architecture {
            config.model.apply_architecture_defaults(architecture);
        }
        fill_runtime_contract_config(&reader, &mut config);

        let embedding_candidates = logits_tensor_candidates_report(
            &reader,
            &["token_embd.weight", "tok_embeddings.weight", "model.embed_tokens.weight"],
        )?;
        let lm_head_candidates = logits_tensor_candidates_report(
            &reader,
            &["output.weight", "lm_head.weight", "model.lm_head.weight"],
        )?;
        let final_norm_candidates = logits_tensor_candidates_report(
            &reader,
            &["output_norm.weight", "norm.weight", "model.norm.weight"],
        )?;

        let summary = logits_contract_summary(
            embedding_candidates.first().map(|tensor| tensor.shape.as_slice()),
            lm_head_candidates.first().map(|tensor| tensor.shape.as_slice()),
            !final_norm_candidates.is_empty(),
            config.model.vocab_size,
            config.model.hidden_size,
        );

        Ok(LogitsContractReport {
            schema_version: 1,
            diagnostic: "bitnet_logits_contract_report".to_string(),
            diagnostic_only: true,
            promotion_allowed: false,
            proof_receipts_written: false,
            manifest_updated: false,
            model: model_path.display().to_string(),
            model_sha256,
            gguf_metadata: LogitsContractGgufMetadata { architecture, file_type },
            effective_config: LogitsContractEffectiveConfig {
                vocab_size: config.model.vocab_size,
                hidden_size: config.model.hidden_size,
                norm_type: format!("{:?}", config.model.norm_type),
                activation_type: format!("{:?}", config.model.activation_type),
                rms_norm_eps: config.model.rms_norm_eps,
            },
            tensor_candidates: LogitsTensorCandidates {
                embedding: embedding_candidates,
                lm_head: lm_head_candidates,
                final_norm: final_norm_candidates,
            },
            token_probes: logits_token_probe_reports(
                &reader,
                &self.logits_token_ids,
                config.model.vocab_size,
                config.model.hidden_size,
            )?,
            runtime_mapping_policy: LogitsRuntimeMappingPolicy {
                embedding_candidates: vec![
                    "token_embd.weight".to_string(),
                    "tok_embeddings.weight".to_string(),
                    "model.embed_tokens.weight".to_string(),
                ],
                lm_head_candidates: vec![
                    "output.weight".to_string(),
                    "lm_head.weight".to_string(),
                    "model.lm_head.weight".to_string(),
                ],
                final_norm_candidates: vec![
                    "output_norm.weight".to_string(),
                    "norm.weight".to_string(),
                    "model.norm.weight".to_string(),
                ],
                dedicated_lm_head_runtime_name: "lm_head.weight".to_string(),
                tied_embedding_runtime_name: "embed_tokens.weight".to_string(),
                transposed_lm_head_flag_name: "lm_head.transposed".to_string(),
            },
            summary,
            not_claims: critical_qk256_report_not_claims()
                .into_iter()
                .chain([
                    "runtime_reference_parity",
                    "semantic_quality",
                    "a770_semantic_quality",
                    "logits_reference_parity",
                ])
                .map(str::to_string)
                .collect(),
        })
    }

    async fn write_bitnet_graph_contract_report(&self) -> Result<()> {
        let report = self.bitnet_graph_contract_report()?;
        self.write_json(serde_json::to_string_pretty(&report)?)?;
        Ok(())
    }

    fn bitnet_graph_contract_report(&self) -> Result<BitNetGraphContractReport> {
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

        let mut config = BitNetConfig::default();
        if let Some(architecture) = &architecture {
            config.model.apply_architecture_defaults(architecture);
        }
        fill_runtime_contract_config(&reader, &mut config);

        let layer0_required_tensors = bitnet_graph_required_tensor_reports(&reader, 0)?;
        let summary = bitnet_graph_contract_summary(&layer0_required_tensors);

        Ok(BitNetGraphContractReport {
            schema_version: 1,
            diagnostic: "bitnet_graph_contract_report".to_string(),
            diagnostic_only: true,
            promotion_allowed: false,
            proof_receipts_written: false,
            manifest_updated: false,
            model: model_path.display().to_string(),
            model_sha256,
            gguf_metadata: BitNetGraphGgufMetadata { architecture, file_type },
            effective_config: BitNetGraphEffectiveConfig {
                num_layers: config.model.num_layers,
                hidden_size: config.model.hidden_size,
                intermediate_size: config.model.intermediate_size,
                num_heads: config.model.num_heads,
                num_key_value_heads: config.model.num_key_value_heads,
                head_dim: config.model.hidden_size / config.model.num_heads.max(1),
                norm_type: format!("{:?}", config.model.norm_type),
                activation_type: format!("{:?}", config.model.activation_type),
                rms_norm_eps: config.model.rms_norm_eps,
            },
            reference_graph: bitnet_reference_graph_contract(),
            rust_graph: bitnet_rust_graph_contract(),
            layer0_required_tensors,
            summary,
            not_claims: critical_qk256_report_not_claims()
                .into_iter()
                .chain([
                    "runtime_reference_parity",
                    "semantic_quality",
                    "a770_semantic_quality",
                    "graph_numeric_parity",
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
struct I2sMatmulContractReport {
    schema_version: u32,
    diagnostic: String,
    diagnostic_only: bool,
    promotion_allowed: bool,
    proof_receipts_written: bool,
    manifest_updated: bool,
    model: String,
    model_sha256: String,
    gguf_metadata: I2sMatmulGgufMetadata,
    qk256_inventory: I2sQk256Inventory,
    rust_policy: I2sRustMatmulPolicy,
    reference_policy: I2sReferenceMatmulPolicy,
    summary: I2sMatmulContractSummary,
    not_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct I2sMatmulGgufMetadata {
    architecture: Option<String>,
    file_type: Option<u32>,
}

#[derive(Debug, Serialize)]
struct I2sQk256Inventory {
    tensor_count: usize,
    total_values: usize,
    total_code3_count: usize,
    total_code3_frequency: f64,
    max_tensor_code3_frequency: f64,
}

#[derive(Debug, Serialize)]
struct I2sRustMatmulPolicy {
    activation_quantization: String,
    dot_formula: String,
    code3_value: f32,
    uses_reference_activation_quantization: bool,
}

#[derive(Debug, Serialize)]
struct I2sReferenceMatmulPolicy {
    source: String,
    activation_quantization: String,
    correction_formula: String,
    matmul_effective_code3_value: f32,
    dequantize_row_i2_s_code3_value: f32,
}

#[derive(Debug, Serialize)]
struct I2sMatmulContractSummary {
    activation_quantization_policy_matched: bool,
    code3_runtime_blocker: bool,
    blocker: Option<String>,
    next_action: String,
}

#[derive(Debug, Serialize)]
struct RopeContractReport {
    schema_version: u32,
    diagnostic: String,
    diagnostic_only: bool,
    promotion_allowed: bool,
    proof_receipts_written: bool,
    manifest_updated: bool,
    model: String,
    model_sha256: String,
    gguf_metadata: RopeGgufMetadata,
    effective_rust_policy: RopeRustPolicy,
    reference_policy: RopeReferencePolicy,
    summary: RopeContractSummary,
    rope_freqs: Vec<RopeFreqsTensorReport>,
    not_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RopeGgufMetadata {
    architecture: Option<String>,
    context_length: Option<u32>,
    rope_freq_base: Option<f32>,
}

#[derive(Debug, Serialize)]
struct RopeRustPolicy {
    policy: String,
    uses_gguf_rope_freqs: bool,
    rope_layout: String,
    rope_layout_source: String,
    ggml_rope_type_id: u32,
    rope_theta: Option<f32>,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: Option<usize>,
    max_position_embeddings: usize,
}

#[derive(Debug, Serialize)]
struct RopeReferencePolicy {
    source: String,
    expects_optional_rope_freqs_tensor: bool,
    expected_tensor_suffix: String,
    rope_type: String,
    rope_type_id: u32,
    rope_layout: String,
    rope_layout_source: String,
}

#[derive(Debug, Serialize)]
struct RopeContractSummary {
    rope_freqs_tensor_count: usize,
    any_rope_freqs_tensor_present: bool,
    rust_uses_gguf_rope_freqs: bool,
    reference_rope_layout: String,
    rust_rope_layout: String,
    rust_rope_layout_matches_reference: bool,
    blocker: Option<String>,
    next_action: String,
}

#[derive(Debug, Serialize)]
struct RopeFreqsTensorReport {
    name: String,
    shape: Vec<usize>,
    tensor_type: String,
    element_count: usize,
    actual_bytes: usize,
    raw_sha256: String,
    sample_len: usize,
    sample_values_first_8: Vec<f32>,
    min: Option<f32>,
    max: Option<f32>,
    mean: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
struct RopeValueStats {
    min: Option<f32>,
    max: Option<f32>,
    mean: Option<f32>,
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

#[derive(Debug, Serialize)]
struct LogitsContractReport {
    schema_version: u32,
    diagnostic: String,
    diagnostic_only: bool,
    promotion_allowed: bool,
    proof_receipts_written: bool,
    manifest_updated: bool,
    model: String,
    model_sha256: String,
    gguf_metadata: LogitsContractGgufMetadata,
    effective_config: LogitsContractEffectiveConfig,
    tensor_candidates: LogitsTensorCandidates,
    token_probes: Vec<LogitsTokenProbeReport>,
    runtime_mapping_policy: LogitsRuntimeMappingPolicy,
    summary: LogitsContractSummary,
    not_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LogitsContractGgufMetadata {
    architecture: Option<String>,
    file_type: Option<u32>,
}

#[derive(Debug, Serialize)]
struct LogitsContractEffectiveConfig {
    vocab_size: usize,
    hidden_size: usize,
    norm_type: String,
    activation_type: String,
    rms_norm_eps: Option<f32>,
}

#[derive(Debug, Serialize)]
struct LogitsTensorCandidates {
    embedding: Vec<LogitsTensorReport>,
    lm_head: Vec<LogitsTensorReport>,
    final_norm: Vec<LogitsTensorReport>,
}

#[derive(Debug, Serialize)]
struct LogitsTensorReport {
    name: String,
    shape: Vec<usize>,
    tensor_type: String,
    actual_bytes: usize,
    sample_sha256_first_4096: String,
}

#[derive(Debug, Serialize)]
struct LogitsTokenProbeReport {
    token_id: u32,
    source_tensor: Option<String>,
    source_orientation: Option<String>,
    extraction_axis: Option<String>,
    present: bool,
    reason: Option<String>,
    value_count: Option<usize>,
    mean: Option<f32>,
    rms: Option<f32>,
    min: Option<f32>,
    max: Option<f32>,
    vector_sha256_f32_le: Option<String>,
    first_values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct LogitsRuntimeMappingPolicy {
    embedding_candidates: Vec<String>,
    lm_head_candidates: Vec<String>,
    final_norm_candidates: Vec<String>,
    dedicated_lm_head_runtime_name: String,
    tied_embedding_runtime_name: String,
    transposed_lm_head_flag_name: String,
}

#[derive(Debug, Serialize)]
struct LogitsContractSummary {
    embedding_present: bool,
    final_norm_present: bool,
    dedicated_lm_head_present: bool,
    tied_logits_expected: bool,
    embedding_orientation: Option<String>,
    lm_head_orientation: Option<String>,
    lm_head_transposed_expected: Option<bool>,
    runtime_logits_source: String,
    blocker: Option<String>,
    next_action: String,
}

#[derive(Debug, Serialize)]
struct BitNetGraphContractReport {
    schema_version: u32,
    diagnostic: String,
    diagnostic_only: bool,
    promotion_allowed: bool,
    proof_receipts_written: bool,
    manifest_updated: bool,
    model: String,
    model_sha256: String,
    gguf_metadata: BitNetGraphGgufMetadata,
    effective_config: BitNetGraphEffectiveConfig,
    reference_graph: BitNetGraphContract,
    rust_graph: BitNetGraphContract,
    layer0_required_tensors: Vec<BitNetGraphTensorReport>,
    summary: BitNetGraphContractSummary,
    not_claims: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BitNetGraphGgufMetadata {
    architecture: Option<String>,
    file_type: Option<u32>,
}

#[derive(Debug, Serialize)]
struct BitNetGraphEffectiveConfig {
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    norm_type: String,
    activation_type: String,
    rms_norm_eps: Option<f32>,
}

#[derive(Debug, Serialize)]
struct BitNetGraphContract {
    source: String,
    per_layer_order: Vec<String>,
    attention_subnorm_position: String,
    ffn_activation: String,
    ffn_mode: String,
    ffn_subnorm_position: String,
    final_norm_position: String,
    logits_source: String,
}

#[derive(Debug, Serialize)]
struct BitNetGraphTensorReport {
    role: String,
    gguf_name: String,
    rust_runtime_name: String,
    present: bool,
    shape: Option<Vec<usize>>,
    tensor_type: Option<String>,
    actual_bytes: Option<usize>,
    sample_sha256_first_4096: Option<String>,
}

#[derive(Debug, Serialize)]
struct BitNetGraphContractSummary {
    layer0_required_tensor_count: usize,
    layer0_required_tensor_present_count: usize,
    layer0_required_tensors_present: bool,
    reference_and_rust_stage_order_match: bool,
    attention_subnorm_before_o_proj: bool,
    ffn_subnorm_before_down_proj: bool,
    residual_after_o_proj: bool,
    residual_after_down_proj: bool,
    final_norm_before_tied_logits: bool,
    blocker: Option<String>,
    next_action: String,
}

fn logits_tensor_candidates_report(
    reader: &GgufReader,
    candidates: &[&str],
) -> Result<Vec<LogitsTensorReport>> {
    let mut reports = Vec::new();
    for name in candidates {
        if let Some(info) = reader.get_tensor_info_by_name(name) {
            let data = reader.get_tensor_data_by_info(info)?;
            let sample_len = data.len().min(4096);
            reports.push(LogitsTensorReport {
                name: info.name.clone(),
                shape: info.shape.clone(),
                tensor_type: format!("{:?}", info.tensor_type),
                actual_bytes: data.len(),
                sample_sha256_first_4096: sha256_hex_bytes(&data[..sample_len]),
            });
        }
    }
    Ok(reports)
}

fn logits_token_probe_reports(
    reader: &GgufReader,
    token_ids: &[u32],
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Vec<LogitsTokenProbeReport>> {
    if token_ids.is_empty() {
        return Ok(Vec::new());
    }

    let Some(info) = ["token_embd.weight", "tok_embeddings.weight", "model.embed_tokens.weight"]
        .iter()
        .find_map(|name| reader.get_tensor_info_by_name(name))
    else {
        return Ok(token_ids
            .iter()
            .copied()
            .map(|token_id| LogitsTokenProbeReport {
                token_id,
                source_tensor: None,
                source_orientation: None,
                extraction_axis: None,
                present: false,
                reason: Some("embedding_tensor_missing".to_string()),
                value_count: None,
                mean: None,
                rms: None,
                min: None,
                max: None,
                vector_sha256_f32_le: None,
                first_values: Vec::new(),
            })
            .collect());
    };

    let data = reader.get_tensor_data_by_info(info)?;
    token_ids
        .iter()
        .copied()
        .map(|token_id| {
            logits_token_probe_report(info, data, token_id, vocab_size, hidden_size)
                .with_context(|| format!("probing logits token id {token_id}"))
        })
        .collect()
}

fn logits_token_probe_report(
    info: &bitnet_models::formats::gguf::TensorInfo,
    data: &[u8],
    token_id: u32,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<LogitsTokenProbeReport> {
    let orientation = logits_matrix_orientation(&info.shape, vocab_size, hidden_size);
    let token_index = token_id as usize;
    if token_index >= vocab_size {
        return Ok(LogitsTokenProbeReport {
            token_id,
            source_tensor: Some(info.name.clone()),
            source_orientation: Some(orientation),
            extraction_axis: None,
            present: false,
            reason: Some("token_id_out_of_vocab".to_string()),
            value_count: None,
            mean: None,
            rms: None,
            min: None,
            max: None,
            vector_sha256_f32_le: None,
            first_values: Vec::new(),
        });
    }

    let (axis, values) = match orientation.as_str() {
        "vocab_hidden" => {
            let base = token_index.checked_mul(hidden_size).ok_or_else(|| {
                anyhow::anyhow!("token probe row offset overflow for {}", info.name)
            })?;
            let values = (0..hidden_size)
                .map(|idx| tensor_scalar_at(data, info.tensor_type, base + idx))
                .collect::<Result<Vec<_>>>()?;
            ("row", values)
        }
        "hidden_vocab" => {
            let values = (0..hidden_size)
                .map(|row| {
                    let idx = row
                        .checked_mul(vocab_size)
                        .and_then(|base| base.checked_add(token_index))
                        .ok_or_else(|| {
                            anyhow::anyhow!("token probe column offset overflow for {}", info.name)
                        })?;
                    tensor_scalar_at(data, info.tensor_type, idx)
                })
                .collect::<Result<Vec<_>>>()?;
            ("column", values)
        }
        _ => {
            return Ok(LogitsTokenProbeReport {
                token_id,
                source_tensor: Some(info.name.clone()),
                source_orientation: Some(orientation),
                extraction_axis: None,
                present: false,
                reason: Some("embedding_shape_unexpected_for_token_probe".to_string()),
                value_count: None,
                mean: None,
                rms: None,
                min: None,
                max: None,
                vector_sha256_f32_le: None,
                first_values: Vec::new(),
            });
        }
    };

    let stats = f32_stats(&values);
    Ok(LogitsTokenProbeReport {
        token_id,
        source_tensor: Some(info.name.clone()),
        source_orientation: Some(orientation),
        extraction_axis: Some(axis.to_string()),
        present: true,
        reason: None,
        value_count: Some(values.len()),
        mean: stats.mean,
        rms: stats.rms,
        min: stats.min,
        max: stats.max,
        vector_sha256_f32_le: Some(sha256_hex_f32_values(&values)),
        first_values: values.into_iter().take(8).collect(),
    })
}

fn tensor_scalar_at(data: &[u8], tensor_type: GgufTensorType, index: usize) -> Result<f32> {
    match tensor_type {
        GgufTensorType::F32 => {
            let offset =
                index.checked_mul(4).ok_or_else(|| anyhow::anyhow!("f32 byte offset overflow"))?;
            let end =
                offset.checked_add(4).ok_or_else(|| anyhow::anyhow!("f32 byte end overflow"))?;
            let bytes = data
                .get(offset..end)
                .ok_or_else(|| anyhow::anyhow!("f32 tensor index {index} out of bounds"))?;
            Ok(f32::from_le_bytes(bytes.try_into()?))
        }
        GgufTensorType::F16 => {
            let offset =
                index.checked_mul(2).ok_or_else(|| anyhow::anyhow!("f16 byte offset overflow"))?;
            let end =
                offset.checked_add(2).ok_or_else(|| anyhow::anyhow!("f16 byte end overflow"))?;
            let bytes = data
                .get(offset..end)
                .ok_or_else(|| anyhow::anyhow!("f16 tensor index {index} out of bounds"))?;
            Ok(half::f16::from_bits(u16::from_le_bytes(bytes.try_into()?)).to_f32())
        }
        other => anyhow::bail!(
            "unsupported logits token probe tensor type {:?}; expected F32 or F16",
            other
        ),
    }
}

struct F32Stats {
    mean: Option<f32>,
    rms: Option<f32>,
    min: Option<f32>,
    max: Option<f32>,
}

fn f32_stats(values: &[f32]) -> F32Stats {
    if values.is_empty() {
        return F32Stats { mean: None, rms: None, min: None, max: None };
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += *value as f64;
        sum_sq += (*value as f64) * (*value as f64);
    }
    let count = values.len() as f64;
    F32Stats {
        mean: Some((sum / count) as f32),
        rms: Some((sum_sq / count).sqrt() as f32),
        min: Some(min),
        max: Some(max),
    }
}

fn sha256_hex_f32_values(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn logits_matrix_orientation(shape: &[usize], vocab_size: usize, hidden_size: usize) -> String {
    match shape {
        [v, h] if *v == vocab_size && *h == hidden_size => "vocab_hidden".to_string(),
        [h, v] if *h == hidden_size && *v == vocab_size => "hidden_vocab".to_string(),
        [_, _] => "unexpected_2d".to_string(),
        _ => "not_2d".to_string(),
    }
}

fn logits_contract_summary(
    embedding_shape: Option<&[usize]>,
    lm_head_shape: Option<&[usize]>,
    final_norm_present: bool,
    vocab_size: usize,
    hidden_size: usize,
) -> LogitsContractSummary {
    let embedding_orientation =
        embedding_shape.map(|shape| logits_matrix_orientation(shape, vocab_size, hidden_size));
    let lm_head_orientation =
        lm_head_shape.map(|shape| logits_matrix_orientation(shape, vocab_size, hidden_size));

    let embedding_present = embedding_shape.is_some();
    let dedicated_lm_head_present = lm_head_shape.is_some();
    let tied_logits_expected = !dedicated_lm_head_present && embedding_present;
    let lm_head_transposed_expected =
        lm_head_orientation.as_deref().map(|orientation| matches!(orientation, "hidden_vocab"));

    let runtime_logits_source =
        match (lm_head_orientation.as_deref(), embedding_present, final_norm_present) {
            (Some("vocab_hidden"), _, true) => "dedicated_lm_head_standard".to_string(),
            (Some("hidden_vocab"), _, true) => "dedicated_lm_head_transposed".to_string(),
            (Some("unexpected_2d" | "not_2d"), _, _) => "blocked_lm_head_shape".to_string(),
            (Some(_), _, true) => "blocked_lm_head_shape".to_string(),
            (None, true, true) => "tied_embeddings".to_string(),
            (None, false, _) => "blocked_missing_logits_source".to_string(),
            (_, _, false) => "blocked_missing_final_norm".to_string(),
        };

    let blocker = if !final_norm_present {
        Some("missing_final_norm_tensor".to_string())
    } else if matches!(lm_head_orientation.as_deref(), Some("unexpected_2d" | "not_2d")) {
        Some("lm_head_shape_unexpected_for_runtime_logits".to_string())
    } else if !dedicated_lm_head_present && !embedding_present {
        Some("missing_lm_head_and_embedding_logits_source".to_string())
    } else {
        None
    };

    let next_action = if blocker.is_none() {
        "compare_reference_rust_logits_or_first_token_after_logits_contract".to_string()
    } else {
        "fix_logits_tensor_contract_before_reference_parity".to_string()
    };

    LogitsContractSummary {
        embedding_present,
        final_norm_present,
        dedicated_lm_head_present,
        tied_logits_expected,
        embedding_orientation,
        lm_head_orientation,
        lm_head_transposed_expected,
        runtime_logits_source,
        blocker,
        next_action,
    }
}

fn bitnet_reference_graph_contract() -> BitNetGraphContract {
    BitNetGraphContract {
        source: "llama.cpp build_bitnet_158".to_string(),
        per_layer_order: vec![
            "attn_norm_rms".to_string(),
            "qkv_projection".to_string(),
            "rope_qk".to_string(),
            "kv_attention_value_mix".to_string(),
            "attn_sub_norm_rms".to_string(),
            "o_proj".to_string(),
            "attention_residual".to_string(),
            "ffn_norm_rms".to_string(),
            "ffn_up_gate_parallel_relu_squared".to_string(),
            "ffn_sub_norm_rms".to_string(),
            "ffn_down".to_string(),
            "ffn_residual".to_string(),
        ],
        attention_subnorm_position: "after_attention_value_mix_before_o_proj".to_string(),
        ffn_activation: "relu_squared".to_string(),
        ffn_mode: "parallel_up_gate".to_string(),
        ffn_subnorm_position: "after_parallel_ffn_before_down_proj".to_string(),
        final_norm_position: "after_last_layer_before_logits".to_string(),
        logits_source: "tied_token_embedding".to_string(),
    }
}

fn bitnet_rust_graph_contract() -> BitNetGraphContract {
    BitNetGraphContract {
        source: "crates/bitnet-transformer/src/lib.rs".to_string(),
        per_layer_order: vec![
            "attention_norm.forward".to_string(),
            "q_proj_k_proj_v_proj".to_string(),
            "rotary_embedding_apply_qk".to_string(),
            "attention_softmax_value_mix".to_string(),
            "attention.sub_layernorm.forward".to_string(),
            "o_proj".to_string(),
            "attention_residual".to_string(),
            "post_attention_layernorm.forward".to_string(),
            "gate_proj_relu_squared_times_up_proj".to_string(),
            "feed_forward.sub_layernorm.forward".to_string(),
            "down_proj".to_string(),
            "ffn_residual".to_string(),
        ],
        attention_subnorm_position: "after_attention_value_mix_before_o_proj".to_string(),
        ffn_activation: "relu_squared".to_string(),
        ffn_mode: "parallel_up_gate".to_string(),
        ffn_subnorm_position: "after_parallel_ffn_before_down_proj".to_string(),
        final_norm_position: "after_last_layer_before_logits".to_string(),
        logits_source: "tied_token_embedding".to_string(),
    }
}

fn bitnet_graph_required_tensor_reports(
    reader: &GgufReader,
    layer_idx: usize,
) -> Result<Vec<BitNetGraphTensorReport>> {
    let tensor_specs = [
        (
            "attention_norm",
            format!("blk.{layer_idx}.attn_norm.weight"),
            format!("layers.{layer_idx}.attention_norm.weight"),
        ),
        (
            "attention_subnorm",
            format!("blk.{layer_idx}.attn_sub_norm.weight"),
            format!("layers.{layer_idx}.attention.sub_layernorm.weight"),
        ),
        (
            "ffn_norm",
            format!("blk.{layer_idx}.ffn_norm.weight"),
            format!("layers.{layer_idx}.post_attention_layernorm.weight"),
        ),
        (
            "ffn_subnorm",
            format!("blk.{layer_idx}.ffn_sub_norm.weight"),
            format!("layers.{layer_idx}.feed_forward.sub_layernorm.weight"),
        ),
    ];

    tensor_specs
        .into_iter()
        .map(|(role, gguf_name, rust_runtime_name)| {
            bitnet_graph_tensor_report(reader, role, gguf_name, rust_runtime_name)
        })
        .collect()
}

fn bitnet_graph_tensor_report(
    reader: &GgufReader,
    role: &str,
    gguf_name: String,
    rust_runtime_name: String,
) -> Result<BitNetGraphTensorReport> {
    let Some(info) = reader.get_tensor_info_by_name(&gguf_name) else {
        return Ok(BitNetGraphTensorReport {
            role: role.to_string(),
            gguf_name,
            rust_runtime_name,
            present: false,
            shape: None,
            tensor_type: None,
            actual_bytes: None,
            sample_sha256_first_4096: None,
        });
    };

    let data = reader.get_tensor_data_by_info(info)?;
    let sample_len = data.len().min(4096);
    Ok(BitNetGraphTensorReport {
        role: role.to_string(),
        gguf_name: info.name.clone(),
        rust_runtime_name,
        present: true,
        shape: Some(info.shape.clone()),
        tensor_type: Some(format!("{:?}", info.tensor_type)),
        actual_bytes: Some(data.len()),
        sample_sha256_first_4096: Some(sha256_hex_bytes(&data[..sample_len])),
    })
}

fn bitnet_graph_contract_summary(
    layer0_required_tensors: &[BitNetGraphTensorReport],
) -> BitNetGraphContractSummary {
    let layer0_required_tensor_count = layer0_required_tensors.len();
    let layer0_required_tensor_present_count =
        layer0_required_tensors.iter().filter(|tensor| tensor.present).count();
    let layer0_required_tensors_present =
        layer0_required_tensor_present_count == layer0_required_tensor_count;

    let reference = bitnet_reference_graph_contract();
    let rust = bitnet_rust_graph_contract();
    let reference_and_rust_stage_order_match = reference.attention_subnorm_position
        == rust.attention_subnorm_position
        && reference.ffn_activation == rust.ffn_activation
        && reference.ffn_mode == rust.ffn_mode
        && reference.ffn_subnorm_position == rust.ffn_subnorm_position
        && reference.final_norm_position == rust.final_norm_position
        && reference.logits_source == rust.logits_source;

    let blocker = if !layer0_required_tensors_present {
        Some("missing_bitnet_layer0_graph_required_tensor".to_string())
    } else if !reference_and_rust_stage_order_match {
        Some("bitnet_reference_rust_static_graph_contract_mismatch".to_string())
    } else {
        None
    };

    let next_action = if blocker.is_none() {
        "run_reference_rust_layer_trace_or_compare_first_divergence".to_string()
    } else {
        "fix_bitnet_graph_contract_before_reference_parity".to_string()
    };

    BitNetGraphContractSummary {
        layer0_required_tensor_count,
        layer0_required_tensor_present_count,
        layer0_required_tensors_present,
        reference_and_rust_stage_order_match,
        attention_subnorm_before_o_proj: true,
        ffn_subnorm_before_down_proj: true,
        residual_after_o_proj: true,
        residual_after_down_proj: true,
        final_norm_before_tied_logits: true,
        blocker,
        next_action,
    }
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

fn qk256_contract_inventory(reader: &GgufReader) -> Result<I2sQk256Inventory> {
    let mut tensor_count = 0usize;
    let mut total_values = 0usize;
    let mut total_code3_count = 0usize;
    let mut max_tensor_code3_frequency = 0.0f64;

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
        let logical_packed_bytes = gguf_rows
            .checked_mul(row_stride_bytes)
            .ok_or_else(|| anyhow::anyhow!("QK256 tensor '{}' byte count overflow", info.name))?;
        let data = reader.get_tensor_data_by_info(info)?;
        if data.len() < logical_packed_bytes {
            continue;
        }

        let hist = qk256_code_histogram_act_parallel_rows(
            &data[..logical_packed_bytes],
            gguf_rows,
            gguf_cols,
            row_stride_bytes,
        );
        let code3_frequency = if nelems == 0 { 0.0 } else { hist[3] as f64 / nelems as f64 };

        tensor_count += 1;
        total_values += nelems;
        total_code3_count += hist[3];
        max_tensor_code3_frequency = max_tensor_code3_frequency.max(code3_frequency);
    }

    let total_code3_frequency =
        if total_values == 0 { 0.0 } else { total_code3_count as f64 / total_values as f64 };

    Ok(I2sQk256Inventory {
        tensor_count,
        total_values,
        total_code3_count,
        total_code3_frequency,
        max_tensor_code3_frequency,
    })
}

fn i2s_matmul_contract_summary(
    _inventory: &I2sQk256Inventory,
    rust_uses_reference_activation_quantization: bool,
) -> I2sMatmulContractSummary {
    let activation_quantization_policy_matched = rust_uses_reference_activation_quantization;
    let code3_runtime_blocker = false;
    let blocker = if !activation_quantization_policy_matched {
        Some("qk256_activation_quantization_rule_unimplemented".to_string())
    } else {
        None
    };
    let next_action = if !activation_quantization_policy_matched {
        "write a focused reference-compatible I2_S activation-quantized GEMV proof before changing claim state"
    } else {
        "continue Rust/reference localization outside I2_S activation quantization"
    };

    I2sMatmulContractSummary {
        activation_quantization_policy_matched,
        code3_runtime_blocker,
        blocker,
        next_action: next_action.to_string(),
    }
}

fn is_rope_freqs_tensor_name(name: &str) -> bool {
    name.ends_with("rope_freqs.weight") || name.contains(".rope_freqs.")
}

fn decode_rope_freq_values(
    name: &str,
    tensor_type: GgufTensorType,
    data: &[u8],
) -> Result<Vec<f32>> {
    match tensor_type {
        GgufTensorType::F32 => {
            if !data.len().is_multiple_of(std::mem::size_of::<f32>()) {
                anyhow::bail!(
                    "rope_freqs tensor '{}' has {} bytes, not divisible by f32 size",
                    name,
                    data.len()
                );
            }
            Ok(bytemuck::cast_slice::<u8, f32>(data).to_vec())
        }
        GgufTensorType::F16 => {
            if !data.len().is_multiple_of(std::mem::size_of::<u16>()) {
                anyhow::bail!(
                    "rope_freqs tensor '{}' has {} bytes, not divisible by f16 size",
                    name,
                    data.len()
                );
            }
            Ok(bytemuck::cast_slice::<u8, u16>(data)
                .iter()
                .map(|bits| half::f16::from_bits(*bits).to_f32())
                .collect())
        }
        other => anyhow::bail!(
            "rope_freqs tensor '{}' has unsupported tensor type {:?}; expected F32 or F16",
            name,
            other
        ),
    }
}

fn rope_value_stats(values: &[f32]) -> RopeValueStats {
    if values.is_empty() {
        return RopeValueStats { min: None, max: None, mean: None };
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += *value as f64;
    }

    RopeValueStats {
        min: Some(min),
        max: Some(max),
        mean: Some((sum / values.len() as f64) as f32),
    }
}

fn rope_contract_summary(
    rope_freqs_tensor_count: usize,
    rust_uses_gguf_rope_freqs: bool,
) -> RopeContractSummary {
    let any_rope_freqs_tensor_present = rope_freqs_tensor_count > 0;
    let reference_rope_layout = "neox_offset_by_half".to_string();
    let rust_rope_layout = "neox_offset_by_half".to_string();
    let rust_rope_layout_matches_reference = rust_rope_layout == reference_rope_layout;
    let blocker = if any_rope_freqs_tensor_present && !rust_uses_gguf_rope_freqs {
        Some("bitnet_b158_rope_factors_present_but_rust_policy_ignores_them".to_string())
    } else if !rust_rope_layout_matches_reference {
        Some("bitnet_b158_rope_layout_mismatch".to_string())
    } else {
        None
    };
    let next_action = if blocker.is_some() {
        "add a focused Rust/reference RoPE factor parity proof before changing runtime math"
    } else if any_rope_freqs_tensor_present {
        "verify RoPE factor use against reference receipt"
    } else {
        "continue shared Rust/reference localization; no GGUF rope_freqs tensor was found"
    };

    RopeContractSummary {
        rope_freqs_tensor_count,
        any_rope_freqs_tensor_present,
        rust_uses_gguf_rope_freqs,
        reference_rope_layout,
        rust_rope_layout,
        rust_rope_layout_matches_reference,
        blocker,
        next_action: next_action.to_string(),
    }
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

    #[test]
    fn rope_freqs_tensor_name_matches_reference_suffix_only() {
        assert!(is_rope_freqs_tensor_name("blk.0.rope_freqs.weight"));
        assert!(is_rope_freqs_tensor_name("model.layers.0.rope_freqs.weight"));
        assert!(!is_rope_freqs_tensor_name("blk.0.attn_q.weight"));
        assert!(!is_rope_freqs_tensor_name("rope.freq_base"));
    }

    #[test]
    fn rope_contract_summary_blocks_present_factors_when_rust_ignores_them() {
        let summary = rope_contract_summary(1, false);

        assert!(summary.any_rope_freqs_tensor_present);
        assert!(!summary.rust_uses_gguf_rope_freqs);
        assert!(summary.rust_rope_layout_matches_reference);
        assert_eq!(
            summary.blocker.as_deref(),
            Some("bitnet_b158_rope_factors_present_but_rust_policy_ignores_them")
        );
    }

    #[test]
    fn rope_contract_summary_allows_absent_factors_without_rope_blocker() {
        let summary = rope_contract_summary(0, false);

        assert!(!summary.any_rope_freqs_tensor_present);
        assert_eq!(summary.reference_rope_layout, "neox_offset_by_half");
        assert_eq!(summary.rust_rope_layout, "neox_offset_by_half");
        assert!(summary.rust_rope_layout_matches_reference);
        assert_eq!(summary.blocker, None);
    }

    #[test]
    fn decode_rope_freq_values_reads_f32_and_f16_values() {
        let f32_bytes = bytemuck::cast_slice::<f32, u8>(&[1.0, 2.5, 4.0]);
        let f32_values =
            decode_rope_freq_values("blk.0.rope_freqs.weight", GgufTensorType::F32, f32_bytes)
                .unwrap();
        assert_eq!(f32_values, vec![1.0, 2.5, 4.0]);

        let f16_bits = [half::f16::from_f32(0.5).to_bits(), half::f16::from_f32(1.5).to_bits()];
        let f16_bytes = bytemuck::cast_slice::<u16, u8>(&f16_bits);
        let f16_values =
            decode_rope_freq_values("blk.0.rope_freqs.weight", GgufTensorType::F16, f16_bytes)
                .unwrap();
        assert_eq!(f16_values, vec![0.5, 1.5]);
    }

    #[test]
    fn i2s_matmul_summary_blocks_missing_activation_quantization() {
        let inventory = I2sQk256Inventory {
            tensor_count: 210,
            total_values: 2_084_044_800,
            total_code3_count: 0,
            total_code3_frequency: 0.0,
            max_tensor_code3_frequency: 0.0,
        };

        let summary = i2s_matmul_contract_summary(&inventory, false);

        assert!(!summary.activation_quantization_policy_matched);
        assert!(!summary.code3_runtime_blocker);
        assert_eq!(
            summary.blocker.as_deref(),
            Some("qk256_activation_quantization_rule_unimplemented")
        );
    }

    #[test]
    fn i2s_matmul_summary_does_not_treat_code3_as_matmul_blocker() {
        let inventory = I2sQk256Inventory {
            tensor_count: 1,
            total_values: 256,
            total_code3_count: 1,
            total_code3_frequency: 1.0 / 256.0,
            max_tensor_code3_frequency: 1.0 / 256.0,
        };

        let summary = i2s_matmul_contract_summary(&inventory, true);

        assert!(summary.activation_quantization_policy_matched);
        assert!(!summary.code3_runtime_blocker);
        assert_eq!(summary.blocker, None);
    }

    #[test]
    fn logits_matrix_orientation_classifies_runtime_shapes() {
        assert_eq!(logits_matrix_orientation(&[128, 16], 128, 16), "vocab_hidden");
        assert_eq!(logits_matrix_orientation(&[16, 128], 128, 16), "hidden_vocab");
        assert_eq!(logits_matrix_orientation(&[7, 9], 128, 16), "unexpected_2d");
        assert_eq!(logits_matrix_orientation(&[128, 16, 1], 128, 16), "not_2d");
    }

    #[test]
    fn logits_contract_summary_detects_transposed_dedicated_head() {
        let summary = logits_contract_summary(Some(&[128, 16]), Some(&[16, 128]), true, 128, 16);

        assert!(summary.embedding_present);
        assert!(summary.dedicated_lm_head_present);
        assert!(!summary.tied_logits_expected);
        assert_eq!(summary.lm_head_orientation.as_deref(), Some("hidden_vocab"));
        assert_eq!(summary.lm_head_transposed_expected, Some(true));
        assert_eq!(summary.runtime_logits_source, "dedicated_lm_head_transposed");
        assert_eq!(summary.blocker, None);
    }

    #[test]
    fn logits_contract_summary_uses_tied_embeddings_without_lm_head() {
        let summary = logits_contract_summary(Some(&[128, 16]), None, true, 128, 16);

        assert!(summary.embedding_present);
        assert!(!summary.dedicated_lm_head_present);
        assert!(summary.tied_logits_expected);
        assert_eq!(summary.runtime_logits_source, "tied_embeddings");
        assert_eq!(summary.blocker, None);
    }

    #[test]
    fn logits_token_probe_extracts_hidden_vocab_column() {
        let info = bitnet_models::formats::gguf::TensorInfo {
            name: "token_embd.weight".to_string(),
            shape: vec![3, 4],
            tensor_type: GgufTensorType::F32,
            offset: 0,
            size: 48,
        };
        let values = [0.0f32, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0];
        let data = bytemuck::cast_slice::<f32, u8>(&values);

        let report = logits_token_probe_report(&info, data, 2, 4, 3).expect("token probe");

        assert!(report.present);
        assert_eq!(report.source_orientation.as_deref(), Some("hidden_vocab"));
        assert_eq!(report.extraction_axis.as_deref(), Some("column"));
        assert_eq!(report.value_count, Some(3));
        assert_eq!(report.first_values, vec![2.0, 12.0, 22.0]);
        assert_eq!(report.mean, Some(12.0));
        assert_eq!(report.min, Some(2.0));
        assert_eq!(report.max, Some(22.0));
        assert!(report.vector_sha256_f32_le.is_some());
    }

    #[test]
    fn logits_token_probe_extracts_vocab_hidden_row() {
        let info = bitnet_models::formats::gguf::TensorInfo {
            name: "token_embd.weight".to_string(),
            shape: vec![4, 3],
            tensor_type: GgufTensorType::F32,
            offset: 0,
            size: 48,
        };
        let values = [0.0f32, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0];
        let data = bytemuck::cast_slice::<f32, u8>(&values);

        let report = logits_token_probe_report(&info, data, 2, 4, 3).expect("token probe");

        assert!(report.present);
        assert_eq!(report.source_orientation.as_deref(), Some("vocab_hidden"));
        assert_eq!(report.extraction_axis.as_deref(), Some("row"));
        assert_eq!(report.value_count, Some(3));
        assert_eq!(report.first_values, vec![20.0, 21.0, 22.0]);
        assert_eq!(report.mean, Some(21.0));
        assert_eq!(report.min, Some(20.0));
        assert_eq!(report.max, Some(22.0));
        assert!(report.vector_sha256_f32_le.is_some());
    }

    #[test]
    fn logits_token_probe_reports_out_of_vocab() {
        let info = bitnet_models::formats::gguf::TensorInfo {
            name: "token_embd.weight".to_string(),
            shape: vec![4, 3],
            tensor_type: GgufTensorType::F32,
            offset: 0,
            size: 48,
        };
        let values = [0.0f32; 12];
        let data = bytemuck::cast_slice::<f32, u8>(&values);

        let report = logits_token_probe_report(&info, data, 99, 4, 3).expect("token probe");

        assert!(!report.present);
        assert_eq!(report.reason.as_deref(), Some("token_id_out_of_vocab"));
        assert_eq!(report.value_count, None);
        assert!(report.first_values.is_empty());
    }

    #[test]
    fn logits_contract_summary_blocks_missing_final_norm() {
        let summary = logits_contract_summary(Some(&[128, 16]), Some(&[128, 16]), false, 128, 16);

        assert_eq!(summary.runtime_logits_source, "blocked_missing_final_norm");
        assert_eq!(summary.blocker.as_deref(), Some("missing_final_norm_tensor"));
    }

    #[test]
    fn bitnet_graph_contract_pins_reference_subnorm_order() {
        let reference = bitnet_reference_graph_contract();
        let rust = bitnet_rust_graph_contract();

        assert_eq!(reference.attention_subnorm_position, "after_attention_value_mix_before_o_proj");
        assert_eq!(reference.ffn_subnorm_position, "after_parallel_ffn_before_down_proj");
        assert_eq!(reference.ffn_activation, "relu_squared");
        assert_eq!(reference.ffn_mode, "parallel_up_gate");
        assert_eq!(reference.final_norm_position, "after_last_layer_before_logits");

        assert_eq!(rust.attention_subnorm_position, reference.attention_subnorm_position);
        assert_eq!(rust.ffn_subnorm_position, reference.ffn_subnorm_position);
        assert_eq!(rust.ffn_activation, reference.ffn_activation);
        assert_eq!(rust.ffn_mode, reference.ffn_mode);
    }

    #[test]
    fn bitnet_graph_summary_allows_complete_static_contract() {
        let tensors = ["attention_norm", "attention_subnorm", "ffn_norm", "ffn_subnorm"]
            .into_iter()
            .map(|role| BitNetGraphTensorReport {
                role: role.to_string(),
                gguf_name: format!("blk.0.{role}.weight"),
                rust_runtime_name: format!("layers.0.{role}.weight"),
                present: true,
                shape: Some(vec![16]),
                tensor_type: Some("F32".to_string()),
                actual_bytes: Some(64),
                sample_sha256_first_4096: Some("hash".to_string()),
            })
            .collect::<Vec<_>>();

        let summary = bitnet_graph_contract_summary(&tensors);

        assert!(summary.layer0_required_tensors_present);
        assert!(summary.reference_and_rust_stage_order_match);
        assert!(summary.attention_subnorm_before_o_proj);
        assert!(summary.ffn_subnorm_before_down_proj);
        assert_eq!(summary.blocker, None);
        assert_eq!(
            summary.next_action,
            "run_reference_rust_layer_trace_or_compare_first_divergence"
        );
    }

    #[test]
    fn bitnet_graph_summary_blocks_missing_required_tensor() {
        let tensors = ["attention_norm", "attention_subnorm", "ffn_norm", "ffn_subnorm"]
            .into_iter()
            .enumerate()
            .map(|(idx, role)| BitNetGraphTensorReport {
                role: role.to_string(),
                gguf_name: format!("blk.0.{role}.weight"),
                rust_runtime_name: format!("layers.0.{role}.weight"),
                present: idx != 1,
                shape: Some(vec![16]),
                tensor_type: Some("F32".to_string()),
                actual_bytes: Some(64),
                sample_sha256_first_4096: Some("hash".to_string()),
            })
            .collect::<Vec<_>>();

        let summary = bitnet_graph_contract_summary(&tensors);

        assert!(!summary.layer0_required_tensors_present);
        assert_eq!(summary.layer0_required_tensor_present_count, 3);
        assert_eq!(summary.blocker.as_deref(), Some("missing_bitnet_layer0_graph_required_tensor"));
    }
}
