//! Output-head and logits-index audit for Lunar Lake BitNet CPU proof.

use anyhow::{Context, Result};
use bitnet_models::formats::gguf::{GgufReader, TensorInfo};
use bitnet_tokenizers::Tokenizer;
use clap::Args;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

const EMBEDDING_CANDIDATES: &[&str] = &[
    "token_embd.weight",
    "tok_embeddings.weight",
    "model.tok_embeddings.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "transformer.wte.weight",
];

const OUTPUT_HEAD_CANDIDATES: &[&str] = &[
    "output.weight",
    "lm_head.weight",
    "model.lm_head.weight",
    "transformer.lm_head.weight",
    "language_model_head.weight",
    "head.weight",
    "generator.weight",
];

const RAW_QK256_TIED_CANDIDATES: &[&str] = &[
    "embed_tokens.weight.qk256_qs",
    "token_embd.weight.qk256_qs",
    "tok_embeddings.weight.qk256_qs",
];

/// Audit the output-head/tied-head/logits-index boundary for 258V CPU proof.
#[derive(Args, Debug)]
pub struct OutputHeadLogitsAuditCommand {
    /// GGUF model to inspect.
    #[arg(long, value_name = "PATH")]
    pub model: PathBuf,

    /// Explicit tokenizer path used by the CPU receipts.
    #[arg(long, value_name = "PATH")]
    pub tokenizer: Option<PathBuf>,

    /// Optional prompt-authority audit artifact for the same model/prompt policy.
    #[arg(long, value_name = "PATH")]
    pub prompt_audit: Option<PathBuf>,

    /// Optional scalar CPU answer-corpus receipt with first-step top-k evidence.
    #[arg(long, value_name = "PATH")]
    pub scalar_answer_corpus: Option<PathBuf>,

    /// Optional AVX2 CPU answer-corpus receipt with first-step top-k evidence.
    #[arg(long, value_name = "PATH")]
    pub avx2_answer_corpus: Option<PathBuf>,

    /// Output audit receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/output-head-logits-index-audit.json"
    )]
    pub json_out: PathBuf,
}

impl OutputHeadLogitsAuditCommand {
    /// Execute the offline output-head/logits-index audit.
    pub async fn execute(&self) -> Result<()> {
        let model_sha256 = compute_sha256_file(&self.model)?;
        let model_bytes = fs::read(&self.model)
            .with_context(|| format!("failed to read model {}", self.model.display()))?;
        let gguf = GgufReader::new(&model_bytes).context("failed to parse GGUF")?;

        let tokenizer_resolution = bitnet_tokenizers::auto::resolve_tokenizer(
            &self.model,
            self.tokenizer.as_deref(),
            true,
        )?;
        let tokenizer_source = tokenizer_resolution.source;
        let tokenizer_path = tokenizer_resolution.path.clone();
        let tokenizer = tokenizer_resolution.tokenizer;

        let prompt_audit = self.prompt_audit.as_deref().map(read_json).transpose()?;
        let scalar = self.scalar_answer_corpus.as_deref().map(read_json).transpose()?;
        let avx2 = self.avx2_answer_corpus.as_deref().map(read_json).transpose()?;

        let receipt = build_output_head_logits_audit_receipt(
            &self.model,
            &model_sha256,
            tokenizer_source,
            tokenizer_path.as_deref(),
            tokenizer.as_ref(),
            &gguf,
            self.prompt_audit.as_deref(),
            prompt_audit.as_ref(),
            self.scalar_answer_corpus.as_deref(),
            scalar.as_ref(),
            self.avx2_answer_corpus.as_deref(),
            avx2.as_ref(),
        );

        if let Some(parent) = self.json_out.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.json_out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("output-head/logits-index audit written to {}", self.json_out.display());

        if receipt["validation"]["passed"].as_bool() != Some(true) {
            anyhow::bail!(
                "output-head/logits-index audit failed validation; receipt written to {}",
                self.json_out.display()
            );
        }
        Ok(())
    }
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn compute_sha256_file(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[allow(clippy::too_many_arguments)]
fn build_output_head_logits_audit_receipt(
    model_path: &Path,
    model_sha256: &str,
    tokenizer_source: bitnet_tokenizers::auto::TokenizerSource,
    tokenizer_path: Option<&Path>,
    tokenizer: &(dyn Tokenizer + Send + Sync),
    gguf: &GgufReader<'_>,
    prompt_audit_path: Option<&Path>,
    prompt_audit: Option<&Value>,
    scalar_path: Option<&Path>,
    scalar: Option<&Value>,
    avx2_path: Option<&Path>,
    avx2: Option<&Value>,
) -> Value {
    let architecture = gguf.get_string_metadata("general.architecture");
    let metadata_vocab_size = gguf_metadata_vocab_size(gguf, architecture.as_deref());
    let tokenizer_vocab_size = tokenizer.real_vocab_size();
    let embedding_name = first_present_tensor(gguf, EMBEDDING_CANDIDATES);
    let output_head_name = first_present_tensor(gguf, OUTPUT_HEAD_CANDIDATES);
    let raw_qk256_name = first_present_tensor(gguf, RAW_QK256_TIED_CANDIDATES);
    let selected_embedding = embedding_name.and_then(|name| tensor_descriptor(gguf, name));
    let selected_output_head = output_head_name.and_then(|name| tensor_descriptor(gguf, name));
    let raw_qk256_tied_candidate = raw_qk256_name.and_then(|name| tensor_descriptor(gguf, name));
    let hidden_size = gguf_hidden_size(gguf, architecture.as_deref()).or_else(|| {
        embedding_name
            .and_then(|name| gguf.get_tensor_info_by_name(name))
            .and_then(|info| infer_hidden_size_from_embedding(info, tokenizer_vocab_size))
    });
    let expected_logits_vector_length =
        metadata_vocab_size.map(|value| value as usize).unwrap_or(tokenizer_vocab_size);
    let tied_output_policy = tied_output_policy(output_head_name, raw_qk256_name, embedding_name);
    let output_orientation =
        output_head_name.and_then(|name| gguf.get_tensor_info_by_name(name)).map(|info| {
            output_orientation(info, metadata_vocab_size.map(|value| value as usize), hidden_size)
        });
    let special_tokens = special_tokens_json(tokenizer, tokenizer_vocab_size);
    let scalar_evidence = answer_corpus_evidence("scalar", scalar_path, scalar, tokenizer);
    let avx2_evidence = answer_corpus_evidence("avx2", avx2_path, avx2, tokenizer);
    let topk_comparison = compare_topk_evidence(&scalar_evidence, &avx2_evidence);
    let validation_failures =
        validate_boundary(embedding_name, output_head_name, raw_qk256_name, tokenizer_vocab_size);
    let classification = classify_boundary(
        metadata_vocab_size,
        tokenizer_vocab_size,
        &topk_comparison,
        &validation_failures,
    );

    json!({
        "schema_version": "1.0.0",
        "artifact_kind": "bitnet_output_head_logits_index_audit",
        "machine_id": "intel-258v",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "proof_stage": "output_head_logits_index_audited",
        "claim": "cpu258v_output_head_logits_index_boundary",
        "inputs": {
            "model": model_path.display().to_string(),
            "tokenizer": tokenizer_path.map(|path| path.display().to_string()),
            "prompt_audit": prompt_audit_path.map(|path| path.display().to_string()),
            "scalar_answer_corpus": scalar_path.map(|path| path.display().to_string()),
            "avx2_answer_corpus": avx2_path.map(|path| path.display().to_string()),
        },
        "validation": {
            "passed": validation_failures.is_empty(),
            "failed_rules": validation_failures,
        },
        "model": {
            "path": model_path.display().to_string(),
            "sha256": model_sha256,
            "gguf_architecture": architecture,
            "gguf_name": gguf.get_string_metadata("general.name"),
            "n_tensors": gguf.tensor_count(),
            "n_kv": gguf.metadata_keys().len(),
            "metadata_vocab_size": metadata_vocab_size,
            "metadata_hidden_size": hidden_size,
        },
        "tokenizer": {
            "source": tokenizer_source.as_str(),
            "path": tokenizer_path.map(|path| path.display().to_string()),
            "vocab_size": tokenizer.vocab_size(),
            "real_vocab_size": tokenizer_vocab_size,
            "bos_token_id": tokenizer.bos_token_id(),
            "eos_token_id": tokenizer.eos_token_id(),
            "pad_token_id": tokenizer.pad_token_id(),
            "eot_token_id": tokenizer.token_to_id("<|eot_id|>"),
            "end_of_text_token_id": tokenizer.token_to_id("<|end_of_text|>"),
        },
        "prompt_authority": prompt_authority_summary(prompt_audit),
        "tensor_boundary": {
            "embedding_candidates": tensor_candidates_json(gguf, EMBEDDING_CANDIDATES),
            "selected_embedding": selected_embedding,
            "output_head_candidates": tensor_candidates_json(gguf, OUTPUT_HEAD_CANDIDATES),
            "selected_output_head": selected_output_head,
            "raw_qk256_tied_candidates": tensor_candidates_json(gguf, RAW_QK256_TIED_CANDIDATES),
            "raw_qk256_tied_candidate": raw_qk256_tied_candidate,
            "tied_output_policy": tied_output_policy,
            "output_orientation": output_orientation,
        },
        "logits_index_boundary": {
            "expected_logits_vector_length": expected_logits_vector_length,
            "expected_logits_vector_length_source": if metadata_vocab_size.is_some() { "gguf_metadata_vocab_size" } else { "tokenizer_real_vocab_size" },
            "tokenizer_real_vocab_size": tokenizer_vocab_size,
            "metadata_vocab_size": metadata_vocab_size,
            "metadata_vocab_matches_tokenizer": metadata_vocab_size.map(|value| value as usize == tokenizer_vocab_size),
            "observed_logits_vector_length": observed_logits_vector_length(&scalar_evidence, &avx2_evidence),
            "observed_logits_vector_length_source": observed_logits_vector_length_source(&scalar_evidence, &avx2_evidence),
            "special_tokens": special_tokens,
        },
        "topk_evidence": {
            "scalar": scalar_evidence,
            "avx2": avx2_evidence,
            "scalar_avx2_comparison": topk_comparison,
        },
        "classification": classification,
        "fallback_used": false,
        "claim_boundary": {
            "may_claim": [
                "The 258V CPU output-head or tied-head tensor boundary is recorded from GGUF metadata.",
                "The tokenizer/logits index contract is audited against vocab and special-token IDs.",
                "Existing scalar and AVX2 first-step top-k token IDs can be decoded and compared when supplied."
            ],
            "must_not_claim": [
                "BitNet answer quality is newly proven.",
                "First-token logits parity with the external reference is proven.",
                "CPU speed or sustained throughput is proven.",
                "Arc 140V or Intel NPU execution is proven.",
                "Full model correctness is proven."
            ]
        }
    })
}

fn tensor_descriptor(gguf: &GgufReader<'_>, name: &str) -> Option<Value> {
    gguf.get_tensor_info_by_name(name).map(|info| {
        json!({
            "name": info.name,
            "shape": info.shape,
            "tensor_type": format!("{:?}", info.tensor_type),
            "offset": info.offset,
            "size_bytes": info.size,
        })
    })
}

fn tensor_candidates_json(gguf: &GgufReader<'_>, names: &[&str]) -> Vec<Value> {
    names
        .iter()
        .map(|name| {
            if let Some(info) = gguf.get_tensor_info_by_name(name) {
                json!({
                    "name": name,
                    "present": true,
                    "shape": info.shape,
                    "tensor_type": format!("{:?}", info.tensor_type),
                    "offset": info.offset,
                    "size_bytes": info.size,
                })
            } else {
                json!({
                    "name": name,
                    "present": false,
                })
            }
        })
        .collect()
}

fn first_present_tensor<'a>(gguf: &GgufReader<'_>, names: &'a [&'a str]) -> Option<&'a str> {
    names.iter().copied().find(|name| gguf.get_tensor_info_by_name(name).is_some())
}

fn gguf_metadata_vocab_size(gguf: &GgufReader<'_>, architecture: Option<&str>) -> Option<u32> {
    architecture
        .and_then(|arch| gguf.get_u32_metadata(&format!("{arch}.vocab_size")))
        .or_else(|| gguf.get_u32_metadata("llama.vocab_size"))
        .or_else(|| gguf.get_u32_metadata("bitnet_b1_58.vocab_size"))
        .or_else(|| gguf.get_u32_metadata("tokenizer.ggml.vocab_size"))
}

fn gguf_hidden_size(gguf: &GgufReader<'_>, architecture: Option<&str>) -> Option<usize> {
    architecture
        .and_then(|arch| gguf.get_u32_metadata(&format!("{arch}.embedding_length")))
        .or_else(|| gguf.get_u32_metadata("llama.embedding_length"))
        .or_else(|| gguf.get_u32_metadata("bitnet_b1_58.embedding_length"))
        .map(|value| value as usize)
}

fn infer_hidden_size_from_embedding(info: &TensorInfo, vocab_size: usize) -> Option<usize> {
    if info.shape.len() != 2 {
        return None;
    }
    match (info.shape[0] == vocab_size, info.shape[1] == vocab_size) {
        (true, false) => Some(info.shape[1]),
        (false, true) => Some(info.shape[0]),
        _ => None,
    }
}

fn output_orientation(
    info: &TensorInfo,
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
) -> Value {
    let classification = classify_output_orientation(&info.shape, vocab_size, hidden_size);
    json!({
        "tensor": info.name,
        "shape": info.shape,
        "classification": classification,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
    })
}

fn classify_output_orientation(
    shape: &[usize],
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
) -> &'static str {
    if shape.len() != 2 {
        return "not_2d";
    }
    let Some(vocab) = vocab_size else {
        return "unknown_vocab_size";
    };
    match hidden_size {
        Some(hidden) if shape == [vocab, hidden] => "vocab_hidden",
        Some(hidden) if shape == [hidden, vocab] => "hidden_vocab_transposed",
        Some(_) => "unexpected_shape_for_vocab_hidden",
        None if shape[0] == vocab => "vocab_first_hidden_unknown",
        None if shape[1] == vocab => "vocab_second_hidden_unknown",
        None => "unexpected_shape_for_vocab",
    }
}

fn tied_output_policy(
    output_head_name: Option<&str>,
    raw_qk256_name: Option<&str>,
    embedding_name: Option<&str>,
) -> &'static str {
    if output_head_name.is_some() {
        "dedicated_output_head"
    } else if raw_qk256_name.is_some() {
        "raw_qk256_tied_token_embeddings_candidate"
    } else if embedding_name.is_some() {
        "tied_token_embeddings"
    } else {
        "missing_embedding_and_output_head"
    }
}

fn special_tokens_json(tokenizer: &(dyn Tokenizer + Send + Sync), vocab_size: usize) -> Value {
    let ids = [
        ("bos", tokenizer.bos_token_id()),
        ("eos", tokenizer.eos_token_id()),
        ("pad", tokenizer.pad_token_id()),
        ("eot", tokenizer.token_to_id("<|eot_id|>")),
        ("end_of_text", tokenizer.token_to_id("<|end_of_text|>")),
    ];
    let entries = ids
        .iter()
        .map(|(name, id)| {
            json!({
                "name": name,
                "token_id": id,
                "within_vocab": id.map(|value| (value as usize) < vocab_size),
                "piece": id.and_then(|value| tokenizer.token_to_piece(value)),
                "decoded": id.and_then(|value| tokenizer.decode(&[value]).ok()),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "entries": entries,
        "all_present_ids_within_vocab": ids
            .iter()
            .filter_map(|(_, id)| *id)
            .all(|id| (id as usize) < vocab_size),
    })
}

fn prompt_authority_summary(prompt_audit: Option<&Value>) -> Value {
    let Some(audit) = prompt_audit else {
        return json!({
            "provided": false,
        });
    };
    json!({
        "provided": true,
        "artifact_kind": audit["artifact_kind"].clone(),
        "classification": audit["classification"].clone(),
        "model": audit["model"].clone(),
        "tokenizer": audit["tokenizer"].clone(),
        "reference_parity": audit["reference_parity"].clone(),
    })
}

fn answer_corpus_evidence(
    label: &str,
    path: Option<&Path>,
    receipt: Option<&Value>,
    tokenizer: &(dyn Tokenizer + Send + Sync),
) -> Value {
    let Some(receipt) = receipt else {
        return json!({
            "provided": false,
            "label": label,
            "path": path.map(|p| p.display().to_string()),
        });
    };

    let cases = receipt["cases"]
        .as_array()
        .map(|items| {
            items.iter().map(|case| answer_case_evidence(case, tokenizer)).collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let cases_with_topk = cases
        .iter()
        .filter(|case| {
            case["first_step_top_logits"].as_array().is_some_and(|items| !items.is_empty())
        })
        .count();
    let observed_lengths = cases
        .iter()
        .filter_map(|case| case["observed_logits_vector_length"].as_u64())
        .collect::<Vec<_>>();

    json!({
        "provided": true,
        "label": label,
        "path": path.map(|p| p.display().to_string()),
        "artifact_kind": receipt["artifact_kind"].clone(),
        "backend": receipt["backend"].clone(),
        "summary": {
            "cases_total": cases.len(),
            "cases_with_first_step_topk": cases_with_topk,
            "observed_logits_vector_lengths": observed_lengths,
        },
        "cases": cases,
    })
}

fn answer_case_evidence(case: &Value, tokenizer: &(dyn Tokenizer + Send + Sync)) -> Value {
    let first_step = case["logits_dump"].as_array().and_then(|steps| steps.first());
    let top_logits = first_step
        .and_then(|step| step["top_logits"].as_array())
        .map(|items| {
            items.iter().filter_map(|item| top_logit_entry(item, tokenizer)).collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let observed_length = case["logits_index_boundary"]["first_step_logits_vector_length"]
        .as_u64()
        .or_else(|| case["logits_index_boundary"]["observed_logits_vector_length"].as_u64())
        .or_else(|| case["model"]["vocab_size"].as_u64());

    json!({
        "id": case["id"].clone(),
        "status": case["status"].clone(),
        "selected_kernel": case["kernel"]["selected_kernel"].clone(),
        "model_vocab_size": case["model"]["vocab_size"].clone(),
        "output_head_tensor": case["model"]["output_head_tensor"].clone(),
        "tie_word_embeddings": case["model"]["tie_word_embeddings"].clone(),
        "observed_logits_vector_length": observed_length,
        "observed_logits_vector_length_source": if case["logits_index_boundary"].is_object() {
            "run_receipt_logits_index_boundary"
        } else if case["model"]["vocab_size"].is_number() {
            "answer_corpus_model_vocab_size"
        } else {
            "not_available"
        },
        "chosen_id": first_step.and_then(|step| step["chosen_id"].as_u64()),
        "first_step_top_logits": top_logits,
    })
}

fn top_logit_entry(item: &Value, tokenizer: &(dyn Tokenizer + Send + Sync)) -> Option<Value> {
    let token_id = item["token_id"].as_u64()?;
    let token_id_u32 = u32::try_from(token_id).ok()?;
    Some(json!({
        "token_id": token_id,
        "logit": item["logit"].clone(),
        "piece": tokenizer.token_to_piece(token_id_u32),
        "decoded": tokenizer.decode(&[token_id_u32]).ok(),
        "is_special": tokenizer.is_special_token(token_id_u32),
    }))
}

fn compare_topk_evidence(scalar: &Value, avx2: &Value) -> Value {
    if scalar["provided"].as_bool() != Some(true) || avx2["provided"].as_bool() != Some(true) {
        return json!({
            "available": false,
            "reason": "scalar_or_avx2_answer_corpus_not_supplied",
        });
    }
    let scalar_by_case = topk_ids_by_case(scalar);
    let avx2_by_case = topk_ids_by_case(avx2);
    let cases = scalar_by_case
        .keys()
        .chain(avx2_by_case.keys())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|case_id| {
            let scalar_ids = scalar_by_case.get(case_id).cloned().unwrap_or_default();
            let avx2_ids = avx2_by_case.get(case_id).cloned().unwrap_or_default();
            json!({
                "id": case_id,
                "scalar_topk_ids": scalar_ids,
                "avx2_topk_ids": avx2_ids,
                "topk_ids_match": scalar_ids == avx2_ids,
            })
        })
        .collect::<Vec<_>>();
    let cases_compared = cases.len();
    let mismatches =
        cases.iter().filter(|case| case["topk_ids_match"].as_bool() != Some(true)).count();

    json!({
        "available": true,
        "cases_compared": cases_compared,
        "mismatches": mismatches,
        "topk_ids_all_match": cases_compared > 0 && mismatches == 0,
        "cases": cases,
    })
}

fn topk_ids_by_case(evidence: &Value) -> BTreeMap<String, Vec<u64>> {
    let mut by_case = BTreeMap::new();
    if let Some(cases) = evidence["cases"].as_array() {
        for case in cases {
            if let Some(id) = case["id"].as_str() {
                let ids = case["first_step_top_logits"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| item["token_id"].as_u64())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                by_case.insert(id.to_string(), ids);
            }
        }
    }
    by_case
}

fn observed_logits_vector_length(scalar: &Value, avx2: &Value) -> Option<u64> {
    [scalar, avx2].iter().find_map(|evidence| {
        evidence["cases"]
            .as_array()?
            .iter()
            .find_map(|case| case["observed_logits_vector_length"].as_u64())
    })
}

fn observed_logits_vector_length_source(scalar: &Value, avx2: &Value) -> String {
    [scalar, avx2]
        .iter()
        .find_map(|evidence| {
            evidence["cases"]
                .as_array()?
                .iter()
                .find_map(|case| case["observed_logits_vector_length_source"].as_str())
        })
        .unwrap_or("not_available")
        .to_string()
}

fn validate_boundary(
    embedding_name: Option<&str>,
    output_head_name: Option<&str>,
    raw_qk256_name: Option<&str>,
    tokenizer_vocab_size: usize,
) -> Vec<&'static str> {
    let mut failures = Vec::new();
    if tokenizer_vocab_size == 0 {
        failures.push("tokenizer_vocab_size");
    }
    if embedding_name.is_none() {
        failures.push("embedding_tensor_present");
    }
    if output_head_name.is_none() && raw_qk256_name.is_none() && embedding_name.is_none() {
        failures.push("output_head_or_tied_embedding_policy");
    }
    failures
}

fn classify_boundary(
    metadata_vocab_size: Option<u32>,
    tokenizer_vocab_size: usize,
    topk_comparison: &Value,
    validation_failures: &[&str],
) -> Value {
    let mut notes = Vec::new();
    if let Some(metadata_vocab_size) = metadata_vocab_size {
        if metadata_vocab_size as usize != tokenizer_vocab_size {
            notes.push("metadata_vocab_size_differs_from_tokenizer_real_vocab_size");
        }
    } else {
        notes.push("gguf_metadata_vocab_size_missing");
    }
    if topk_comparison["available"].as_bool() == Some(true)
        && topk_comparison["topk_ids_all_match"].as_bool() != Some(true)
    {
        notes.push("scalar_avx2_first_step_topk_ids_differ");
    }
    if validation_failures.is_empty() && notes.is_empty() {
        json!({
            "classification": "output_head_logits_index_boundary_recorded",
            "first_mismatch_stage": null,
            "notes": notes,
        })
    } else {
        let first_mismatch_stage = if !validation_failures.is_empty() {
            "tensor_boundary"
        } else if notes.contains(&"metadata_vocab_size_differs_from_tokenizer_real_vocab_size") {
            "vocab_index_contract"
        } else if notes.contains(&"scalar_avx2_first_step_topk_ids_differ") {
            "local_topk_contract"
        } else {
            "metadata_gap"
        };
        json!({
            "classification": "output_head_logits_index_boundary_has_gaps",
            "first_mismatch_stage": first_mismatch_stage,
            "notes": notes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_tokenizers::MockTokenizer;

    #[test]
    fn output_orientation_detects_vocab_hidden_shapes() {
        assert_eq!(
            classify_output_orientation(&[128256, 2560], Some(128256), Some(2560)),
            "vocab_hidden"
        );
        assert_eq!(
            classify_output_orientation(&[2560, 128256], Some(128256), Some(2560)),
            "hidden_vocab_transposed"
        );
        assert_eq!(
            classify_output_orientation(&[128256, 4096], Some(128256), Some(2560)),
            "unexpected_shape_for_vocab_hidden"
        );
    }

    #[test]
    fn tied_policy_prefers_dedicated_then_raw_qk256_then_embedding() {
        assert_eq!(
            tied_output_policy(
                Some("output.weight"),
                Some("embed_tokens.weight.qk256_qs"),
                Some("token_embd.weight")
            ),
            "dedicated_output_head"
        );
        assert_eq!(
            tied_output_policy(
                None,
                Some("embed_tokens.weight.qk256_qs"),
                Some("token_embd.weight")
            ),
            "raw_qk256_tied_token_embeddings_candidate"
        );
        assert_eq!(
            tied_output_policy(None, None, Some("token_embd.weight")),
            "tied_token_embeddings"
        );
    }

    #[test]
    fn topk_evidence_decodes_token_ids_and_compares_cases() {
        let tokenizer = MockTokenizer::new();
        let receipt = json!({
            "artifact_kind": "bitnet_cpu_answer_corpus",
            "cases": [{
                "id": "math",
                "status": "passed",
                "model": {
                    "vocab_size": 50257,
                    "output_head_tensor": "tied_token_embeddings",
                    "tie_word_embeddings": true
                },
                "kernel": {
                    "selected_kernel": "i2_s-scalar-reference"
                },
                "logits_dump": [{
                    "chosen_id": 52,
                    "top_logits": [
                        {"token_id": 52, "logit": 1.0},
                        {"token_id": 53, "logit": 0.5}
                    ]
                }]
            }]
        });

        let scalar = answer_corpus_evidence("scalar", None, Some(&receipt), &tokenizer);
        let avx2 = answer_corpus_evidence("avx2", None, Some(&receipt), &tokenizer);
        assert_eq!(scalar["cases"][0]["first_step_top_logits"][0]["decoded"], "4");
        assert_eq!(scalar["cases"][0]["first_step_top_logits"][0]["piece"], "4");
        assert_eq!(compare_topk_evidence(&scalar, &avx2)["topk_ids_all_match"], true);
    }
}
