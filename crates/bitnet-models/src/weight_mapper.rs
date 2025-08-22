use bitnet_common::Result;
use candle_core::{DType, Device, Tensor};
/// Weight mapping utilities for loading model weights from various formats
use std::borrow::Cow;
use std::collections::HashMap;

/// Map GGUF tensor names to transformer module names
pub fn remap_gguf_weights(tensors: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    remap_gguf_weights_with_options(tensors, false)
}

/// Normalize exporter name drift to our canonical names.
/// Known drifts:
///  - attn_sub_norm <-> attention_sub_norm  
///  - ffn_sub_norm  <-> mlp_sub_layernorm
fn normalize_name(name: &str) -> Cow<'_, str> {
    if name.contains("attention_sub_norm") {
        // Map Microsoft's variation to our canonical name
        let s = name.replace("attention_sub_norm", "attn_sub_norm");
        return Cow::Owned(s);
    }
    if name.contains("mlp_sub_layernorm") {
        // Map to our canonical FFN sub norm
        let s = name.replace("mlp_sub_layernorm", "ffn_sub_norm");
        return Cow::Owned(s);
    }
    Cow::Borrowed(name)
}

/// Helper to get 2D tensor dimensions
fn dims2(tensor: &Tensor, name: &str) -> Result<(usize, usize)> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(bitnet_common::BitNetError::Validation(format!(
            "{} must be 2D, got {:?}",
            name, dims
        )));
    }
    Ok((dims[0], dims[1]))
}

/// Find a tensor by trying multiple aliases
fn pick<'a>(tensors: &'a HashMap<String, Tensor>, candidates: &[&str]) -> Option<&'a Tensor> {
    for k in candidates {
        if let Some(t) = tensors.get(*k) {
            return Some(t);
        }
    }
    None
}

/// Map GGUF tensor names to transformer module names with strict option
pub fn remap_gguf_weights_with_options(tensors: &HashMap<String, Tensor>, strict: bool) -> Result<HashMap<String, Tensor>> {
    let mut mapped = HashMap::new();
    let mut unmapped = Vec::new();

    // First pass: map all tensors
    for (name, tensor) in tensors {
        // First normalize any known name variations
        let normalized = normalize_name(name);
        let new_name = if let Some(mapped_name) = map_tensor_name(&normalized) {
            mapped_name
        } else {
            unmapped.push(name.clone());
            name.clone()
        };

        mapped.insert(new_name, tensor.clone());
    }

    // Handle unmapped tensors
    if !unmapped.is_empty() {
        if strict {
            return Err(bitnet_common::BitNetError::Validation(format!(
                "Strict mapping mode: {} unmapped tensors found: {:?}",
                unmapped.len(),
                &unmapped[..5.min(unmapped.len())]
            )));
        } else {
            eprintln!(
                "Warning: {} unmapped tensors: {:?}",
                unmapped.len(),
                &unmapped[..5.min(unmapped.len())]
            );
        }
    }

    // Check if we have lm_head
    let has_lm_head = mapped.contains_key("lm_head.weight");
    let has_embed = mapped.contains_key("embed_tokens.weight");
    tracing::info!(
        "Mapped tensors: has lm_head.weight={}, has embed_tokens.weight={}",
        has_lm_head,
        has_embed
    );

    // If no lm_head but we have embeddings, that's OK (tied weights)
    if !has_lm_head && has_embed {
        tracing::info!("No lm_head.weight found, will use tied weights with embed_tokens");
    }

    Ok(mapped)
}

/// Map individual tensor name from GGUF to our transformer naming
fn map_tensor_name(name: &str) -> Option<String> {
    // Token embeddings variations
    if name == "token_embd.weight"
        || name == "tok_embeddings.weight"
        || name == "model.embed_tokens.weight"
    {
        return Some("embed_tokens.weight".to_string());
    }

    // Output layer variations
    if name == "output.weight" || name == "lm_head.weight" || name == "model.lm_head.weight" {
        return Some("lm_head.weight".to_string());
    }

    // Final normalization
    if name == "output_norm.weight" || name == "norm.weight" || name == "model.norm.weight" {
        return Some("final_norm.weight".to_string());
    }

    // Handle "blk.N." prefix (common in GGUF)
    if name.starts_with("blk.") {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() >= 3 {
            let layer_num = parts[1];
            let component = parts[2..].join(".");

            let mapped_component = match component.as_str() {
                // Attention weights
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" | "attn_o.weight" => "self_attn.o_proj.weight",

                // Attention normalization
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_sub_norm.weight" => "self_attn.sub_layernorm.weight",  // BitNet specific

                // Feed-forward weights
                "ffn_gate.weight" | "ffn_gate_inp.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" | "ffn_up_proj.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" | "ffn_down_proj.weight" => "mlp.down_proj.weight",

                // FFN normalization
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_sub_norm.weight" => "mlp.sub_layernorm.weight",  // BitNet specific

                _ => return None,
            };

            return Some(format!("layers.{}.{}", layer_num, mapped_component));
        }
    }

    // Handle "layers.N." prefix (LLaMA style)
    if name.starts_with("layers.") || name.starts_with("model.layers.") {
        let clean_name = name.strip_prefix("model.").unwrap_or(name);

        let parts: Vec<&str> = clean_name.split('.').collect();
        if parts.len() >= 3 && parts[0] == "layers" {
            let layer_num = parts[1];
            let component = parts[2..].join(".");

            let mapped_component = match component.as_str() {
                // LLaMA-style attention
                "attention.wq.weight" | "self_attn.q_proj.weight" => "self_attn.q_proj.weight",
                "attention.wk.weight" | "self_attn.k_proj.weight" => "self_attn.k_proj.weight",
                "attention.wv.weight" | "self_attn.v_proj.weight" => "self_attn.v_proj.weight",
                "attention.wo.weight" | "self_attn.o_proj.weight" => "self_attn.o_proj.weight",

                // Normalization
                "attention_norm.weight" | "input_layernorm.weight" => "input_layernorm.weight",
                "ffn_norm.weight" | "post_attention_layernorm.weight" => {
                    "post_attention_layernorm.weight"
                }

                // LLaMA-style FFN
                "feed_forward.w1.weight" | "mlp.gate_proj.weight" => "mlp.gate_proj.weight",
                "feed_forward.w3.weight" | "mlp.up_proj.weight" => "mlp.up_proj.weight",
                "feed_forward.w2.weight" | "mlp.down_proj.weight" => "mlp.down_proj.weight",

                _ => return Some(format!("layers.{}.{}", layer_num, component)),
            };

            return Some(format!("layers.{}.{}", layer_num, mapped_component));
        }
    }

    None
}

/// Dry-run tensor name mapping for testing without loading actual tensors
/// Returns list of unmapped tensor names
pub fn dry_run_remap_names<I>(names: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut unmapped = Vec::new();
    for name in names {
        if map_tensor_name(&name).is_none() {
            unmapped.push(name);
        }
    }
    unmapped
}

/// Detect vocab size and normalize embedding/lm_head tensors
/// Returns (vocab_size, actual_hidden_size)
pub fn normalize_model_tensors(
    tensors: &mut HashMap<String, Tensor>,
    expected_hidden_size: usize,
) -> Result<(usize, usize)> {
    // 1) Locate embedding with robust aliases
    let emb_candidates = [
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "tok_embeddings.weight",
        "token_embd.weight",
        "transformer.wte.weight",
    ];
    
    let emb_key = emb_candidates.iter()
        .find(|k| tensors.contains_key(**k))
        .ok_or_else(|| bitnet_common::BitNetError::Validation(
            "embed tokens not found (tried embed_tokens/tok_embeddings/token_embd/transformer.wte)".to_string()
        ))?;
    
    let emb = tensors.get(*emb_key).unwrap();
    
    // 2) Infer vocab + orientation from the embedding shape
    let (er, ec) = dims2(emb, "embed_tokens.weight")?;
    tracing::info!(
        "Embedding tensor shape: [{}, {}], expected_hidden_size: {}",
        er, ec, expected_hidden_size
    );
    
    // Detect actual hidden size and vocab from tensor
    let (vocab_size, hidden_size, emb_needs_t) = if er > ec {
        // Likely [vocab, hidden]
        (er, ec, false)
    } else {
        // Likely [hidden, vocab]
        (ec, er, true)
    };
    
    // Warn if detected hidden size doesn't match expected
    if hidden_size != expected_hidden_size && expected_hidden_size != 0 {
        tracing::warn!(
            "Detected hidden_size {} from tensor shape differs from expected {}",
            hidden_size, expected_hidden_size
        );
    };
    
    // 3) Normalize embedding to [n_vocab, n_embd]
    if emb_needs_t {
        let emb_norm = emb.t()?.contiguous()?;
        tracing::info!(
            "embed_tokens normalized -> [vocab={}, hidden={}], transposed=true",
            vocab_size, hidden_size
        );
        tensors.insert("embed_tokens.weight".to_string(), emb_norm);
        // Remove old key if different
        if emb_key != &"embed_tokens.weight" {
            tensors.remove(*emb_key);
        }
    } else if emb_key != &"embed_tokens.weight" {
        // Just rename the key if needed
        let emb = tensors.remove(*emb_key).unwrap();
        tensors.insert("embed_tokens.weight".to_string(), emb);
    }
    
    // 4) Locate lm_head with robust aliases, normalize to [n_vocab, n_embd]
    let lm_candidates = [
        "lm_head.weight",
        "output.weight",
        "model.lm_head.weight",
        "generator.weight",
    ];
    
    if let Some(lm_key) = lm_candidates.iter().find(|k| tensors.contains_key(**k)) {
        let lm = tensors.get(*lm_key).unwrap();
        let (lr, lc) = dims2(lm, "lm_head.weight")?;
        
        let lm_needs_t = match (lr, lc) {
            (v, h) if v == vocab_size && h == hidden_size => false,
            (h, v) if h == hidden_size && v == vocab_size => {
                tracing::info!("lm_head appears transposed; normalizing.");
                true
            }
            _ => {
                return Err(bitnet_common::BitNetError::Validation(format!(
                    "lm_head.weight bad shape [{},{}], want [{},{}] or transposed",
                    lr, lc, vocab_size, hidden_size
                )));
            }
        };
        
        if lm_needs_t {
            let lm_norm = lm.t()?.contiguous()?;
            tensors.insert("lm_head.weight".to_string(), lm_norm);
            if lm_key != &"lm_head.weight" {
                tensors.remove(*lm_key);
            }
        } else if lm_key != &"lm_head.weight" {
            let lm = tensors.remove(*lm_key).unwrap();
            tensors.insert("lm_head.weight".to_string(), lm);
        }
    } else {
        tracing::info!("No lm_head.weight found; using tied weights with embed_tokens.");
    }
    
    Ok((vocab_size, hidden_size))
}

/// Create a VarBuilder from mapped tensors
pub fn create_var_builder(
    tensors: HashMap<String, Tensor>,
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'_>> {
    // Convert tensors to the target dtype if needed
    let mut converted = HashMap::new();
    for (name, tensor) in tensors {
        tracing::trace!(
            "Processing tensor {}: shape={:?}, dtype={:?}",
            name,
            tensor.shape(),
            tensor.dtype()
        );

        let tensor = if tensor.dtype() != dtype {
            tracing::trace!("Converting {} from {:?} to {:?}", name, tensor.dtype(), dtype);
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };

        // Move to target device if needed
        let tensor = if !tensor.device().same_device(device) {
            tracing::trace!("Moving {} to device", name);
            tensor.to_device(device)?
        } else {
            tensor
        };

        converted.insert(name, tensor);
    }

    Ok(candle_nn::VarBuilder::from_tensors(converted, dtype, device))
}
