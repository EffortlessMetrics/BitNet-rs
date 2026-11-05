use bitnet_common::Result;
use candle_core::{DType, Device, Tensor};
// Weight mapping utilities for loading model weights from various formats
use std::borrow::Cow;
use std::collections::HashMap;

/// Model dimensions for tensor shape validation and transposition
#[derive(Clone, Copy, Debug)]
struct ModelDims {
    hidden: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    inter: usize,
    vocab: usize,
}

impl ModelDims {
    fn head_dim(&self) -> Result<usize> {
        Ok(self.head_dim)
    }

    fn kv_head_dim(&self) -> Result<usize> {
        self.head_dim()
    }

    fn q_dim(&self) -> Result<usize> {
        Ok(self.head_dim()? * self.n_head)
    }

    fn kv_dim(&self) -> Result<usize> {
        Ok(self.kv_head_dim()? * self.n_kv_head)
    }
}

/// Ensures tensor is [out, in] by transposing if it's [in, out].
/// Accepts fused alternative shapes where applicable (e.g., [hidden, hidden]).
fn ensure_matrix_or_transpose(
    t: Tensor,
    expected_out: usize,
    expected_in: usize,
    name: &str,
) -> Result<Tensor> {
    let shp = t.shape().dims();
    match shp {
        // already correct [out, in]
        [o, i] if *o == expected_out && *i == expected_in => {
            tracing::trace!("{}: already correct [{}, {}]", name, o, i);
            Ok(t)
        }
        // transposed [in, out] - need to transpose
        [i, o] if *i == expected_in && *o == expected_out => {
            tracing::info!(
                "{}: transposing from [{}, {}] to [{}, {}]",
                name,
                i,
                o,
                expected_out,
                expected_in
            );
            let transposed = t.t()?;
            let res = match transposed.contiguous() {
                Ok(t2) => t2,
                Err(_) => {
                    tracing::warn!(
                        "{}: Tensor transpose contiguous failed, falling back to F32 cast",
                        name
                    );
                    t.to_dtype(candle_core::DType::F32)?.t()?.contiguous()?
                }
            };
            Ok(res)
        }
        // allow fused hidden x hidden in cases where both dims equal hidden
        [o, i] if *o == expected_out && *i == expected_out && expected_out == expected_in => {
            tracing::trace!("{}: fused square shape [{}, {}]", name, o, i);
            Ok(t)
        }
        [i, o] if *i == expected_out && *o == expected_out && expected_out == expected_in => {
            tracing::info!(
                "{}: fused square shape but transposed, transposing [{}, {}]",
                name,
                i,
                o
            );
            let transposed = t.t()?;
            let res = match transposed.contiguous() {
                Ok(t2) => t2,
                Err(_) => {
                    tracing::warn!(
                        "{}: Tensor transpose contiguous failed, falling back to F32 cast",
                        name
                    );
                    t.to_dtype(candle_core::DType::F32)?.t()?.contiguous()?
                }
            };
            Ok(res)
        }
        _ => Err(bitnet_common::BitNetError::Validation(format!(
            "{}: unexpected matrix shape {:?}, expected [{}, {}] or its transpose",
            name, shp, expected_out, expected_in
        ))),
    }
}

/// Returns canonical key if `k` matches a known vendor pattern.
///
/// Canonical target schema:
/// `layers.{i}.attention.{q_proj|k_proj|v_proj|o_proj}.weight`
/// `layers.{i}.feed_forward.{gate_proj|up_proj|down_proj}.weight`
/// `layers.{i}.attention_norm.weight`
/// `layers.{i}.post_attention_layernorm.weight`
pub fn normalize_vendor_key(k: &str) -> Option<String> {
    bitnet_weight_name_core::normalize_vendor_key(k)
}

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

/// Find a tensor by trying multiple aliases. Returns the first matching
/// key and tensor pair.
fn pick<'a, 'b>(
    tensors: &'a HashMap<String, Tensor>,
    candidates: &[&'b str],
) -> Option<(&'b str, &'a Tensor)> {
    for k in candidates {
        if let Some(t) = tensors.get(*k) {
            return Some((*k, t));
        }
    }
    None
}

/// Map GGUF tensor names to transformer module names with strict option
pub fn remap_gguf_weights_with_options(
    tensors: &HashMap<String, Tensor>,
    strict: bool,
) -> Result<HashMap<String, Tensor>> {
    let mut mapped = HashMap::new();
    let mut unmapped = Vec::new();

    // First pass: map all tensors
    for (name, tensor) in tensors {
        // Handle raw QK256 tensor keys with side-map suffixes.
        // Strip suffix -> remap base -> re-append suffix
        let (base_name, suffix) = if let Some(base) = name.strip_suffix(".qk256_qs") {
            (base, Some(".qk256_qs"))
        } else if let Some(base) = name.strip_suffix(".qk256_scale") {
            (base, Some(".qk256_scale"))
        } else {
            (name.as_str(), None)
        };

        // First normalize any known name variations
        let normalized = normalize_name(base_name);
        let new_base_name = if let Some(canonical) = normalize_vendor_key(&normalized) {
            canonical
        } else if let Some(mapped_name) = map_tensor_name(&normalized) {
            mapped_name
        } else {
            unmapped.push(name.clone());
            base_name.to_string()
        };

        // Re-append suffix if it was present
        let new_name = if let Some(sfx) = suffix {
            format!("{}{}", new_base_name, sfx)
        } else {
            new_base_name
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
            tracing::warn!(
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
    // Token embeddings variations - comprehensive list
    if name == "token_embd.weight"
        || name == "tok_embeddings.weight"
        || name == "model.embed_tokens.weight"
        || name == "transformer.wte.weight"
        || name == "transformer.word_embeddings.weight"
        || name == "embeddings.word_embeddings.weight"
        || name == "embed.weight"
        || name == "embedding.weight"
        || name == "word_embeddings.weight"
    {
        return Some("embed_tokens.weight".to_string());
    }

    // Output layer variations - comprehensive list
    if name == "output.weight"
        || name == "lm_head.weight"
        || name == "model.lm_head.weight"
        || name == "generator.weight"
        || name == "transformer.lm_head.weight"
        || name == "language_model_head.weight"
        || name == "head.weight"
        || name == "cls.weight"
    {
        return Some("lm_head.weight".to_string());
    }

    // Final normalization - comprehensive list
    if name == "output_norm.weight"
        || name == "norm.weight"
        || name == "model.norm.weight"
        || name == "transformer.ln_f.weight"
        || name == "ln_f.weight"
        || name == "final_norm.weight"
        || name == "final_layernorm.weight"
        || name == "final_rmsnorm.weight"
    {
        return Some("final_norm.weight".to_string());
    }

    // Handle "blk.N." prefix (common in GGUF)
    if name.starts_with("blk.") {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() >= 3 {
            let layer_num = parts[1];
            let component = parts[2..].join(".");

            let mapped_component = match component.as_str() {
                // Attention weights - map to attention.* (not self_attn.*)
                "attn_q.weight" => "attention.q_proj.weight",
                "attn_k.weight" => "attention.k_proj.weight",
                "attn_v.weight" => "attention.v_proj.weight",
                "attn_output.weight" | "attn_o.weight" => "attention.o_proj.weight",
                "attn_q.bias" => "attention.q_proj.bias",
                "attn_k.bias" => "attention.k_proj.bias",
                "attn_v.bias" => "attention.v_proj.bias",
                "attn_output.bias" | "attn_o.bias" => "attention.o_proj.bias",

                // Attention normalization - use attention_norm prefix
                "attn_norm.weight" => "attention_norm.weight",
                "attn_q_norm.weight" => "attention.q_norm.weight",
                "attn_k_norm.weight" => "attention.k_norm.weight",
                "attn_sub_norm.weight" => "attention.sub_layernorm.weight", // BitNet specific

                // Feed-forward weights - map to feed_forward.* (not mlp.*)
                "ffn_gate.weight" | "ffn_gate_inp.weight" => "feed_forward.gate_proj.weight",
                "ffn_up.weight" | "ffn_up_proj.weight" => "feed_forward.up_proj.weight",
                "ffn_down.weight" | "ffn_down_proj.weight" => "feed_forward.down_proj.weight",

                // FFN normalization
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_sub_norm.weight" => "feed_forward.sub_layernorm.weight", // BitNet specific

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
                // LLaMA-style attention - map to attention.* for consistency
                "attention.wq.weight" | "self_attn.q_proj.weight" => "attention.q_proj.weight",
                "attention.wk.weight" | "self_attn.k_proj.weight" => "attention.k_proj.weight",
                "attention.wv.weight" | "self_attn.v_proj.weight" => "attention.v_proj.weight",
                "attention.wo.weight" | "self_attn.o_proj.weight" => "attention.o_proj.weight",
                "attention.wq.bias" | "self_attn.q_proj.bias" => "attention.q_proj.bias",
                "attention.wk.bias" | "self_attn.k_proj.bias" => "attention.k_proj.bias",
                "attention.wv.bias" | "self_attn.v_proj.bias" => "attention.v_proj.bias",
                "attention.wo.bias" | "self_attn.o_proj.bias" => "attention.o_proj.bias",

                // Normalization - map to expected names
                "attention_norm.weight" | "input_layernorm.weight" => "attention_norm.weight",
                "self_attn.q_norm.weight" | "attention.q_norm.weight" => "attention.q_norm.weight",
                "self_attn.k_norm.weight" | "attention.k_norm.weight" => "attention.k_norm.weight",
                "ffn_norm.weight" | "post_attention_layernorm.weight" => {
                    "post_attention_layernorm.weight"
                }

                // LLaMA-style FFN - map to feed_forward.* for consistency
                "feed_forward.w1.weight" | "mlp.gate_proj.weight" => {
                    "feed_forward.gate_proj.weight"
                }
                "feed_forward.w3.weight" | "mlp.up_proj.weight" => "feed_forward.up_proj.weight",
                "feed_forward.w2.weight" | "mlp.down_proj.weight" => {
                    "feed_forward.down_proj.weight"
                }

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

/// Detect hidden size from model weights that are guaranteed to be square or have known dimensions
fn detect_hidden_size_from_weights(
    tensors: &HashMap<String, Tensor>,
    fallback: usize,
) -> Result<usize> {
    // Check q_proj/k_proj/v_proj weights - these are typically [hidden, hidden]
    let projection_candidates = [
        "layers.0.attn_q.weight",
        "layers.0.attn_k.weight",
        "layers.0.attn_v.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
    ];

    for key in &projection_candidates {
        if let Some(tensor) = tensors.get(*key) {
            let shape = tensor.shape();
            if shape.dims().len() == 2 {
                let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
                // Q/K/V projections are typically square [hidden, hidden]
                if rows == cols {
                    tracing::info!("Detected hidden_size={} from {}", rows, key);
                    return Ok(rows);
                }
            }
        }
    }

    // Check layer norm weights - these are 1D with size=hidden
    let norm_candidates = [
        "layers.0.input_norm.weight",
        "layers.0.post_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "final_norm.weight",
        "model.norm.weight",
    ];

    for key in &norm_candidates {
        if let Some(tensor) = tensors.get(*key) {
            let shape = tensor.shape();
            if shape.dims().len() == 1 {
                let size = shape.dims()[0];
                tracing::info!("Detected hidden_size={} from {}", size, key);
                return Ok(size);
            }
        }
    }

    // Check MLP weights - gate/up are [hidden, intermediate], down is [intermediate, hidden]
    let mlp_down_candidates = [
        "layers.0.mlp_down.weight",
        "model.layers.0.mlp.down_proj.weight",
        "blk.0.ffn_down.weight",
    ];

    for key in &mlp_down_candidates {
        if let Some(tensor) = tensors.get(*key) {
            let shape = tensor.shape();
            if shape.dims().len() == 2 {
                // down_proj is [intermediate, hidden], so second dim is hidden
                let hidden = shape.dims()[1];
                tracing::info!("Detected hidden_size={} from {}", hidden, key);
                return Ok(hidden);
            }
        }
    }

    // If we couldn't detect, use the fallback
    if fallback > 0 {
        tracing::warn!("Could not detect hidden_size from weights, using fallback={}", fallback);
        Ok(fallback)
    } else {
        Err(bitnet_common::BitNetError::Validation(
            "Could not detect hidden_size from model weights".to_string(),
        ))
    }
}

/// Helper to find tensor by trying multiple key aliases
/// Returns (matched_key, tensor) tuple for better diagnostics
fn find_and_remove(
    tensors: &mut HashMap<String, Tensor>,
    keys: &[&str],
) -> Option<(String, Tensor)> {
    for k in keys {
        if let Some(t) = tensors.remove(*k) {
            return Some((k.to_string(), t));
        }
    }
    None
}

fn infer_projection_out_dim(
    tensors: &HashMap<String, Tensor>,
    keys: &[String],
    hidden: usize,
) -> Option<usize> {
    for key in keys {
        if let Some(tensor) = tensors.get(key.as_str()) {
            let dims = tensor.shape().dims();
            if let [a, b] = dims {
                if *b == hidden {
                    return Some(*a);
                }
                if *a == hidden {
                    return Some(*b);
                }
            }
        }
    }
    None
}

/// Normalize attention and FFN weights for a single layer to [out, in] layout
fn normalize_layer_weights(
    tensors: &mut HashMap<String, Tensor>,
    layer_idx: usize,
    dims: &ModelDims,
) -> Result<()> {
    let hidden = dims.hidden;

    // Build key prefixes for this layer (try both blk.N and layers.N)
    let blk_prefix = format!("blk.{}", layer_idx);
    let layers_prefix = format!("layers.{}", layer_idx);

    let q_keys = [
        format!("{}.attn_q.weight", blk_prefix),
        format!("{}.attention.q_proj.weight", layers_prefix),
        format!("{}.attention.wq.weight", layers_prefix),
        format!("{}.self_attn.q_proj.weight", layers_prefix),
    ];
    let k_keys = [
        format!("{}.attn_k.weight", blk_prefix),
        format!("{}.attention.k_proj.weight", layers_prefix),
        format!("{}.attention.wk.weight", layers_prefix),
        format!("{}.self_attn.k_proj.weight", layers_prefix),
    ];
    let v_keys = [
        format!("{}.attn_v.weight", blk_prefix),
        format!("{}.attention.v_proj.weight", layers_prefix),
        format!("{}.attention.wv.weight", layers_prefix),
        format!("{}.self_attn.v_proj.weight", layers_prefix),
    ];
    let o_keys = [
        format!("{}.attn_output.weight", blk_prefix),
        format!("{}.attn_o.weight", blk_prefix),
        format!("{}.attention.o_proj.weight", layers_prefix),
        format!("{}.attention.wo.weight", layers_prefix),
        format!("{}.self_attn.o_proj.weight", layers_prefix),
    ];

    let q_dim = infer_projection_out_dim(tensors, &q_keys, hidden).unwrap_or(dims.q_dim()?);
    let configured_kv_dim = dims.kv_dim()?;
    let inferred_kv_dim = infer_projection_out_dim(tensors, &k_keys, hidden)
        .or_else(|| infer_projection_out_dim(tensors, &v_keys, hidden))
        .unwrap_or(configured_kv_dim);
    let kv_dim = if inferred_kv_dim == hidden && configured_kv_dim != hidden {
        configured_kv_dim
    } else {
        inferred_kv_dim
    };

    // Attention Q/K/V/O projections
    let attn_keys = [
        // Q projection: [q_dim, hidden] where q_dim = head_dim * n_head
        ("q_proj", &q_keys as &[String], q_dim, hidden),
        // K projection: [kv_dim, hidden] where kv_dim = head_dim * n_kv_head
        ("k_proj", &k_keys, kv_dim, hidden),
        // V projection: [kv_dim, hidden]
        ("v_proj", &v_keys, kv_dim, hidden),
        // O projection: [hidden, q_dim]
        ("o_proj", &o_keys, hidden, q_dim),
    ];

    for (name, keys, out_dim, in_dim) in attn_keys {
        let key_strs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        if let Some((matched_key, tensor)) = find_and_remove(tensors, &key_strs) {
            tracing::debug!(
                "layer{}.attention.{}: resolved from '{}' with shape {:?}",
                layer_idx,
                name,
                matched_key,
                tensor.shape().dims()
            );

            // Validate shape before transposition to catch wrong tensor assignments
            let shape = tensor.shape().dims();

            // Special case: allow hidden×hidden only for K/V (exporter emitted full hidden)
            // We need to slice heads NOW in the mapper to produce correct [kv_dim, hidden] shape
            let is_kv_hidden_square = matches!(shape, [o, i] if (name == "k_proj" || name == "v_proj") && kv_dim != hidden && *o == hidden && *i == hidden);

            if is_kv_hidden_square {
                let n_heads = dims.n_head;
                let n_kv_heads = dims.n_kv_head;
                let head_dim = dims.head_dim()?;
                let group_size = n_heads / n_kv_heads;

                tracing::warn!(
                    "layer{}: Sliced K/V [hidden,hidden] -> [kv_dim,hidden] (GQA group_size={})",
                    layer_idx,
                    group_size
                );

                // Slice heads: [hidden, hidden] -> [kv_dim, hidden]
                // The weight is stored as [rows=hidden, cols=hidden]
                // We need to select rows corresponding to KV heads: [kv_dim, hidden]
                // where kv_dim = n_kv_heads * head_dim

                // Transpose to [hidden, hidden] if needed (tensor might be [hidden, hidden] already)
                // We want to slice ROWS, so we need [rows, cols] = [hidden, hidden]
                let weight = if tensor.shape().dims() == [hidden, hidden] {
                    tensor // Already correct orientation
                } else {
                    tensor.transpose(0, 1)? // Transpose to [hidden, hidden]
                };

                // Select the first row of each group: indices [0, group_size*head_dim, 2*group_size*head_dim, ...]
                // Actually, we want to select ALL rows for the first n_kv_heads worth of heads
                // Head 0: rows [0..head_dim]
                // Head 1: rows [head_dim..2*head_dim]
                // ...
                // Head (group_size-1): rows [(group_size-1)*head_dim..group_size*head_dim]  <- this is the last head of first group
                // Head group_size: rows [group_size*head_dim..(group_size+1)*head_dim]  <- first head of second group
                // ...
                // So for GQA, we want heads [0, group_size, 2*group_size, ...] each with head_dim rows
                // Which means rows [0..head_dim, group_size*head_dim..(group_size+1)*head_dim, ...]

                let mut row_indices = Vec::with_capacity(n_kv_heads * head_dim);
                for kv_idx in 0..n_kv_heads {
                    let head_idx = kv_idx * group_size; // First head of this group
                    let row_start = head_idx * head_dim;
                    let row_end = row_start + head_dim;
                    for row in row_start..row_end {
                        row_indices.push(row as i64);
                    }
                }

                // Create index tensor and slice
                let idx_tensor = candle_core::Tensor::new(row_indices.as_slice(), weight.device())?;
                let sliced = weight.index_select(&idx_tensor, 0)?; // Select rows

                tracing::debug!(
                    "layer{}.attention.{}: sliced shape {:?} -> {:?}",
                    layer_idx,
                    name,
                    weight.shape().dims(),
                    sliced.shape().dims()
                );

                // Store sliced weight [kv_dim, hidden]
                tensors.insert(format!("layers.{}.attention.{}.weight", layer_idx, name), sliced);
            } else {
                // Normal validation path
                match shape {
                    [o, i] if *o == out_dim && *i == in_dim => { /* already correct */ }
                    [i, o] if *i == in_dim && *o == out_dim => { /* transposed, will fix */ }
                    // Only allow square fused shapes when BOTH dims equal hidden (not just one)
                    // This catches GQA cases where K/V have kv_dim != hidden
                    [o, i]
                        if *o == hidden
                            && *i == hidden
                            && out_dim == hidden
                            && in_dim == hidden =>
                    { /* fused square shape for MHA (not GQA) */ }
                    _ => {
                        // Dump available keys for this layer to help diagnosis
                        tracing::error!(
                            "layer{}: available attention keys in tensor map:",
                            layer_idx
                        );
                        for k in tensors.keys().filter(|k| {
                            k.contains(&format!("layers.{}", layer_idx))
                                || k.contains(&format!("blk.{}", layer_idx))
                        }) {
                            tracing::error!(
                                "  - {} (shape: {:?})",
                                k,
                                tensors.get(k).map(|t| t.shape().dims())
                            );
                        }

                        return Err(bitnet_common::BitNetError::Validation(format!(
                            "layer{}.attention.{}: unexpected matrix shape {:?}, expected [{}, {}] or transpose. \
                             Resolved from key '{}'. This may indicate wrong tensor assignment. \
                             Expected dims: out={}, in={}; q_dim={}, kv_dim={}, hidden={}",
                            layer_idx,
                            name,
                            shape,
                            out_dim,
                            in_dim,
                            matched_key,
                            out_dim,
                            in_dim,
                            q_dim,
                            kv_dim,
                            hidden
                        )));
                    }
                }

                let normalized = ensure_matrix_or_transpose(
                    tensor,
                    out_dim,
                    in_dim,
                    &format!("layer{}.attention.{}", layer_idx, name),
                )?;
                tensors
                    .insert(format!("layers.{}.attention.{}.weight", layer_idx, name), normalized);
            }
        }
    }

    // FFN gate/up/down projections
    let ffn_keys = [
        // gate_proj (w1): [inter, hidden]
        (
            "gate_proj",
            &[
                format!("{}.ffn_gate.weight", blk_prefix),
                format!("{}.ffn_gate_inp.weight", blk_prefix),
                format!("{}.feed_forward.gate_proj.weight", layers_prefix),
                format!("{}.feed_forward.w1.weight", layers_prefix),
                format!("{}.mlp.gate_proj.weight", layers_prefix),
            ] as &[String],
            dims.inter,
            hidden,
        ),
        // up_proj (w3): [inter, hidden]
        (
            "up_proj",
            &[
                format!("{}.ffn_up.weight", blk_prefix),
                format!("{}.ffn_up_proj.weight", blk_prefix),
                format!("{}.feed_forward.up_proj.weight", layers_prefix),
                format!("{}.feed_forward.w3.weight", layers_prefix),
                format!("{}.mlp.up_proj.weight", layers_prefix),
            ],
            dims.inter,
            hidden,
        ),
        // down_proj (w2): [hidden, inter]
        (
            "down_proj",
            &[
                format!("{}.ffn_down.weight", blk_prefix),
                format!("{}.ffn_down_proj.weight", blk_prefix),
                format!("{}.feed_forward.down_proj.weight", layers_prefix),
                format!("{}.feed_forward.w2.weight", layers_prefix),
                format!("{}.mlp.down_proj.weight", layers_prefix),
            ],
            hidden,
            dims.inter,
        ),
    ];

    for (name, keys, out_dim, in_dim) in ffn_keys {
        let key_strs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        if let Some((matched_key, tensor)) = find_and_remove(tensors, &key_strs) {
            tracing::debug!(
                "layer{}.ffn.{}: resolved from '{}' with shape {:?}",
                layer_idx,
                name,
                matched_key,
                tensor.shape().dims()
            );

            let normalized = ensure_matrix_or_transpose(
                tensor,
                out_dim,
                in_dim,
                &format!("layer{}.ffn.{}", layer_idx, name),
            )?;
            tensors
                .insert(format!("layers.{}.feed_forward.{}.weight", layer_idx, name), normalized);
        }
    }

    Ok(())
}

/// Detect vocab size and normalize embedding/lm_head tensors, and transpose all layer weights
/// Returns (vocab_size, actual_hidden_size)
pub fn normalize_model_tensors(
    tensors: &mut HashMap<String, Tensor>,
    config: &bitnet_common::BitNetConfig,
) -> Result<(usize, usize)> {
    let expected_hidden_size = config.model.hidden_size;

    // Build dimensions from config
    tracing::info!(
        "Building ModelDims from config: hidden={}, n_head={}, num_key_value_heads={} (raw)",
        config.model.hidden_size,
        config.model.num_heads,
        config.model.num_key_value_heads
    );

    let n_kv_head_resolved = if config.model.num_key_value_heads > 0 {
        config.model.num_key_value_heads
    } else {
        tracing::warn!("num_key_value_heads is 0, falling back to num_heads (MHA)");
        config.model.num_heads // MHA fallback
    };

    let dims = ModelDims {
        hidden: config.model.hidden_size,
        n_head: config.model.num_heads,
        n_kv_head: n_kv_head_resolved,
        head_dim: config
            .model
            .attention_head_dim
            .unwrap_or_else(|| config.model.hidden_size / config.model.num_heads),
        inter: config.model.intermediate_size,
        vocab: config.model.vocab_size,
    };

    // Calculate derived dimensions for validation
    let head_dim = dims.head_dim()?;
    let q_dim = dims.q_dim()?;
    let kv_dim = dims.kv_dim()?;

    tracing::info!(
        "Model dimensions from config: hidden={}, n_head={}, n_kv_head={}, inter={}, vocab={}, num_layers={}",
        dims.hidden,
        dims.n_head,
        dims.n_kv_head,
        dims.inter,
        dims.vocab,
        config.model.num_layers
    );
    tracing::info!("Derived dimensions: head_dim={}, q_dim={}, kv_dim={}", head_dim, q_dim, kv_dim);

    // 1) Normalize all layer weights to [out, in] layout
    for layer_idx in 0..config.model.num_layers {
        normalize_layer_weights(tensors, layer_idx, &dims)?;
    }

    // 2) Locate embedding with robust aliases
    let emb_candidates = [
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "tok_embeddings.weight",
        "token_embd.weight",
        "transformer.wte.weight",
    ];

    let (emb_key, er, ec, _emb_device) = {
        let (key, emb) = pick(tensors, &emb_candidates).ok_or_else(|| {
            bitnet_common::BitNetError::Validation(
                "embed tokens not found (tried embed_tokens/tok_embeddings/token_embd/transformer.wte)"
                    .to_string(),
            )
        })?;

        // Get embedding info first (before mutating tensors)
        let (er, ec) = dims2(emb, "embed_tokens.weight")?;
        let device = emb.device().clone();
        (key, er, ec, device)
    };

    // 3) Try to detect hidden size from other model weights first
    let detected_hidden = detect_hidden_size_from_weights(tensors, expected_hidden_size)?;

    // 3) Infer vocab + orientation from the embedding shape
    tracing::info!(
        "Embedding tensor shape: [{}, {}], detected_hidden_size: {}",
        er,
        ec,
        detected_hidden
    );

    // Detect actual hidden size and vocab from tensor using our detection
    let (vocab_size, hidden_size, emb_needs_t) = if er == detected_hidden {
        // Shape is [hidden, vocab]
        (ec, er, true)
    } else if ec == detected_hidden {
        // Shape is [vocab, hidden]
        (er, ec, false)
    } else {
        // Fallback to size heuristic if detection failed
        tracing::warn!(
            "Could not match hidden size {} to embedding dims [{}, {}], using size heuristic",
            detected_hidden,
            er,
            ec
        );
        if er > ec {
            (er, ec, false) // Assume [vocab, hidden]
        } else {
            (ec, er, true) // Assume [hidden, vocab]
        }
    };

    // Log final detection results
    tracing::info!(
        "Model dimensions detected: vocab_size={}, hidden_size={}, embedding_transposed={}",
        vocab_size,
        hidden_size,
        emb_needs_t
    );

    // 3) Transpose embeddings if needed to ensure [vocab, hidden] layout
    if emb_needs_t {
        tracing::info!(
            "Transposing embed_tokens from [hidden={}, vocab={}] to [vocab={}, hidden={}]",
            hidden_size,
            vocab_size,
            vocab_size,
            hidden_size
        );
        let emb = tensors.remove(emb_key).unwrap();
        let emb_t = emb.t()?.contiguous()?; // Transpose to [vocab, hidden] and make contiguous
        tensors.insert("embed_tokens.weight".to_string(), emb_t);
    } else if emb_key != "embed_tokens.weight" {
        // Just rename the key if needed
        let emb = tensors.remove(emb_key).unwrap();
        tensors.insert("embed_tokens.weight".to_string(), emb);
    }

    // 4) Locate lm_head with robust aliases, normalize to [n_vocab, n_embd]
    let lm_candidates =
        ["lm_head.weight", "output.weight", "model.lm_head.weight", "generator.weight"];

    if let Some((lm_key, lm_needs_t, lm_device)) = {
        if let Some((key, lm)) = pick(tensors, &lm_candidates) {
            let (lr, lc) = dims2(lm, "lm_head.weight")?;
            let device = lm.device().clone();

            let needs_t = match (lr, lc) {
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
            Some((key, needs_t, device))
        } else {
            None
        }
    } {
        if lm_needs_t {
            tracing::warn!(
                "lm_head is transposed - avoiding {} MB transpose",
                (vocab_size * hidden_size * 4) / (1024 * 1024)
            );
            // Store metadata instead of transposing
            if lm_key != "lm_head.weight" {
                let lm = tensors.remove(lm_key).unwrap();
                tensors.insert("lm_head.weight".to_string(), lm);
            }
            let transpose_flag = Tensor::from_slice(&[1.0f32], 1, &lm_device)?;
            tensors.insert("lm_head.transposed".to_string(), transpose_flag);
        } else if lm_key != "lm_head.weight" {
            let lm = tensors.remove(lm_key).unwrap();
            tensors.insert("lm_head.weight".to_string(), lm);
            let transpose_flag = Tensor::from_slice(&[0.0f32], 1, &lm_device)?;
            tensors.insert("lm_head.transposed".to_string(), transpose_flag);
        }
    } else {
        tracing::info!("No lm_head.weight found; using tied weights with embed_tokens.");
    }

    Ok((vocab_size, hidden_size))
}

// ---------------------------------------------------------------------------
// WeightMapper: HuggingFace ↔ Internal (GGUF-style) name translation
// ---------------------------------------------------------------------------

use crate::architecture::ModelArchitecture;

/// A single mapping rule using `{n}` as a layer-number placeholder.
#[derive(Debug, Clone)]
struct MappingRule {
    hf: String,
    internal: String,
}

fn rule(hf: &str, internal: &str) -> MappingRule {
    MappingRule { hf: hf.to_string(), internal: internal.to_string() }
}

/// Extract the layer number from a concrete name given a pattern containing `{n}`.
fn extract_layer_number(name: &str, pattern: &str) -> Option<usize> {
    let (prefix, suffix) = pattern.split_once("{n}")?;
    let rest = name.strip_prefix(prefix)?;
    let num_str = rest.strip_suffix(suffix)?;
    num_str.parse().ok()
}

/// Common LLaMA-family rules shared by Phi-4, LLaMA-3, Qwen2.5, Mistral, etc.
fn llama_family_rules() -> Vec<MappingRule> {
    vec![
        rule("model.embed_tokens.weight", "token_embd.weight"),
        rule("model.layers.{n}.self_attn.q_proj.weight", "blk.{n}.attn_q.weight"),
        rule("model.layers.{n}.self_attn.k_proj.weight", "blk.{n}.attn_k.weight"),
        rule("model.layers.{n}.self_attn.v_proj.weight", "blk.{n}.attn_v.weight"),
        rule("model.layers.{n}.self_attn.o_proj.weight", "blk.{n}.attn_output.weight"),
        rule("model.layers.{n}.self_attn.q_proj.bias", "blk.{n}.attn_q.bias"),
        rule("model.layers.{n}.self_attn.k_proj.bias", "blk.{n}.attn_k.bias"),
        rule("model.layers.{n}.self_attn.v_proj.bias", "blk.{n}.attn_v.bias"),
        rule("model.layers.{n}.self_attn.o_proj.bias", "blk.{n}.attn_output.bias"),
        rule("model.layers.{n}.mlp.gate_proj.weight", "blk.{n}.ffn_gate.weight"),
        rule("model.layers.{n}.mlp.up_proj.weight", "blk.{n}.ffn_up.weight"),
        rule("model.layers.{n}.mlp.down_proj.weight", "blk.{n}.ffn_down.weight"),
        rule("model.layers.{n}.input_layernorm.weight", "blk.{n}.attn_norm.weight"),
        rule("model.layers.{n}.post_attention_layernorm.weight", "blk.{n}.ffn_norm.weight"),
        rule("model.norm.weight", "output_norm.weight"),
        rule("lm_head.weight", "output.weight"),
    ]
}

/// Gemma-2 extends the LLaMA pattern with extra norm layers.
fn gemma_rules() -> Vec<MappingRule> {
    let mut rules = llama_family_rules();
    rules.extend([
        rule("model.layers.{n}.pre_feedforward_layernorm.weight", "blk.{n}.ffn_pre_norm.weight"),
        rule("model.layers.{n}.post_feedforward_layernorm.weight", "blk.{n}.ffn_post_norm.weight"),
    ]);
    rules
}

/// Select the mapping rules for a given architecture.
fn build_rules(arch: &ModelArchitecture) -> Vec<MappingRule> {
    match arch {
        ModelArchitecture::Gemma => gemma_rules(),
        // All other known architectures follow the LLaMA-family pattern.
        _ => llama_family_rules(),
    }
}

/// Maps HuggingFace model weight names to internal GGUF-style tensor names.
///
/// Supports pattern-based translation with `{n}` layer-number placeholders,
/// forward mapping (HF → internal), and reverse mapping (internal → HF).
///
/// # Example
///
/// ```
/// use bitnet_models::weight_mapper::WeightMapper;
/// use bitnet_models::architecture::ModelArchitecture;
///
/// let mapper = WeightMapper::for_architecture(ModelArchitecture::Phi);
/// assert_eq!(
///     mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
///     Some("blk.0.attn_q.weight".to_string()),
/// );
/// assert_eq!(
///     mapper.reverse_map("blk.0.attn_q.weight"),
///     Some("model.layers.0.self_attn.q_proj.weight".to_string()),
/// );
/// ```
pub struct WeightMapper {
    architecture: ModelArchitecture,
    rules: Vec<MappingRule>,
}

impl WeightMapper {
    /// Build a mapper for the given model architecture.
    pub fn for_architecture(arch: ModelArchitecture) -> Self {
        let rules = build_rules(&arch);
        Self { architecture: arch, rules }
    }

    /// Translate a HuggingFace weight name to the internal GGUF-style name.
    ///
    /// Returns `None` if the name does not match any known mapping rule.
    pub fn map_name(&self, hf_name: &str) -> Option<String> {
        for r in &self.rules {
            if r.hf.contains("{n}") {
                if let Some(n) = extract_layer_number(hf_name, &r.hf) {
                    return Some(r.internal.replace("{n}", &n.to_string()));
                }
            } else if hf_name == r.hf {
                return Some(r.internal.clone());
            }
        }
        None
    }

    /// Translate an internal GGUF-style name back to the HuggingFace name.
    ///
    /// Returns `None` if the name does not match any known mapping rule.
    pub fn reverse_map(&self, internal_name: &str) -> Option<String> {
        for r in &self.rules {
            if r.internal.contains("{n}") {
                if let Some(n) = extract_layer_number(internal_name, &r.internal) {
                    return Some(r.hf.replace("{n}", &n.to_string()));
                }
            } else if internal_name == r.internal {
                return Some(r.hf.clone());
            }
        }
        None
    }

    /// The architecture this mapper was built for.
    pub fn architecture(&self) -> &ModelArchitecture {
        &self.architecture
    }

    /// The number of mapping rules (useful for weight-count validation).
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Return an iterator over (hf_pattern, internal_pattern) pairs.
    pub fn rules(&self) -> impl Iterator<Item = (&str, &str)> {
        self.rules.iter().map(|r| (r.hf.as_str(), r.internal.as_str()))
    }

    /// Generate all concrete weight names for a model with `num_layers` layers.
    ///
    /// Returns `(hf_names, internal_names)` — both sorted for stable comparison.
    pub fn all_names(&self, num_layers: usize) -> (Vec<String>, Vec<String>) {
        let mut hf = Vec::new();
        let mut internal = Vec::new();
        for r in &self.rules {
            if r.hf.contains("{n}") {
                for n in 0..num_layers {
                    hf.push(r.hf.replace("{n}", &n.to_string()));
                    internal.push(r.internal.replace("{n}", &n.to_string()));
                }
            } else {
                hf.push(r.hf.clone());
                internal.push(r.internal.clone());
            }
        }
        hf.sort();
        internal.sort();
        (hf, internal)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device as CDevice, Tensor as CandleTensor};

    #[test]
    fn maps_blk_and_llama_variants() {
        assert_eq!(
            normalize_vendor_key("blk.7.attn_q.weight").as_deref(),
            Some("layers.7.attention.q_proj.weight")
        );
        assert_eq!(
            normalize_vendor_key("model.layers.0.self_attn.wo.weight").as_deref(),
            Some("layers.0.attention.o_proj.weight")
        );
        assert_eq!(
            normalize_vendor_key("blk.2.ffn_gate.weight").as_deref(),
            Some("layers.2.feed_forward.gate_proj.weight")
        );
        assert_eq!(
            normalize_vendor_key("layers.3.post_attention_layernorm.weight").as_deref(),
            Some("layers.3.post_attention_layernorm.weight")
        );
    }

    #[test]
    fn maps_qwen_attention_bias_variants() {
        assert_eq!(
            map_tensor_name("blk.0.attn_q.bias").as_deref(),
            Some("layers.0.attention.q_proj.bias")
        );
        assert_eq!(
            map_tensor_name("blk.0.attn_k.bias").as_deref(),
            Some("layers.0.attention.k_proj.bias")
        );
        assert_eq!(
            map_tensor_name("blk.0.attn_v.bias").as_deref(),
            Some("layers.0.attention.v_proj.bias")
        );
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.q_proj.bias").as_deref(),
            Some("layers.0.attention.q_proj.bias")
        );
    }

    /// Regression test for GQA K/V head slicing when exporter emits hidden×hidden weights
    ///
    /// This test verifies the fix for Issue #XXX where some exporters emit K/V projections
    /// as [hidden, hidden] square matrices instead of [kv_dim, hidden]. The mapper should
    /// slice these to the correct [kv_dim, hidden] shape by selecting rows corresponding
    /// to the KV heads in a GQA configuration.
    ///
    /// Test case:
    /// - hidden_size = 2560
    /// - n_heads = 20 (head_dim = 128)
    /// - n_kv_heads = 5 (GQA with group_size = 4)
    /// - kv_dim = 5 * 128 = 640
    ///
    /// Expected behavior:
    /// - Input K weight: [2560, 2560] (square matrix from exporter)
    /// - Output K weight: [640, 2560] (sliced to kv_dim rows)
    /// - Selected row ranges: [0..128, 512..640, 1024..1152, 1536..1664, 2048..2176]
    ///   (first head of each group in GQA)
    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)] // Intentional for readability: [row * hidden + col]
    fn test_kv_slicing_for_gqa() {
        let device = CDevice::Cpu;
        let hidden_size = 2560;
        let n_heads = 20;
        let n_kv_heads = 5;
        let head_dim = 128;
        let kv_dim = n_kv_heads * head_dim; // 640

        // Create ModelDims for GQA configuration
        let dims = ModelDims {
            hidden: hidden_size,
            n_head: n_heads,
            n_kv_head: n_kv_heads,
            head_dim,
            inter: 10240,  // Not used in this test
            vocab: 128000, // Not used in this test
        };

        // Create a fake [hidden, hidden] K weight tensor
        // Fill with unique values so we can verify correct row slicing
        let mut k_data = Vec::with_capacity(hidden_size * hidden_size);
        for row in 0..hidden_size {
            for col in 0..hidden_size {
                k_data.push((row * 10000 + col) as f32);
            }
        }

        let k_weight = CandleTensor::from_vec(k_data, &[hidden_size, hidden_size], &device)
            .expect("Failed to create K weight tensor");

        // Simulate the mapper's tensor map with the square K weight
        let mut tensors = HashMap::new();
        tensors.insert("blk.0.attn_k.weight".to_string(), k_weight);

        // Run normalize_layer_weights which should slice the K weight
        normalize_layer_weights(&mut tensors, 0, &dims)
            .expect("normalize_layer_weights should succeed");

        // Verify the K weight was sliced to [kv_dim, hidden]
        let k_sliced = tensors
            .get("layers.0.attention.k_proj.weight")
            .expect("K weight should be present in canonical name");

        let shape = k_sliced.shape().dims();
        assert_eq!(
            shape,
            &[kv_dim, hidden_size],
            "K weight should be sliced to [kv_dim={}, hidden={}], got {:?}",
            kv_dim,
            hidden_size,
            shape
        );

        // Verify the selected rows match the first head of each group
        // Expected row indices for GQA (group_size = 4):
        // Group 0: head 0 → rows [0..128]
        // Group 1: head 4 → rows [512..640]   (4 * 128 = 512)
        // Group 2: head 8 → rows [1024..1152] (8 * 128 = 1024)
        // Group 3: head 12 → rows [1536..1664] (12 * 128 = 1536)
        // Group 4: head 16 → rows [2048..2176] (16 * 128 = 2048)

        let k_vec = k_sliced
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("Failed to extract K weight data");

        // Verify first element of each group's head
        // Row 0, col 0: should be (0 * 10000 + 0) = 0
        assert_eq!(k_vec[0 * hidden_size + 0], 0.0, "Group 0, head 0, row 0");

        // Row 128, col 0: should be from original row 512 → (512 * 10000 + 0) = 5,120,000
        assert_eq!(k_vec[128 * hidden_size + 0], 512.0 * 10000.0, "Group 1, head 4, row 512");

        // Row 256, col 0: should be from original row 1024 → (1024 * 10000 + 0) = 10,240,000
        assert_eq!(k_vec[256 * hidden_size + 0], 1024.0 * 10000.0, "Group 2, head 8, row 1024");

        // Row 384, col 0: should be from original row 1536 → (1536 * 10000 + 0) = 15,360,000
        assert_eq!(k_vec[384 * hidden_size + 0], 1536.0 * 10000.0, "Group 3, head 12, row 1536");

        // Row 512, col 0: should be from original row 2048 → (2048 * 10000 + 0) = 20,480,000
        assert_eq!(k_vec[512 * hidden_size + 0], 2048.0 * 10000.0, "Group 4, head 16, row 2048");

        println!(
            "✅ KV slicing regression test passed: [{}×{}] → [{}×{}]",
            hidden_size, hidden_size, kv_dim, hidden_size
        );
    }

    #[test]
    fn test_remap_qk256_suffix() {
        // Test that .qk256_qs suffix is preserved during remapping
        let mut tensors = HashMap::new();

        // Create a dummy tensor
        let tensor =
            CandleTensor::zeros(&[32, 128], candle_core::DType::U8, &CDevice::Cpu).unwrap();

        // Test with a GGUF key that should be remapped
        tensors.insert("blk.0.attn_q.weight.qk256_qs".to_string(), tensor.clone());
        tensors.insert("blk.0.attn_q.weight.qk256_scale".to_string(), tensor.clone());

        let mapped = remap_gguf_weights(&tensors).unwrap();

        // Verify the key was remapped correctly with suffix preserved
        assert!(
            mapped.contains_key("layers.0.attention.q_proj.weight.qk256_qs"),
            "Expected remapped key with .qk256_qs suffix, got keys: {:?}",
            mapped.keys().collect::<Vec<_>>()
        );
        assert!(
            mapped.contains_key("layers.0.attention.q_proj.weight.qk256_scale"),
            "Expected remapped key with .qk256_scale suffix, got keys: {:?}",
            mapped.keys().collect::<Vec<_>>()
        );

        // Verify the base key (without suffix) is NOT present
        assert!(
            !mapped.contains_key("layers.0.attention.q_proj.weight"),
            "Base key without suffix should not be present"
        );
    }

    #[test]
    fn test_remap_regular_key_no_suffix() {
        // Test that regular keys without .qk256_qs work normally
        let mut tensors = HashMap::new();

        let tensor =
            CandleTensor::zeros(&[32, 128], candle_core::DType::F32, &CDevice::Cpu).unwrap();
        tensors.insert("blk.0.attn_q.weight".to_string(), tensor.clone());

        let mapped = remap_gguf_weights(&tensors).unwrap();

        // Verify the key was remapped correctly without any suffix
        assert!(
            mapped.contains_key("layers.0.attention.q_proj.weight"),
            "Expected remapped key without suffix, got keys: {:?}",
            mapped.keys().collect::<Vec<_>>()
        );
    }

    // -----------------------------------------------------------------------
    // WeightMapper tests
    // -----------------------------------------------------------------------

    use crate::architecture::ModelArchitecture;

    #[test]
    fn phi4_maps_all_weight_names() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Phi);
        assert_eq!(m.map_name("model.embed_tokens.weight"), Some("token_embd.weight".into()));
        assert_eq!(
            m.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.5.self_attn.k_proj.weight"),
            Some("blk.5.attn_k.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.31.self_attn.v_proj.weight"),
            Some("blk.31.attn_v.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.0.self_attn.o_proj.weight"),
            Some("blk.0.attn_output.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.0.mlp.gate_proj.weight"),
            Some("blk.0.ffn_gate.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.0.mlp.up_proj.weight"),
            Some("blk.0.ffn_up.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.0.mlp.down_proj.weight"),
            Some("blk.0.ffn_down.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.0.input_layernorm.weight"),
            Some("blk.0.attn_norm.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.0.post_attention_layernorm.weight"),
            Some("blk.0.ffn_norm.weight".into()),
        );
        assert_eq!(m.map_name("model.norm.weight"), Some("output_norm.weight".into()));
        assert_eq!(m.map_name("lm_head.weight"), Some("output.weight".into()));
    }

    #[test]
    fn llama3_maps_all_weight_names() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Llama);
        assert_eq!(m.map_name("model.embed_tokens.weight"), Some("token_embd.weight".into()));
        assert_eq!(
            m.map_name("model.layers.12.self_attn.q_proj.weight"),
            Some("blk.12.attn_q.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.12.self_attn.o_proj.weight"),
            Some("blk.12.attn_output.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.12.mlp.gate_proj.weight"),
            Some("blk.12.ffn_gate.weight".into()),
        );
        assert_eq!(m.map_name("model.norm.weight"), Some("output_norm.weight".into()));
        assert_eq!(m.map_name("lm_head.weight"), Some("output.weight".into()));
    }

    #[test]
    fn qwen25_uses_llama_pattern() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Qwen);
        assert_eq!(
            m.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".into()),
        );
        assert_eq!(m.map_name("lm_head.weight"), Some("output.weight".into()));
    }

    #[test]
    fn mistral_uses_llama_pattern() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Mistral);
        assert_eq!(
            m.map_name("model.layers.3.self_attn.v_proj.weight"),
            Some("blk.3.attn_v.weight".into()),
        );
    }

    #[test]
    fn gemma_has_extra_norm_rules() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Gemma);
        // Standard rules still work
        assert_eq!(m.map_name("model.embed_tokens.weight"), Some("token_embd.weight".into()));
        assert_eq!(
            m.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".into()),
        );
        // Gemma-specific norms
        assert_eq!(
            m.map_name("model.layers.1.pre_feedforward_layernorm.weight"),
            Some("blk.1.ffn_pre_norm.weight".into()),
        );
        assert_eq!(
            m.map_name("model.layers.1.post_feedforward_layernorm.weight"),
            Some("blk.1.ffn_post_norm.weight".into()),
        );
    }

    #[test]
    fn reverse_mapping_works() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Phi);
        assert_eq!(m.reverse_map("token_embd.weight"), Some("model.embed_tokens.weight".into()),);
        assert_eq!(
            m.reverse_map("blk.7.attn_q.weight"),
            Some("model.layers.7.self_attn.q_proj.weight".into()),
        );
        assert_eq!(
            m.reverse_map("blk.0.ffn_gate.weight"),
            Some("model.layers.0.mlp.gate_proj.weight".into()),
        );
        assert_eq!(m.reverse_map("output_norm.weight"), Some("model.norm.weight".into()),);
        assert_eq!(m.reverse_map("output.weight"), Some("lm_head.weight".into()),);
    }

    #[test]
    fn unknown_name_returns_none() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Phi);
        assert_eq!(m.map_name("totally.unknown.tensor"), None);
        assert_eq!(m.reverse_map("totally.unknown.tensor"), None);
    }

    #[test]
    fn layer_number_extraction() {
        assert_eq!(
            extract_layer_number(
                "model.layers.42.self_attn.q_proj.weight",
                "model.layers.{n}.self_attn.q_proj.weight"
            ),
            Some(42),
        );
        assert_eq!(extract_layer_number("blk.0.attn_q.weight", "blk.{n}.attn_q.weight"), Some(0),);
        assert_eq!(extract_layer_number("no_match", "blk.{n}.attn_q.weight"), None,);
    }

    #[test]
    fn all_names_count_matches_expected() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Phi);
        let num_layers = 32;
        let (hf, internal) = m.all_names(num_layers);
        // 13 per-layer rules × 32 layers + 3 global (embed + norm + lm_head) = 419.
        // Qwen-style attention biases are part of the shared LLaMA-family map.
        let expected = 13 * num_layers + 3;
        assert_eq!(hf.len(), expected, "HF names count");
        assert_eq!(internal.len(), expected, "internal names count");
    }

    #[test]
    fn all_names_roundtrip_consistency() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Llama);
        let (hf_names, internal_names) = m.all_names(4);
        // Every generated HF name should map to its corresponding internal name
        for hf in &hf_names {
            assert!(m.map_name(hf).is_some(), "HF name '{}' should map", hf);
        }
        for internal in &internal_names {
            assert!(
                m.reverse_map(internal).is_some(),
                "internal '{}' should reverse-map",
                internal
            );
        }
    }

    #[test]
    fn architecture_accessor() {
        let m = WeightMapper::for_architecture(ModelArchitecture::Phi);
        assert_eq!(m.architecture(), &ModelArchitecture::Phi);
    }

    #[test]
    fn gemma_has_more_rules_than_llama() {
        let llama = WeightMapper::for_architecture(ModelArchitecture::Llama);
        let gemma = WeightMapper::for_architecture(ModelArchitecture::Gemma);
        assert!(gemma.rule_count() > llama.rule_count(), "Gemma should have extra norm rules");
    }
}
