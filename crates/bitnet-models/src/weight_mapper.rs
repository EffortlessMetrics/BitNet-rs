use bitnet_common::Result;
use candle_core::{DType, Device, Tensor};
use once_cell::sync::Lazy;
use regex::Regex;
/// Weight mapping utilities for loading model weights from various formats
use std::borrow::Cow;
use std::collections::HashMap;

/// Model dimensions for tensor shape validation and transposition
#[derive(Clone, Copy, Debug)]
struct ModelDims {
    hidden: usize,
    n_head: usize,
    n_kv_head: usize,
    inter: usize,
    vocab: usize,
}

impl ModelDims {
    fn head_dim(&self) -> Result<usize> {
        let d = self.hidden / self.n_head;
        if d * self.n_head != self.hidden {
            return Err(bitnet_common::BitNetError::Validation(format!(
                "hidden_size {} not divisible by n_head {}",
                self.hidden, self.n_head
            )));
        }
        Ok(d)
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
            Ok(t.t()?.contiguous()?)
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
            Ok(t.t()?.contiguous()?)
        }
        _ => Err(bitnet_common::BitNetError::Validation(format!(
            "{}: unexpected matrix shape {:?}, expected [{}, {}] or its transpose",
            name, shp, expected_out, expected_in
        ))),
    }
}

/// Canonical target schema:
/// layers.{i}.attention.{q_proj|k_proj|v_proj|o_proj}.weight
/// layers.{i}.feed_forward.{gate_proj|up_proj|down_proj}.weight
/// layers.{i}.attention_norm.weight
/// layers.{i}.post_attention_layernorm.weight
static RE_BLK_ATTN_Q: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^blk\.(\d+)\.attn_q\.weight$").unwrap());
static RE_BLK_ATTN_K: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^blk\.(\d+)\.attn_k\.weight$").unwrap());
static RE_BLK_ATTN_V: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^blk\.(\d+)\.attn_v\.weight$").unwrap());
static RE_BLK_ATTN_O: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^blk\.(\d+)\.attn_o(?:utput)?\.weight$").unwrap());

static RE_LLAMA_WQ: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:self_)?attn\.wq\.weight$").unwrap());
static RE_LLAMA_WK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:self_)?attn\.wk\.weight$").unwrap());
static RE_LLAMA_WV: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:self_)?attn\.wv\.weight$").unwrap());
static RE_LLAMA_WO: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:self_)?attn\.wo\.weight$").unwrap());

// FFN / MLP variants
static RE_BLK_FFN_GATE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^blk\.(\d+)\.ffn_gate(?:_inp)?\.weight$").unwrap());
static RE_BLK_FFN_UP: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^blk\.(\d+)\.ffn_(?:up|up_proj)\.weight$").unwrap());
static RE_BLK_FFN_DOWN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^blk\.(\d+)\.ffn_(?:down|down_proj)\.weight$").unwrap());

static RE_FFN_W1: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:mlp|feed_forward)\.(?:w1|gate_proj)\.weight$")
        .unwrap()
});
static RE_FFN_W3: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:mlp|feed_forward)\.(?:w3|up_proj)\.weight$")
        .unwrap()
});
static RE_FFN_W2: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:mlp|feed_forward)\.(?:w2|down_proj)\.weight$")
        .unwrap()
});

// Norm aliases
static RE_ATTN_NORM: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:attention_norm|input_layernorm)\.weight$").unwrap()
});
static RE_FFN_NORM: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:post_attention_layernorm|ffn_norm)\.weight$")
        .unwrap()
});

/// Returns canonical key if `k` matches a known vendor pattern.
pub fn normalize_vendor_key(k: &str) -> Option<String> {
    macro_rules! cap {
        ($re:expr, $k:expr, $fmt:expr) => {{ if let Some(c) = $re.captures($k) { Some(format!($fmt, &c[1])) } else { None } }};
    }

    // Attention (blk.*)
    cap!(RE_BLK_ATTN_Q, k, "layers.{}.attention.q_proj.weight")
        .or_else(|| cap!(RE_BLK_ATTN_K, k, "layers.{}.attention.k_proj.weight"))
        .or_else(|| cap!(RE_BLK_ATTN_V, k, "layers.{}.attention.v_proj.weight"))
        .or_else(|| cap!(RE_BLK_ATTN_O, k, "layers.{}.attention.o_proj.weight"))
        // LLaMA-style attention
        .or_else(|| cap!(RE_LLAMA_WQ, k, "layers.{}.attention.q_proj.weight"))
        .or_else(|| cap!(RE_LLAMA_WK, k, "layers.{}.attention.k_proj.weight"))
        .or_else(|| cap!(RE_LLAMA_WV, k, "layers.{}.attention.v_proj.weight"))
        .or_else(|| cap!(RE_LLAMA_WO, k, "layers.{}.attention.o_proj.weight"))
        // FFN / MLP
        .or_else(|| cap!(RE_BLK_FFN_GATE, k, "layers.{}.feed_forward.gate_proj.weight"))
        .or_else(|| cap!(RE_BLK_FFN_UP,   k, "layers.{}.feed_forward.up_proj.weight"))
        .or_else(|| cap!(RE_BLK_FFN_DOWN, k, "layers.{}.feed_forward.down_proj.weight"))
        .or_else(|| cap!(RE_FFN_W1, k, "layers.{}.feed_forward.gate_proj.weight"))
        .or_else(|| cap!(RE_FFN_W3, k, "layers.{}.feed_forward.up_proj.weight"))
        .or_else(|| cap!(RE_FFN_W2, k, "layers.{}.feed_forward.down_proj.weight"))
        // Norms
        .or_else(|| cap!(RE_ATTN_NORM, k, "layers.{}.attention_norm.weight"))
        .or_else(|| cap!(RE_FFN_NORM,  k, "layers.{}.post_attention_layernorm.weight"))
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
        // First normalize any known name variations
        let normalized = normalize_name(name);
        let new_name = if let Some(canonical) = normalize_vendor_key(&normalized) {
            canonical
        } else if let Some(mapped_name) = map_tensor_name(&normalized) {
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

                // Attention normalization - use attention_norm prefix
                "attn_norm.weight" => "attention_norm.weight",
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

                // Normalization - map to expected names
                "attention_norm.weight" | "input_layernorm.weight" => "attention_norm.weight",
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
fn find_and_remove(tensors: &mut HashMap<String, Tensor>, keys: &[&str]) -> Option<Tensor> {
    for k in keys {
        if let Some(t) = tensors.remove(*k) {
            return Some(t);
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
    let q_dim = dims.q_dim()?;
    let kv_dim = dims.kv_dim()?;

    // Build key prefixes for this layer (try both blk.N and layers.N)
    let blk_prefix = format!("blk.{}", layer_idx);
    let layers_prefix = format!("layers.{}", layer_idx);

    // Attention Q/K/V/O projections
    let attn_keys = [
        // Q projection: [q_dim, hidden] where q_dim = head_dim * n_head
        (
            "q_proj",
            &[
                format!("{}.attn_q.weight", blk_prefix),
                format!("{}.attention.q_proj.weight", layers_prefix),
                format!("{}.attention.wq.weight", layers_prefix),
                format!("{}.self_attn.q_proj.weight", layers_prefix),
            ] as &[String],
            q_dim,
            hidden,
        ),
        // K projection: [kv_dim, hidden] where kv_dim = head_dim * n_kv_head
        (
            "k_proj",
            &[
                format!("{}.attn_k.weight", blk_prefix),
                format!("{}.attention.k_proj.weight", layers_prefix),
                format!("{}.attention.wk.weight", layers_prefix),
                format!("{}.self_attn.k_proj.weight", layers_prefix),
            ],
            kv_dim,
            hidden,
        ),
        // V projection: [kv_dim, hidden]
        (
            "v_proj",
            &[
                format!("{}.attn_v.weight", blk_prefix),
                format!("{}.attention.v_proj.weight", layers_prefix),
                format!("{}.attention.wv.weight", layers_prefix),
                format!("{}.self_attn.v_proj.weight", layers_prefix),
            ],
            kv_dim,
            hidden,
        ),
        // O projection: [hidden, q_dim]
        (
            "o_proj",
            &[
                format!("{}.attn_output.weight", blk_prefix),
                format!("{}.attn_o.weight", blk_prefix),
                format!("{}.attention.o_proj.weight", layers_prefix),
                format!("{}.attention.wo.weight", layers_prefix),
                format!("{}.self_attn.o_proj.weight", layers_prefix),
            ],
            hidden,
            q_dim,
        ),
    ];

    for (name, keys, out_dim, in_dim) in attn_keys {
        let key_strs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        if let Some(tensor) = find_and_remove(tensors, &key_strs) {
            let normalized = ensure_matrix_or_transpose(
                tensor,
                out_dim,
                in_dim,
                &format!("layer{}.attention.{}", layer_idx, name),
            )?;
            tensors.insert(format!("layers.{}.attention.{}.weight", layer_idx, name), normalized);
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
        if let Some(tensor) = find_and_remove(tensors, &key_strs) {
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
    let dims = ModelDims {
        hidden: config.model.hidden_size,
        n_head: config.model.num_heads,
        n_kv_head: if config.model.num_key_value_heads > 0 {
            config.model.num_key_value_heads
        } else {
            config.model.num_heads // MHA fallback
        },
        inter: config.model.intermediate_size,
        vocab: config.model.vocab_size,
    };

    tracing::info!(
        "Model dimensions from config: hidden={}, n_head={}, n_kv_head={}, inter={}, vocab={}, num_layers={}",
        dims.hidden,
        dims.n_head,
        dims.n_kv_head,
        dims.inter,
        dims.vocab,
        config.model.num_layers
    );

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
}
