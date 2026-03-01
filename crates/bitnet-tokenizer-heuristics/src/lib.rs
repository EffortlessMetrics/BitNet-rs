//! Pure heuristics used by tokenizer discovery.

/// Lightweight tensor shape view for heuristic inference.
#[derive(Debug, Clone, Copy)]
pub struct NamedTensorShape<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
}

/// Detect architecture from a model display/name string.
#[must_use]
pub fn detect_architecture_from_name(name: &str) -> Option<&'static str> {
    let name_lower = name.to_lowercase();
    let name_patterns = [
        ("bitnet", &["bitnet", "bitlinear"] as &[&str]),
        ("llama", &["llama"]),
        ("gpt2", &["gpt2", "gpt-2"]),
        ("gptneox", &["gpt-neo", "gptneox", "gpt-j"]),
        ("bert", &["bert"]),
        ("t5", &["t5"]),
    ];

    name_patterns.iter().find_map(|(arch, patterns)| {
        patterns.iter().any(|p| name_lower.contains(p)).then_some(*arch)
    })
}

/// Detect architecture from tensor naming patterns.
#[must_use]
pub fn detect_architecture_from_tensor_names(tensor_names: &[&str]) -> &'static str {
    let architecture_patterns = [
        ("bitnet", &["bitlinear", "bitnet"] as &[&str]),
        ("llama", &["attn_q", "attn_k", "attn_v", "attention.wq", "attention.wk"]),
        ("t5", &["encoder", "decoder", "relative_attention_bias"]),
        ("bert", &["encoder", "self", "attention"]),
        ("gptneox", &["gpt_neox", "gptneox"]),
    ];

    for (arch, patterns) in architecture_patterns {
        let has_patterns = if arch == "bert" || arch == "t5" {
            patterns.iter().all(|p| tensor_names.iter().any(|n| n.contains(p)))
        } else {
            patterns.iter().any(|p| tensor_names.iter().any(|n| n.contains(p)))
        };

        if has_patterns {
            return arch;
        }
    }

    let has_mlp = tensor_names.iter().any(|name| name.contains("mlp") || name.contains("c_fc"));
    let has_attn = tensor_names.iter().any(|name| name.contains("attn") || name.contains("c_attn"));

    if has_mlp && has_attn { "gpt2" } else { "transformer" }
}

/// Architecture-specific default vocabulary sizes.
#[must_use]
pub fn default_vocab_for_architecture(arch: &str, model_name: Option<&str>) -> Option<usize> {
    match arch {
        "llama" => {
            if let Some(name) = model_name {
                if name.contains("llama-3") || name.contains("llama3") {
                    Some(128_256)
                } else {
                    Some(32_000)
                }
            } else {
                Some(32_000)
            }
        }
        "gpt2" | "gptneox" => Some(50_257),
        "bert" => Some(30_522),
        "t5" => Some(32_128),
        _ => None,
    }
}

/// Infer vocabulary size from likely embedding tensor shapes.
#[must_use]
pub fn infer_vocab_from_embedding_tensors(tensors: &[NamedTensorShape<'_>]) -> Option<usize> {
    for tensor in tensors {
        if (tensor.name.contains("token_embd")
            || tensor.name.contains("wte")
            || tensor.name.contains("embed")
            || tensor.name.contains("embeddings"))
            && !tensor.shape.is_empty()
        {
            let possible_vocab = tensor.shape[0];
            if (100..2_000_000).contains(&possible_vocab) {
                return Some(possible_vocab);
            }
        }
    }
    None
}
