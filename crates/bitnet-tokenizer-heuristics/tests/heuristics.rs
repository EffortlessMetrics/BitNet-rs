use bitnet_tokenizer_heuristics::{
    NamedTensorShape, default_vocab_for_architecture, detect_architecture_from_name,
    detect_architecture_from_tensor_names, infer_vocab_from_embedding_tensors,
};

#[test]
fn detects_architecture_from_name_patterns() {
    assert_eq!(detect_architecture_from_name("BitNet-b1.58"), Some("bitnet"));
    assert_eq!(detect_architecture_from_name("llama-3.1-8b"), Some("llama"));
    assert_eq!(detect_architecture_from_name("unknown"), None);
}

#[test]
fn detects_architecture_from_tensor_patterns() {
    let llama = ["layers.0.attention.wq.weight", "layers.0.attention.wk.weight"];
    assert_eq!(detect_architecture_from_tensor_names(&llama), "llama");

    let gpt2 = ["h.0.attn.c_attn.weight", "h.0.mlp.c_fc.weight"];
    assert_eq!(detect_architecture_from_tensor_names(&gpt2), "gpt2");
}

#[test]
fn infers_vocab_from_embedding_tensor_shape() {
    let tensors = [
        NamedTensorShape { name: "layers.0.attn", shape: &[32, 32] },
        NamedTensorShape { name: "tok_embeddings.weight", shape: &[128_256, 4096] },
    ];
    assert_eq!(infer_vocab_from_embedding_tensors(&tensors), Some(128_256));
}

#[test]
fn provides_default_vocab_by_architecture() {
    assert_eq!(default_vocab_for_architecture("llama", Some("llama-3")), Some(128_256));
    assert_eq!(default_vocab_for_architecture("bert", None), Some(30_522));
    assert_eq!(default_vocab_for_architecture("custom", None), None);
}
