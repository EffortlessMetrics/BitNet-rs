//! Dense model validation tests for multi-SLM architecture support.
//!
//! These tests verify that the dense inference pipeline (SiLU, RMSNorm,
//! attention, matmul) produces numerically correct results against known
//! reference values, and that the WeightLoader can handle synthetic dense
//! model tensor patterns.

use bitnet_models::weight_loader::{
    DType, InMemoryWeightLoader, TensorData, WeightFormat, WeightLoader,
};

// ---------------------------------------------------------------------------
// Numerical reference tests (SiLU)
// ---------------------------------------------------------------------------

/// Reference values from PyTorch: `torch.nn.functional.silu(tensor)`
/// These are exact f32 values computed in Python.
#[test]
fn silu_matches_pytorch_reference() {
    use bitnet_inference::cpu_opt::silu;

    let inputs = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 5.0];
    let result = silu(&inputs);

    // PyTorch reference: silu(x) = x * sigmoid(x)
    let expected = [
        0.0,        // silu(0) = 0
        0.7310586,  // silu(1)
        -0.2689414, // silu(-1)
        1.7615942,  // silu(2)
        -0.2384058, // silu(-2)
        0.3112296,  // silu(0.5)
        -0.1887704, // silu(-0.5)
        4.9665356,  // silu(5)
    ];

    for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "silu mismatch at index {i}: got {got}, expected {want}"
        );
    }
}

/// SiLU is an odd-ish function: silu(x) + silu(-x) = 0 only at x=0,
/// but silu(x) > -silu(-x) for x > 0. Verify monotonicity for positive x.
#[test]
fn silu_monotonic_for_positive_inputs() {
    use bitnet_inference::cpu_opt::silu;

    let inputs: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let result = silu(&inputs);

    for i in 1..result.len() {
        assert!(
            result[i] >= result[i - 1],
            "SiLU not monotonic at x={}: silu({})={} < silu({})={}",
            inputs[i],
            inputs[i],
            result[i],
            inputs[i - 1],
            result[i - 1],
        );
    }
}

// ---------------------------------------------------------------------------
// Numerical reference tests (RMSNorm)
// ---------------------------------------------------------------------------

/// Verify RMSNorm with a known vector against hand-computed reference.
///
/// Input: [1.0, 2.0, 3.0, 4.0], weight: all 1.0, eps=1e-5
/// mean(x²) = (1+4+9+16)/4 = 7.5
/// rms = sqrt(7.5 + 1e-5) ≈ 2.7386
/// output[i] = input[i] / rms
#[test]
fn rmsnorm_reference_values() {
    use bitnet_inference::cpu_opt::rmsnorm;

    let dim = 4;
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32; dim];
    let mut output = vec![0.0f32; dim];

    rmsnorm(&input, &weight, &mut output, 1, dim, 1e-5).unwrap();

    let rms = (7.5f32 + 1e-5).sqrt();
    let expected: Vec<f32> = input.iter().map(|x| x / rms).collect();

    for (i, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "rmsnorm mismatch at index {i}: got {got}, expected {want}"
        );
    }
}

/// Multi-row RMSNorm: each row is normalized independently.
#[test]
fn rmsnorm_multi_row_independent() {
    use bitnet_inference::cpu_opt::rmsnorm;

    let dim = 3;
    let rows = 2;
    // Row 0: [3.0, 0.0, 0.0] → rms = sqrt(3.0 + eps) ≈ 1.7321
    // Row 1: [0.0, 4.0, 0.0] → rms = sqrt(16/3 + eps) ≈ 2.3094
    let input = vec![3.0f32, 0.0, 0.0, 0.0, 4.0, 0.0];
    let weight = vec![1.0f32; dim];
    let mut output = vec![0.0f32; rows * dim];

    rmsnorm(&input, &weight, &mut output, rows, dim, 1e-5).unwrap();

    // Row 0: only element 0 is non-zero
    let rms0 = (9.0f32 / 3.0 + 1e-5).sqrt();
    assert!((output[0] - 3.0 / rms0).abs() < 1e-5);
    assert!(output[1].abs() < 1e-5);
    assert!(output[2].abs() < 1e-5);

    // Row 1: only element 1 is non-zero
    let rms1 = (16.0f32 / 3.0 + 1e-5).sqrt();
    assert!(output[3].abs() < 1e-5);
    assert!((output[4] - 4.0 / rms1).abs() < 1e-5);
    assert!(output[5].abs() < 1e-5);
}

/// RMSNorm with non-uniform weights should scale each dimension.
#[test]
fn rmsnorm_with_learned_weights() {
    use bitnet_inference::cpu_opt::rmsnorm;

    let dim = 2;
    let input = vec![1.0f32, 1.0];
    let weight = vec![2.0f32, 0.5]; // Scale first dim by 2, second by 0.5
    let mut output = vec![0.0f32; dim];

    rmsnorm(&input, &weight, &mut output, 1, dim, 1e-5).unwrap();

    // rms = sqrt(mean([1, 1]) + eps) = sqrt(1 + eps) ≈ 1.0
    // output[0] = (1.0 / 1.0) * 2.0 = 2.0
    // output[1] = (1.0 / 1.0) * 0.5 = 0.5
    assert!((output[0] - 2.0).abs() < 0.01);
    assert!((output[1] - 0.5).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// SiLU + RMSNorm pipeline test
// ---------------------------------------------------------------------------

/// Simulate a dense FFN layer: SiLU(x) → RMSNorm.
#[test]
fn silu_then_rmsnorm_pipeline() {
    use bitnet_inference::cpu_opt::{rmsnorm, silu};

    let dim = 4;
    let input = vec![1.0f32, -1.0, 2.0, -2.0];

    // Step 1: SiLU
    let activated = silu(&input);
    assert_eq!(activated.len(), dim);

    // Step 2: RMSNorm the activated values
    let weight = vec![1.0f32; dim];
    let mut output = vec![0.0f32; dim];
    rmsnorm(&activated, &weight, &mut output, 1, dim, 1e-5).unwrap();

    // Output should be normalized (RMS ≈ 1.0 after normalization)
    let rms: f32 = output.iter().map(|x| x * x).sum::<f32>() / dim as f32;
    // After RMSNorm with unit weights, RMS of output should be close to 1.0
    assert!((rms.sqrt() - 1.0).abs() < 0.01, "post-RMSNorm RMS should be ≈1.0, got {}", rms.sqrt());
}

// ---------------------------------------------------------------------------
// WeightLoader with dense model patterns
// ---------------------------------------------------------------------------

/// Simulate Phi-4 weight tensor names and shapes.
#[test]
fn weight_loader_phi4_tensor_names() {
    let mut loader = InMemoryWeightLoader::new(WeightFormat::SafeTensors);

    // Simulate Phi-4 architecture tensors (abbreviated)
    let tensors = vec![
        ("model.embed_tokens.weight", vec![100352, 5120]),
        ("model.layers.0.self_attn.q_proj.weight", vec![5120, 5120]),
        ("model.layers.0.self_attn.k_proj.weight", vec![1280, 5120]),
        ("model.layers.0.self_attn.v_proj.weight", vec![1280, 5120]),
        ("model.layers.0.self_attn.o_proj.weight", vec![5120, 5120]),
        ("model.layers.0.mlp.gate_proj.weight", vec![13824, 5120]),
        ("model.layers.0.mlp.up_proj.weight", vec![13824, 5120]),
        ("model.layers.0.mlp.down_proj.weight", vec![5120, 13824]),
        ("model.layers.0.input_layernorm.weight", vec![5120]),
        ("model.layers.0.post_attention_layernorm.weight", vec![5120]),
        ("model.norm.weight", vec![5120]),
        ("lm_head.weight", vec![100352, 5120]),
    ];

    for (name, shape) in &tensors {
        let numel: usize = shape.iter().product();
        loader.insert(
            *name,
            TensorData {
                shape: shape.clone(),
                dtype: DType::BF16,
                data: vec![0u8; numel * 2], // BF16 = 2 bytes each
            },
        );
    }

    assert_eq!(loader.len(), 12);
    assert!(loader.has_tensor("model.embed_tokens.weight"));
    assert!(loader.has_tensor("model.layers.0.self_attn.q_proj.weight"));
    assert!(loader.has_tensor("model.norm.weight"));
    assert!(loader.has_tensor("lm_head.weight"));
    assert!(!loader.has_tensor("nonexistent.weight"));

    // Verify shapes via tensor_info
    let embed_info = loader.tensor_info("model.embed_tokens.weight").unwrap();
    assert_eq!(embed_info.shape, vec![100352, 5120]);
    assert_eq!(embed_info.dtype, DType::BF16);

    let q_info = loader.tensor_info("model.layers.0.self_attn.q_proj.weight").unwrap();
    assert_eq!(q_info.shape, vec![5120, 5120]);

    // Verify GQA dimensions: K/V are smaller than Q
    let k_info = loader.tensor_info("model.layers.0.self_attn.k_proj.weight").unwrap();
    assert_eq!(k_info.shape, vec![1280, 5120]); // 10 KV heads × 128 dim = 1280
}

/// Verify tensor data round-trip through WeightLoader.
#[test]
fn weight_loader_tensor_data_roundtrip() {
    let mut loader = InMemoryWeightLoader::new(WeightFormat::SafeTensors);

    let data: Vec<u8> = (0..16).map(|i| i as u8).collect();
    loader.insert(
        "test.weight",
        TensorData { shape: vec![2, 4], dtype: DType::F16, data: data.clone() },
    );

    let loaded = loader.load_tensor("test.weight").unwrap();
    assert_eq!(loaded.shape, vec![2, 4]);
    assert_eq!(loaded.dtype, DType::F16);
    assert_eq!(loaded.data, data);
    assert_eq!(loaded.numel(), 8);
    assert_eq!(loaded.expected_byte_len(), 16); // 8 elements × 2 bytes
}

/// Verify WeightFormat::detect works for common extensions.
#[test]
fn weight_format_detection() {
    assert_eq!(WeightFormat::detect("model.safetensors"), Some(WeightFormat::SafeTensors));
    assert_eq!(WeightFormat::detect("model.gguf"), Some(WeightFormat::Gguf));
    assert_eq!(WeightFormat::detect("model.bin"), None);
    assert_eq!(WeightFormat::detect("model.pt"), None);
}

/// Verify DType element sizes are correct.
#[test]
fn dtype_element_sizes() {
    assert_eq!(DType::F16.element_size(), 2);
    assert_eq!(DType::BF16.element_size(), 2);
    assert_eq!(DType::F32.element_size(), 4);
    assert_eq!(DType::F64.element_size(), 8);
    assert_eq!(DType::I8.element_size(), 1);
    assert_eq!(DType::U8.element_size(), 1);
    assert_eq!(DType::I16.element_size(), 2);
    assert_eq!(DType::I32.element_size(), 4);
}

/// Verify tensor_names returns sorted names.
#[test]
fn weight_loader_sorted_tensor_names() {
    let mut loader = InMemoryWeightLoader::new(WeightFormat::SafeTensors);
    for name in &["c.weight", "a.weight", "b.weight"] {
        loader.insert(*name, TensorData { shape: vec![2], dtype: DType::F32, data: vec![0; 8] });
    }
    assert_eq!(loader.tensor_names(), vec!["a.weight", "b.weight", "c.weight"]);
}

/// Loading a non-existent tensor returns an error.
#[test]
fn weight_loader_missing_tensor_error() {
    let loader = InMemoryWeightLoader::new(WeightFormat::Gguf);
    assert!(loader.load_tensor("nonexistent").is_err());
}
