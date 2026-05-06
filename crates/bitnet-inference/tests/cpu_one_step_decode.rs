use anyhow::Result;
use bitnet_common::{BitNetConfig, ConcreteTensor, Device, ModelConfig, Tensor as _};
use bitnet_inference::{CPU_DECODE_SCALAR_KERNEL_FAMILY, decode_one_cpu_token};
use bitnet_models::{BitNetModel, transformer::KVCache as TransformerKVCache};
use bitnet_quantization::i2s_qk256::QK256_SCALAR_GEMV_KERNEL_ID;
use candle_core::Tensor as CandleTensor;
use std::{collections::HashMap, sync::Arc};

fn tiny_config() -> BitNetConfig {
    BitNetConfig {
        model: ModelConfig {
            vocab_size: 32,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            num_key_value_heads: 4,
            intermediate_size: 64,
            max_position_embeddings: 16,
            rope_theta: Some(10_000.0),
            rms_norm_eps: Some(1e-5),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn tiny_tensor(len: usize, scale: f32, phase: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32 * scale) + phase).sin() * 0.05).collect()
}

fn add_norm(
    tensors: &mut HashMap<String, CandleTensor>,
    prefix: &str,
    hidden: usize,
    device: &candle_core::Device,
) -> Result<()> {
    tensors.insert(
        format!("{prefix}.weight"),
        CandleTensor::from_vec(vec![1.0f32; hidden], &[hidden], device)?,
    );
    tensors.insert(
        format!("{prefix}.bias"),
        CandleTensor::from_vec(vec![0.0f32; hidden], &[hidden], device)?,
    );
    Ok(())
}

fn tiny_weighted_model() -> Result<(Arc<BitNetModel>, BitNetConfig)> {
    let config = tiny_config();
    let device = candle_core::Device::Cpu;
    let vocab = config.model.vocab_size;
    let hidden = config.model.hidden_size;
    let intermediate = config.model.intermediate_size;
    let mut tensors = HashMap::new();

    tensors.insert(
        "token_embd.weight".to_string(),
        CandleTensor::from_vec(tiny_tensor(vocab * hidden, 0.013, 0.1), &[vocab, hidden], &device)?,
    );
    tensors.insert(
        "output.weight".to_string(),
        CandleTensor::from_vec(tiny_tensor(vocab * hidden, 0.017, 0.4), &[vocab, hidden], &device)?,
    );

    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        tensors.insert(
            format!("layers.0.self_attn.{name}.weight"),
            CandleTensor::from_vec(
                tiny_tensor(hidden * hidden, 0.007, name.len() as f32),
                &[hidden, hidden],
                &device,
            )?,
        );
    }

    tensors.insert(
        "layers.0.mlp.gate_proj.weight".to_string(),
        CandleTensor::from_vec(
            tiny_tensor(intermediate * hidden, 0.005, 0.7),
            &[intermediate, hidden],
            &device,
        )?,
    );
    tensors.insert(
        "layers.0.mlp.up_proj.weight".to_string(),
        CandleTensor::from_vec(
            tiny_tensor(intermediate * hidden, 0.006, 0.8),
            &[intermediate, hidden],
            &device,
        )?,
    );
    tensors.insert(
        "layers.0.mlp.down_proj.weight".to_string(),
        CandleTensor::from_vec(
            tiny_tensor(hidden * intermediate, 0.004, 0.9),
            &[hidden, intermediate],
            &device,
        )?,
    );

    add_norm(&mut tensors, "layers.0.attention_norm", hidden, &device)?;
    add_norm(&mut tensors, "layers.0.ffn_norm", hidden, &device)?;
    add_norm(&mut tensors, "final_norm", hidden, &device)?;

    let model = BitNetModel::from_gguf(config.clone(), tensors, HashMap::new(), Device::Cpu)?;
    Ok((Arc::new(model), config))
}

fn logits_vec(logits: &ConcreteTensor) -> Result<Vec<f32>> {
    Ok(logits.to_candle()?.flatten_all()?.to_vec1::<f32>()?)
}

#[test]
fn one_step_cpu_decode_uses_real_tensors_and_updates_transformer_kv_cache() -> Result<()> {
    let (model, config) = tiny_weighted_model()?;
    let mut kv_cache = TransformerKVCache::new(&config, 1, &candle_core::Device::Cpu)?;

    let first = decode_one_cpu_token(
        model.as_ref(),
        &mut kv_cache,
        7,
        Some(QK256_SCALAR_GEMV_KERNEL_ID),
        true,
    )?;

    assert_eq!(first.logits.shape(), &[1, 1, config.model.vocab_size]);
    assert_eq!(first.report.requested_kernel, Some(QK256_SCALAR_GEMV_KERNEL_ID));
    assert_eq!(first.report.selected_kernel, QK256_SCALAR_GEMV_KERNEL_ID);
    assert_eq!(first.report.selected_kernel_family, CPU_DECODE_SCALAR_KERNEL_FAMILY);
    assert!(!first.report.fallback_used);
    assert_eq!(first.report.kv_cache_seq_lens, vec![1]);
    assert!(first.report.ops.embedding_gather);
    assert!(first.report.ops.transformer_layers);
    assert!(first.report.ops.kv_cache_append_read);
    assert!(first.report.ops.logits_output_head);
    assert!(first.report.ops.sampling_handoff);

    let values = logits_vec(&first.logits)?;
    assert_eq!(values.len(), config.model.vocab_size);
    assert!(
        values.windows(2).any(|pair| (pair[0] - pair[1]).abs() > 1e-8),
        "real weighted decode logits should not collapse to identical values"
    );

    let second = decode_one_cpu_token(
        model.as_ref(),
        &mut kv_cache,
        11,
        Some(QK256_SCALAR_GEMV_KERNEL_ID),
        true,
    )?;
    assert_eq!(second.report.kv_cache_seq_lens, vec![2]);

    Ok(())
}

#[test]
fn one_step_cpu_decode_is_deterministic_for_same_model_token_and_cache_state() -> Result<()> {
    let (model, config) = tiny_weighted_model()?;
    let mut left_cache = TransformerKVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let mut right_cache = TransformerKVCache::new(&config, 1, &candle_core::Device::Cpu)?;

    let left = decode_one_cpu_token(
        model.as_ref(),
        &mut left_cache,
        5,
        Some(QK256_SCALAR_GEMV_KERNEL_ID),
        true,
    )?;
    let right = decode_one_cpu_token(
        model.as_ref(),
        &mut right_cache,
        5,
        Some(QK256_SCALAR_GEMV_KERNEL_ID),
        true,
    )?;

    assert_eq!(left.report.kv_cache_seq_lens, right.report.kv_cache_seq_lens);
    assert_eq!(left.report.selected_kernel, right.report.selected_kernel);
    assert_eq!(logits_vec(&left.logits)?, logits_vec(&right.logits)?);
    Ok(())
}
