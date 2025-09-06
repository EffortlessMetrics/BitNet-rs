use bitnet_common::{BitNetError, Device, Result};
use candle_core::{DType, Device as CDevice, Tensor as CandleTensor};
use std::collections::HashMap;
use std::path::Path;

/// Load a GGUF model file - simplified version for CLI
///
/// This helper loads the token embeddings and output projection from the
/// provided GGUF file. Remaining tensors are zero-initialized based on the
/// default configuration so a model can still be constructed for testing.
pub fn load_gguf(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    let two =
        crate::gguf_min::load_two(path).map_err(|e| BitNetError::Validation(e.to_string()))?;

    // Start from default config and update basic dimensions from the file
    let mut config = bitnet_common::BitNetConfig::default();
    config.model.vocab_size = two.vocab;
    config.model.hidden_size = two.dim;

    let num_layers = config.model.num_layers;
    let intermediate_size = config.model.intermediate_size;
    let hidden_size = config.model.hidden_size;
    let vocab_size = config.model.vocab_size;

    let cdevice = match device {
        Device::Cpu => CDevice::Cpu,
        Device::Cuda(id) => {
            CDevice::new_cuda(id).map_err(|e| BitNetError::Validation(e.to_string()))?
        }
        Device::Metal => {
            return Err(BitNetError::Validation("Metal not yet supported".to_string()));
        }
    };

    let dtype = DType::F32;
    let mut tensor_map = HashMap::new();

    tensor_map.insert(
        "token_embd.weight".to_string(),
        CandleTensor::from_vec(two.tok_embeddings, (vocab_size, hidden_size), &cdevice)?,
    );
    tensor_map.insert(
        "output.weight".to_string(),
        CandleTensor::from_vec(two.lm_head, (hidden_size, vocab_size), &cdevice)?,
    );

    for layer in 0..num_layers {
        let prefix = format!("blk.{}", layer);

        tensor_map.insert(
            format!("{}.attn_q.weight", prefix),
            CandleTensor::zeros(&[hidden_size, hidden_size], dtype, &cdevice)?,
        );
        tensor_map.insert(
            format!("{}.attn_k.weight", prefix),
            CandleTensor::zeros(&[hidden_size, hidden_size], dtype, &cdevice)?,
        );
        tensor_map.insert(
            format!("{}.attn_v.weight", prefix),
            CandleTensor::zeros(&[hidden_size, hidden_size], dtype, &cdevice)?,
        );
        tensor_map.insert(
            format!("{}.attn_output.weight", prefix),
            CandleTensor::zeros(&[hidden_size, hidden_size], dtype, &cdevice)?,
        );

        tensor_map.insert(
            format!("{}.ffn_gate.weight", prefix),
            CandleTensor::zeros(&[intermediate_size, hidden_size], dtype, &cdevice)?,
        );
        tensor_map.insert(
            format!("{}.ffn_up.weight", prefix),
            CandleTensor::zeros(&[intermediate_size, hidden_size], dtype, &cdevice)?,
        );
        tensor_map.insert(
            format!("{}.ffn_down.weight", prefix),
            CandleTensor::zeros(&[hidden_size, intermediate_size], dtype, &cdevice)?,
        );

        tensor_map.insert(
            format!("{}.attn_norm.weight", prefix),
            CandleTensor::ones(&[hidden_size], dtype, &cdevice)?,
        );
        tensor_map.insert(
            format!("{}.ffn_norm.weight", prefix),
            CandleTensor::ones(&[hidden_size], dtype, &cdevice)?,
        );
    }

    tensor_map.insert(
        "output_norm.weight".to_string(),
        CandleTensor::ones(&[hidden_size], dtype, &cdevice)?,
    );

    Ok((config, tensor_map))
}
