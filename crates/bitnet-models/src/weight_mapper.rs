/// Weight mapping utilities for loading model weights from various formats

use std::collections::HashMap;
use candle_core::{Tensor, DType, Device};
use bitnet_common::Result;

/// Map GGUF tensor names to transformer module names
pub fn remap_gguf_weights(
    tensors: &HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut mapped = HashMap::new();
    
    for (name, tensor) in tensors {
        // Common patterns for GGUF model names
        let new_name = if name.contains("tok_embeddings.weight") || name.contains("token_embd.weight") {
            "embed_tokens.weight".to_string()
        } else if name.contains("output.weight") || name.contains("lm_head.weight") {
            "lm_head.weight".to_string()
        } else if name.contains("norm.weight") || name.contains("output_norm.weight") {
            "norm.weight".to_string()
        } else if name.contains("norm.bias") || name.contains("output_norm.bias") {
            "norm.bias".to_string()
        } else if let Some(layer_idx) = extract_layer_index(name) {
            // Layer-specific mappings
            map_layer_weight(name, layer_idx)?
        } else {
            // Keep original name if no mapping found
            name.clone()
        };
        
        mapped.insert(new_name, tensor.clone());
    }
    
    Ok(mapped)
}

/// Extract layer index from tensor name
fn extract_layer_index(name: &str) -> Option<usize> {
    // Common patterns: "layers.0.", "blk.0.", "h.0."
    if let Some(pos) = name.find("layers.") {
        let after = &name[pos + 7..];
        if let Some(dot_pos) = after.find('.') {
            return after[..dot_pos].parse().ok();
        }
    }
    
    if let Some(pos) = name.find("blk.") {
        let after = &name[pos + 4..];
        if let Some(dot_pos) = after.find('.') {
            return after[..dot_pos].parse().ok();
        }
    }
    
    if let Some(pos) = name.find("h.") {
        let after = &name[pos + 2..];
        if let Some(dot_pos) = after.find('.') {
            return after[..dot_pos].parse().ok();
        }
    }
    
    None
}

/// Map layer-specific weight names
fn map_layer_weight(name: &str, layer_idx: usize) -> Result<String> {
    let prefix = format!("layers.{}", layer_idx);
    
    // Attention weights
    if name.contains("attention.wq") || name.contains("attn.q_proj") || name.contains("self_attn.q") {
        Ok(format!("{}.attention.q_proj.weight", prefix))
    } else if name.contains("attention.wk") || name.contains("attn.k_proj") || name.contains("self_attn.k") {
        Ok(format!("{}.attention.k_proj.weight", prefix))
    } else if name.contains("attention.wv") || name.contains("attn.v_proj") || name.contains("self_attn.v") {
        Ok(format!("{}.attention.v_proj.weight", prefix))
    } else if name.contains("attention.wo") || name.contains("attn.o_proj") || name.contains("self_attn.o") {
        Ok(format!("{}.attention.o_proj.weight", prefix))
    }
    // Feed-forward weights
    else if name.contains("feed_forward.w1") || name.contains("ffn.gate") || name.contains("mlp.gate") {
        Ok(format!("{}.feed_forward.gate_proj.weight", prefix))
    } else if name.contains("feed_forward.w3") || name.contains("ffn.up") || name.contains("mlp.up") {
        Ok(format!("{}.feed_forward.up_proj.weight", prefix))
    } else if name.contains("feed_forward.w2") || name.contains("ffn.down") || name.contains("mlp.down") {
        Ok(format!("{}.feed_forward.down_proj.weight", prefix))
    }
    // Layer norms
    else if name.contains("attention_norm") || name.contains("input_layernorm") {
        if name.contains("weight") {
            Ok(format!("{}.attention_norm.weight", prefix))
        } else {
            Ok(format!("{}.attention_norm.bias", prefix))
        }
    } else if name.contains("ffn_norm") || name.contains("post_attention_layernorm") {
        if name.contains("weight") {
            Ok(format!("{}.ffn_norm.weight", prefix))
        } else {
            Ok(format!("{}.ffn_norm.bias", prefix))
        }
    } else {
        // Keep original name with layer prefix
        Ok(name.to_string())
    }
}

/// Create a VarBuilder from mapped tensors
pub fn create_var_builder(
    tensors: HashMap<String, Tensor>,
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder> {
    // Convert tensors to the target dtype if needed
    let mut converted = HashMap::new();
    for (name, tensor) in tensors {
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?
        } else {
            tensor
        };
        
        // Move to target device if needed
        let tensor = if !tensor.device().same_device(device) {
            tensor.to_device(device)?
        } else {
            tensor
        };
        
        converted.insert(name, tensor);
    }
    
    Ok(candle_nn::VarBuilder::from_tensors(converted, dtype, device))
}