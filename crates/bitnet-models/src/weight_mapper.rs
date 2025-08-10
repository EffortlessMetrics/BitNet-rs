/// Weight mapping utilities for loading model weights from various formats

use std::collections::HashMap;
use candle_core::{Tensor, DType, Device};
use bitnet_common::Result;

/// Map GGUF tensor names to transformer module names
pub fn remap_gguf_weights(
    tensors: &HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut mapped = HashMap::new();
    let mut unmapped = Vec::new();
    
    for (name, tensor) in tensors {
        let new_name = if let Some(mapped_name) = map_tensor_name(name) {
            mapped_name
        } else {
            unmapped.push(name.clone());
            name.clone()
        };
        
        mapped.insert(new_name, tensor.clone());
    }
    
    // Log unmapped tensors for debugging
    if !unmapped.is_empty() {
        eprintln!("Warning: {} unmapped tensors: {:?}", unmapped.len(), &unmapped[..5.min(unmapped.len())]);
    }
    
    Ok(mapped)
}

/// Map individual tensor name from GGUF to our transformer naming
fn map_tensor_name(name: &str) -> Option<String> {
    // Token embeddings variations
    if name == "token_embd.weight" || name == "tok_embeddings.weight" || name == "model.embed_tokens.weight" {
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
                
                // Feed-forward weights
                "ffn_gate.weight" | "ffn_gate_inp.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" | "ffn_up_proj.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" | "ffn_down_proj.weight" => "mlp.down_proj.weight",
                
                // FFN normalization
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                
                _ => return None,
            };
            
            return Some(format!("layers.{}.{}", layer_num, mapped_component));
        }
    }
    
    // Handle "layers.N." prefix (LLaMA style)
    if name.starts_with("layers.") || name.starts_with("model.layers.") {
        let clean_name = if name.starts_with("model.") {
            &name[6..] // Remove "model." prefix
        } else {
            name
        };
        
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
                "ffn_norm.weight" | "post_attention_layernorm.weight" => "post_attention_layernorm.weight",
                
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