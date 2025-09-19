use std::collections::HashMap;
use candle_core::{Device, DType, Tensor};

fn main() -> anyhow::Result<()> {
    // Test tensor name mapping with Microsoft BitNet names
    let mut tensors = HashMap::new();
    
    // Simulate Microsoft BitNet tensor names
    tensors.insert("blk.0.attn_q.weight".to_string(), Tensor::zeros((512, 512), DType::F32, &Device::Cpu)?);
    tensors.insert("blk.0.ffn_gate.weight".to_string(), Tensor::zeros((512, 2048), DType::F32, &Device::Cpu)?);
    tensors.insert("layers.0.self_attn.q_proj.weight".to_string(), Tensor::zeros((512, 512), DType::F32, &Device::Cpu)?);
    tensors.insert("layers.0.mlp.gate_proj.weight".to_string(), Tensor::zeros((512, 2048), DType::F32, &Device::Cpu)?);
    
    println!("Original tensor names:");
    for name in tensors.keys() {
        println!("  {}", name);
    }
    
    // Apply our weight mapping
    let mapped = bitnet_models::weight_mapper::remap_gguf_weights(&tensors)?;
    
    println!("\nMapped tensor names:");
    for name in mapped.keys() {
        println!("  {}", name);
    }
    
    // Check that mapping worked correctly
    assert!(mapped.contains_key("layers.0.attention.q_proj.weight"));
    assert!(mapped.contains_key("layers.0.feed_forward.gate_proj.weight"));
    
    println!("\n✅ Weight mapping test passed!");
    Ok(())
}
