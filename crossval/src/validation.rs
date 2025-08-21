//! Validation tests for BitNet.rs
//!
//! Comprehensive validation suite ensuring correctness against C++ implementation.

use crate::Result;
use bitnet_models::{weight_mapper, GgufReader};

/// Test tensor name mapping without loading actual tensors
pub fn dry_run_remap_names(tensor_names: Vec<String>) -> Vec<String> {
    weight_mapper::dry_run_remap_names(tensor_names)
}

/// Run comprehensive validation against a GGUF model
pub fn validate_model(model_path: &str) -> Result<ValidationReport> {
    let bytes = std::fs::read(model_path)?;
    let reader = GgufReader::new(&bytes)
        .map_err(|e| crate::CrossvalError::ModelLoadError(e.to_string()))?;
    
    // Get tensor names
    let tensor_names: Vec<String> = reader
        .tensor_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    
    // Test mapping
    let unmapped = dry_run_remap_names(tensor_names.clone());
    
    // Count metadata
    let n_kv = reader.metadata_keys().len();
    let n_tensors = reader.tensor_count();
    
    Ok(ValidationReport {
        model_path: model_path.to_string(),
        n_tensors: n_tensors as usize,
        n_kv: n_kv as usize,
        unmapped_count: unmapped.len(),
        unmapped_tensors: unmapped,
        tensor_names,
    })
}

/// Validation report for a model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationReport {
    pub model_path: String,
    pub n_tensors: usize,
    pub n_kv: usize,
    pub unmapped_count: usize,
    pub unmapped_tensors: Vec<String>,
    pub tensor_names: Vec<String>,
}

impl ValidationReport {
    /// Check if validation passed (no unmapped tensors)
    pub fn is_valid(&self) -> bool {
        self.unmapped_count == 0
    }
    
    /// Get a summary of the validation
    pub fn summary(&self) -> String {
        format!(
            "Model: {}\nTensors: {}\nMetadata KV: {}\nUnmapped: {}\nStatus: {}",
            self.model_path,
            self.n_tensors,
            self.n_kv,
            self.unmapped_count,
            if self.is_valid() { "✅ VALID" } else { "❌ INVALID" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dry_run_known_tensors() {
        // Test with known tensor names
        let names = vec![
            "blk.0.attn_norm.weight".to_string(),
            "blk.0.ffn_down.weight".to_string(),
            "blk.0.ffn_gate.weight".to_string(),
            "blk.0.ffn_up.weight".to_string(),
            "blk.0.ffn_norm.weight".to_string(),
            "blk.0.attn_k.weight".to_string(),
            "blk.0.attn_output.weight".to_string(),
            "blk.0.attn_q.weight".to_string(),
            "blk.0.attn_v.weight".to_string(),
        ];
        
        let unmapped = dry_run_remap_names(names);
        assert_eq!(unmapped.len(), 0, "All standard tensors should map");
    }
    
    #[test]
    fn test_dry_run_unmapped() {
        // Test with unknown tensor names
        let names = vec![
            "unknown.tensor.name".to_string(),
            "not.a.real.tensor".to_string(),
        ];
        
        let unmapped = dry_run_remap_names(names.clone());
        assert_eq!(unmapped.len(), 2, "Unknown tensors should be unmapped");
        assert_eq!(unmapped, names);
    }
    
    #[test]
    #[ignore] // Requires actual model file
    fn test_ms_bitnet_validation() {
        let model_path = "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping: model not found at {}", model_path);
            return;
        }
        
        let report = validate_model(model_path).expect("validation should succeed");
        println!("{}", report.summary());
        
        assert!(report.is_valid(), "MS BitNet model should have all tensors mapped");
        assert!(report.n_tensors > 0, "Should have tensors");
        assert!(report.n_kv > 0, "Should have metadata");
    }
}