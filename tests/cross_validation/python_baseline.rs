//! Cross-validation tests against Python baseline

use std::process::Command;
use tempfile::TempDir;
use serde_json::Value;

/// Python baseline runner for cross-validation
pub struct PythonBaseline {
    python_path: String,
    script_dir: TempDir,
}

impl PythonBaseline {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let script_dir = TempDir::new()?;
        
        // Create Python validation script
        let script_content = r#"
import sys
import json
import numpy as np

def run_inference(model_path, prompt, config):
    """Run inference using original Python implementation"""
    # Placeholder - would call actual BitNet.cpp Python code
    return {
        "tokens": [1, 2, 3, 4, 5],
        "text": f"Python response to: {prompt}",
        "logits": [0.1, 0.2, 0.3, 0.4, 0.5]
    }

def quantize_tensor(data, qtype):
    """Quantize tensor using original implementation"""
    # Placeholder - would call actual quantization code
    return {
        "quantized": data.tolist(),
        "scales": [1.0],
        "qtype": qtype
    }

if __name__ == "__main__":
    command = sys.argv[1]
    
    if command == "inference":
        model_path = sys.argv[2]
        prompt = sys.argv[3]
        config = json.loads(sys.argv[4])
        result = run_inference(model_path, prompt, config)
        print(json.dumps(result))
    
    elif command == "quantize":
        data = json.loads(sys.argv[2])
        qtype = sys.argv[3]
        result = quantize_tensor(np.array(data), qtype)
        print(json.dumps(result))
"#;
        
        std::fs::write(
            script_dir.path().join("baseline.py"),
            script_content,
        )?;
        
        Ok(Self {
            python_path: "python".to_string(),
            script_dir,
        })
    }
    
    pub fn run_inference(
        &self,
        model_path: &str,
        prompt: &str,
        config: &Value,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let output = Command::new(&self.python_path)
            .arg(self.script_dir.path().join("baseline.py"))
            .arg("inference")
            .arg(model_path)
            .arg(prompt)
            .arg(config.to_string())
            .output()?;
        
        if !output.status.success() {
            return Err(format!(
                "Python script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ).into());
        }
        
        let result: Value = serde_json::from_slice(&output.stdout)?;
        Ok(result)
    }
    
    pub fn quantize_tensor(
        &self,
        data: &[f32],
        qtype: &str,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let data_json = serde_json::to_string(data)?;
        
        let output = Command::new(&self.python_path)
            .arg(self.script_dir.path().join("baseline.py"))
            .arg("quantize")
            .arg(data_json)
            .arg(qtype)
            .output()?;
        
        if !output.status.success() {
            return Err(format!(
                "Python script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ).into());
        }
        
        let result: Value = serde_json::from_slice(&output.stdout)?;
        Ok(result)
    }
}

/// Compare two floating point arrays with tolerance
pub fn compare_arrays_with_tolerance(
    rust_output: &[f32],
    python_output: &[f32],
    tolerance: f32,
) -> bool {
    if rust_output.len() != python_output.len() {
        return false;
    }
    
    rust_output
        .iter()
        .zip(python_output.iter())
        .all(|(r, p)| (r - p).abs() <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_python_baseline_inference() {
        let baseline = PythonBaseline::new().unwrap();
        let config = json!({
            "max_tokens": 100,
            "temperature": 1.0
        });
        
        let result = baseline
            .run_inference("dummy_model.gguf", "Hello", &config)
            .unwrap();
        
        assert!(result["text"].is_string());
        assert!(result["tokens"].is_array());
    }
    
    #[test]
    fn test_python_baseline_quantization() {
        let baseline = PythonBaseline::new().unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = baseline.quantize_tensor(&data, "I2_S").unwrap();
        
        assert!(result["quantized"].is_array());
        assert!(result["scales"].is_array());
    }
    
    #[test]
    fn test_array_comparison() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.01, 1.99, 3.01];
        
        assert!(compare_arrays_with_tolerance(&a, &b, 0.02));
        assert!(!compare_arrays_with_tolerance(&a, &b, 0.005));
    }
}