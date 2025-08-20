use std::path::Path;
use anyhow::{Result, anyhow};
use log::{info, warn};
use std::collections::HashMap;

use super::reader::GgufReader;
use super::writer::GgufWriter;

/// GGUF compatibility fixer that auto-patches missing metadata
pub struct GgufCompatibilityFixer;

impl GgufCompatibilityFixer {
    /// Diagnose compatibility issues in a GGUF file
    pub fn diagnose<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
        let reader = GgufReader::open(path)?;
        let mut issues = Vec::new();
        
        // Check tokenizer model
        let tokenizer_model = reader.tokenizer_kind()?.unwrap_or_default();
        
        // Check for missing pre-tokenizer (critical for llama.cpp)
        if tokenizer_model == "gpt2" || tokenizer_model == "llama3" {
            let has_pre = reader.value("tokenizer.ggml.pre").ok().flatten().is_some();
            if !has_pre {
                issues.push(format!(
                    "Missing tokenizer.ggml.pre for {} tokenizer - llama.cpp will fail",
                    tokenizer_model
                ));
            }
        }
        
        // Check for add_space_prefix
        if tokenizer_model == "gpt2" {
            let has_space_prefix = reader.value("tokenizer.ggml.add_space_prefix")
                .ok()
                .flatten()
                .is_some();
            if !has_space_prefix {
                issues.push("Missing tokenizer.ggml.add_space_prefix for GPT-2 tokenizer".to_string());
            }
        }
        
        // Check special tokens
        let bos_id = reader.bos_token_id()?.unwrap_or(None);
        let eos_id = reader.eos_token_id()?.unwrap_or(None);
        
        if bos_id.is_none() && (tokenizer_model == "llama" || tokenizer_model == "gpt2") {
            issues.push("Missing BOS token ID".to_string());
        }
        
        if eos_id.is_none() {
            issues.push("Missing EOS token ID".to_string());
        }
        
        // Check for byte_fallback
        if tokenizer_model == "gpt2" || tokenizer_model == "llama3" {
            let has_byte_fallback = reader.value("tokenizer.ggml.byte_fallback")
                .ok()
                .flatten()
                .is_some();
            if !has_byte_fallback {
                issues.push("Missing tokenizer.ggml.byte_fallback setting".to_string());
            }
        }
        
        // Check vocab size consistency
        let vocab_size = reader.vocab_size()?;
        let vocab_entries = reader.tokenizer_vocab()?.len();
        if vocab_size != vocab_entries {
            issues.push(format!(
                "Vocabulary size mismatch: metadata says {} but found {} entries",
                vocab_size, vocab_entries
            ));
        }
        
        // Check for unknown words handling
        let has_unk = reader.value("tokenizer.ggml.unknown_token_id")
            .ok()
            .flatten()
            .is_some();
        if !has_unk && tokenizer_model != "llama" {
            issues.push("Missing unknown token ID".to_string());
        }
        
        Ok(issues)
    }
    
    /// Apply fixes to a GGUF file and write to a new file
    pub fn apply_fixes<P: AsRef<Path>, Q: AsRef<Path>>(
        input_path: P,
        output_path: Q,
    ) -> Result<()> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();
        
        info!("Fixing GGUF compatibility issues: {} -> {}", 
              input_path.display(), output_path.display());
        
        // Read the original file
        let reader = GgufReader::open(input_path)?;
        let mut metadata = HashMap::new();
        
        // Copy all existing metadata
        for (key, value) in reader.metadata()? {
            metadata.insert(key, value);
        }
        
        // Apply fixes based on tokenizer type
        let tokenizer_model = metadata.get("tokenizer.ggml.model")
            .and_then(|v| v.as_string())
            .unwrap_or("unknown");
        
        match tokenizer_model {
            "gpt2" | "llama3" => {
                // Fix missing pre-tokenizer
                if !metadata.contains_key("tokenizer.ggml.pre") {
                    warn!("Adding missing tokenizer.ggml.pre = gpt2");
                    metadata.insert("tokenizer.ggml.pre".to_string(), 
                                  GgufValue::String("gpt2".to_string()));
                }
                
                // Fix missing add_space_prefix
                if !metadata.contains_key("tokenizer.ggml.add_space_prefix") {
                    warn!("Adding missing tokenizer.ggml.add_space_prefix = true");
                    metadata.insert("tokenizer.ggml.add_space_prefix".to_string(),
                                  GgufValue::Bool(true));
                }
                
                // Fix missing byte_fallback
                if !metadata.contains_key("tokenizer.ggml.byte_fallback") {
                    warn!("Adding missing tokenizer.ggml.byte_fallback = false");
                    metadata.insert("tokenizer.ggml.byte_fallback".to_string(),
                                  GgufValue::Bool(false));
                }
            }
            "llama" | "spm" | "sentencepiece" => {
                // Fix missing pre-tokenizer
                if !metadata.contains_key("tokenizer.ggml.pre") {
                    warn!("Adding missing tokenizer.ggml.pre = llama");
                    metadata.insert("tokenizer.ggml.pre".to_string(),
                                  GgufValue::String("llama".to_string()));
                }
            }
            _ => {
                warn!("Unknown tokenizer type: {}, applying generic fixes", tokenizer_model);
            }
        }
        
        // Fix missing special tokens if we can infer them
        if !metadata.contains_key("tokenizer.ggml.bos_token_id") {
            // Try to infer from vocab
            if let Ok(vocab) = reader.tokenizer_vocab() {
                for (id, (token, _score)) in vocab.iter().enumerate() {
                    if token == "<s>" || token == "<|startoftext|>" || token == "[CLS]" {
                        warn!("Inferred BOS token: {} -> {}", token, id);
                        metadata.insert("tokenizer.ggml.bos_token_id".to_string(),
                                      GgufValue::U32(id as u32));
                        break;
                    }
                }
            }
        }
        
        if !metadata.contains_key("tokenizer.ggml.eos_token_id") {
            // Try to infer from vocab
            if let Ok(vocab) = reader.tokenizer_vocab() {
                for (id, (token, _score)) in vocab.iter().enumerate() {
                    if token == "</s>" || token == "<|endoftext|>" || token == "[SEP]" {
                        warn!("Inferred EOS token: {} -> {}", token, id);
                        metadata.insert("tokenizer.ggml.eos_token_id".to_string(),
                                      GgufValue::U32(id as u32));
                        break;
                    }
                }
            }
        }
        
        // Write the fixed file
        let writer = GgufWriter::new(output_path)?;
        writer.write_header(&metadata)?;
        
        // Copy tensor data
        writer.copy_tensors_from(&reader)?;
        
        info!("Successfully created fixed GGUF file: {}", output_path.display());
        Ok(())
    }
    
    /// Print diagnostic information about a GGUF file
    pub fn print_diagnostics<P: AsRef<Path>>(path: P) -> Result<()> {
        let issues = Self::diagnose(path.as_ref())?;
        
        println!("=== GGUF Compatibility Report ===");
        println!("File: {}", path.as_ref().display());
        
        if issues.is_empty() {
            println!("✅ No compatibility issues detected!");
        } else {
            println!("⚠️  Found {} compatibility issues:", issues.len());
            for (i, issue) in issues.iter().enumerate() {
                println!("  {}. {}", i + 1, issue);
            }
            println!("\nSuggested action: Run with --fix flag to create a compatible version");
        }
        
        Ok(())
    }
}

// Placeholder for GgufValue - this should match your actual GGUF value type
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_diagnose_missing_metadata() {
        // This would test with actual GGUF files
        // For now, just ensure the module compiles
    }
}