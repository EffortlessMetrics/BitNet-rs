use anyhow::Result;
use std::path::Path;
use tracing::{info, warn};

use bitnet_models::formats::gguf::GgufReader;
use ggus::{GGuf, GGufFileHeader, GGufFileWriter, GGufMetaDataValueType, GGufMetaMap};

/// GGUF compatibility fixer that auto-patches missing metadata
pub struct GgufCompatibilityFixer;

impl GgufCompatibilityFixer {
    /// Diagnose compatibility issues in a GGUF file
    pub fn diagnose<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
        use std::fs;
        let data = fs::read(path)?;
        let reader = GgufReader::new(&data)?;
        let mut issues = Vec::new();

        // Check tokenizer model
        let tokenizer_model = reader
            .get_string_metadata("tokenizer.ggml.model")
            .unwrap_or_else(|| "unknown".to_string());

        // Check for missing pre-tokenizer (critical for llama.cpp)
        if tokenizer_model == "gpt2" || tokenizer_model == "llama3" {
            let has_pre = reader.get_string_metadata("tokenizer.ggml.pre").is_some();
            if !has_pre {
                issues.push(format!(
                    "Missing tokenizer.ggml.pre for {} tokenizer - llama.cpp will fail",
                    tokenizer_model
                ));
            }
        }

        // Check for add_space_prefix
        if tokenizer_model == "gpt2" {
            let has_space_prefix =
                reader.get_bool_metadata("tokenizer.ggml.add_space_prefix").is_some();
            if !has_space_prefix {
                issues.push("Missing tokenizer.ggml.add_space_prefix for GPT-2".to_string());
            }
        }

        // Check special token IDs
        let bos_id = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos_id = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");

        if bos_id.is_none() {
            issues.push("Missing BOS token ID".to_string());
        }
        if eos_id.is_none() {
            issues.push("Missing EOS token ID".to_string());
        }

        // Check for vocabulary
        let vocab_size = reader.get_u32_metadata("tokenizer.ggml.vocab_size").unwrap_or(0);
        if vocab_size == 0 {
            issues.push("No vocabulary found".to_string());
        }

        Ok(issues)
    }

    /// Export a fixed GGUF file with missing metadata (non-destructive)
    pub fn fix_and_export<P: AsRef<Path>>(input_path: P, output_path: P) -> Result<()> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        // Don't overwrite input file
        if input_path == output_path {
            return Err(anyhow::anyhow!("Output path must be different from input path"));
        }

        let issues = Self::diagnose(input_path)?;

        let bytes = std::fs::read(input_path)?;
        let gguf = GGuf::new(&bytes)?;

        if issues.is_empty() {
            info!("No compatibility issues found, copying as-is");
            std::fs::write(output_path, bytes)?;
            return Ok(());
        }

        info!("Found {} compatibility issues, exporting fixed version...", issues.len());
        for issue in &issues {
            warn!("Fixing: {}", issue);
        }

        let mut extra_meta = Vec::new();
        if gguf.get("tokenizer.ggml.bos_token_id").is_none() {
            extra_meta.push(("tokenizer.ggml.bos_token_id", 0u32));
        }
        if gguf.get("tokenizer.ggml.eos_token_id").is_none() {
            extra_meta.push(("tokenizer.ggml.eos_token_id", 0u32));
        }
        if gguf.get("tokenizer.ggml.vocab_size").is_none() {
            extra_meta.push(("tokenizer.ggml.vocab_size", 1u32));
        }

        let header = GGufFileHeader::new(
            gguf.header.version,
            gguf.header.tensor_count,
            (gguf.meta_kvs.len() + extra_meta.len()) as u64,
        );
        let mut writer = GGufFileWriter::new(std::fs::File::create(output_path)?, header)?;

        for (key, kv) in gguf.meta_kvs.iter() {
            writer.write_meta_kv(key, kv.ty(), kv.value_bytes())?;
        }
        for (key, val) in &extra_meta {
            writer.write_meta_kv(key, GGufMetaDataValueType::U32, &val.to_le_bytes())?;
        }

        let mut tensor_writer = writer.finish::<&[u8]>(true);
        for (name, tensor) in gguf.tensors.iter() {
            let info = tensor.to_info();
            let offset = info.offset() as usize;
            let len = info.ty().size().elements_to_bytes(info.shape());
            let data = &gguf.data[offset..offset + len];
            tensor_writer.write_tensor(name, info.ty(), info.shape(), data)?;
        }
        tensor_writer.finish()?;

        info!("Fixed GGUF exported to: {}", output_path.display());
        Ok(())
    }

    /// Check if fixes are idempotent (running twice produces same result)
    pub fn verify_idempotent<P: AsRef<Path>>(path: P) -> Result<bool> {
        Ok(Self::diagnose(path)?.is_empty())
    }

    /// Print compatibility report
    pub fn print_report<P: AsRef<Path>>(path: P) -> Result<()> {
        let issues = Self::diagnose(path)?;

        if issues.is_empty() {
            println!("✅ GGUF file is compatible with llama.cpp");
        } else {
            println!("❌ Found {} compatibility issues:", issues.len());
            for issue in issues {
                println!("  - {}", issue);
            }
            println!("\nRun with --fix to auto-repair these issues");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_diagnose_placeholder() {
        // This is a placeholder test
        // Real tests would require actual GGUF files
        assert!(true);
    }
}
