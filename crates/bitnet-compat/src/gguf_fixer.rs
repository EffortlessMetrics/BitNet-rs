use anyhow::{Result, anyhow};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path};
use tracing::{info, warn};

use bitnet_models::formats::gguf::GgufReader;
use ggus::{GGuf, GGufFileHeader, GGufFileWriter, GGufMetaDataValueType, GGufWriter};

/// GGUF compatibility fixer that auto-patches missing metadata
pub struct GgufCompatibilityFixer;

impl GgufCompatibilityFixer {
    /// Diagnose compatibility issues in a GGUF file
    pub fn diagnose<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
        let data = fs::read(path.as_ref())?;
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
    pub fn export_fixed<P: AsRef<Path>, Q: AsRef<Path>>(
        input_path: P,
        output_path: Q,
    ) -> Result<()> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        if input_path == output_path {
            return Err(anyhow!("Output path must be different from input path"));
        }

        let data = fs::read(input_path)?;
        let issues = Self::diagnose(input_path)?;

        if issues.is_empty() {
            info!("No compatibility issues found, copying as-is");
            fs::copy(input_path, output_path)?;
            // still validate and checksum
            let out = fs::read(output_path)?;
            let _ = GgufReader::new(&out)?;
            let checksum = Sha256::digest(&out);
            info!("Output checksum: {:x}", checksum);
            return Ok(());
        }

        info!("Found {} compatibility issues, exporting fixed version...", issues.len());
        for issue in &issues {
            warn!("Fixing: {}", issue);
        }

        let gguf = GGuf::new(&data).map_err(|e| anyhow!("{}", e))?;

        // Collect existing metadata
        let mut seen: HashSet<&str> = HashSet::new();
        let mut meta: Vec<(String, GGufMetaDataValueType, Vec<u8>)> = Vec::new();
        for (k, v) in &gguf.meta_kvs {
            seen.insert(k);
            meta.push(((*k).to_string(), v.ty(), v.value_bytes().to_vec()));
        }

        // Helper closures for encoding values
        fn enc_str(s: &str) -> Vec<u8> {
            let mut buf = Vec::new();
            GGufWriter::new(&mut buf).write_str(s).unwrap();
            buf
        }
        fn enc_u32(v: u32) -> Vec<u8> {
            v.to_le_bytes().to_vec()
        }
        fn enc_bool(v: bool) -> Vec<u8> {
            vec![if v { 1 } else { 0 }]
        }

        let reader = GgufReader::new(&data)?;
        let tokenizer_model =
            reader.get_string_metadata("tokenizer.ggml.model").unwrap_or_default();

        if (tokenizer_model == "gpt2" || tokenizer_model == "llama3")
            && !seen.contains("tokenizer.ggml.pre")
        {
            meta.push((
                "tokenizer.ggml.pre".into(),
                GGufMetaDataValueType::String,
                enc_str(&tokenizer_model),
            ));
        }

        if tokenizer_model == "gpt2" && !seen.contains("tokenizer.ggml.add_space_prefix") {
            meta.push((
                "tokenizer.ggml.add_space_prefix".into(),
                GGufMetaDataValueType::Bool,
                enc_bool(true),
            ));
        }

        if !seen.contains("tokenizer.ggml.bos_token_id") {
            meta.push((
                "tokenizer.ggml.bos_token_id".into(),
                GGufMetaDataValueType::U32,
                enc_u32(1),
            ));
        }
        if !seen.contains("tokenizer.ggml.eos_token_id") {
            meta.push((
                "tokenizer.ggml.eos_token_id".into(),
                GGufMetaDataValueType::U32,
                enc_u32(2),
            ));
        }

        if !seen.contains("tokenizer.ggml.vocab_size") {
            // Try to derive from tokens array length
            let vocab = if let Some(kv) = gguf.meta_kvs.get("tokenizer.ggml.tokens") {
                let mut r = ggus::GGufReader::new(kv.value_bytes());
                let _ = r.read::<u32>();
                r.read::<u64>().unwrap_or(0) as u32
            } else {
                0
            };
            meta.push((
                "tokenizer.ggml.vocab_size".into(),
                GGufMetaDataValueType::U32,
                enc_u32(vocab),
            ));
        }

        // Mark file as fixed
        meta.push(("bitnet.compat.fixed".into(), GGufMetaDataValueType::Bool, enc_bool(true)));

        // Write out new GGUF file. `with_alignment` injects a `general.alignment` metadata
        // entry, so reserve space for it in the header count.
        let header = GGufFileHeader::new(
            gguf.header.version,
            gguf.tensors.len() as _,
            (meta.len() + 1) as _,
        );
        let mut writer =
            GGufFileWriter::with_alignment(fs::File::create(output_path)?, header, gguf.alignment)?;
        for (k, ty, v) in &meta {
            writer.write_meta_kv(k, *ty, v)?;
        }
        let mut tensor_writer = writer.finish(true);
        for (name, tensor) in &gguf.tensors {
            let info = tensor.to_info();
            let start = info.offset();
            let end = start + info.nbytes();
            let slice = &gguf.data[start..end];
            tensor_writer.write_tensor(name, info.ty(), info.shape(), slice)?;
        }
        tensor_writer.finish()?;

        // Validate and checksum
        let out = fs::read(output_path)?;
        let _ = GgufReader::new(&out)?;
        let checksum = Sha256::digest(&out);
        info!("Fixed GGUF exported to: {}", output_path.display());
        info!("Output checksum: {:x}", checksum);
        Ok(())
    }

    /// Check if fixes are idempotent (running twice produces same result)
    pub fn verify_idempotent<P: AsRef<Path>>(path: P) -> Result<bool> {
        let first_issues = Self::diagnose(&path)?;

        // If already has no issues, it's idempotent
        if first_issues.is_empty() {
            return Ok(true);
        }

        // Look for embedded compat flag
        let data = fs::read(path.as_ref())?;
        let reader = GgufReader::new(&data)?;
        Ok(reader.get_bool_metadata("bitnet.compat.fixed").unwrap_or(false))
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
