use anyhow::{Result, anyhow};
use serde::Serialize;
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
        if reader.get_u32_metadata("tokenizer.ggml.vocab_size").is_none() {
            issues.push("No vocabulary found".to_string());
        }

        // BitNet-specific checks for GPT-2 tokenizer
        if tokenizer_model == "gpt2" {
            // Check tokens array length
            if let Some(tokens) = reader.get_string_array_metadata("tokenizer.ggml.tokens") {
                let vocab_len = tokens.len();
                if vocab_len < 50_000 {
                    issues.push(format!(
                        "Vocabulary too small: {} tokens (expected >= 50,000 for GPT-2 family)",
                        vocab_len
                    ));
                }
            } else {
                issues.push(
                    "Missing tokenizer.ggml.tokens array - cannot verify vocabulary".to_string(),
                );
            }

            // Check merges array exists (try both keys)
            let has_merges = reader.get_string_array_metadata("tokenizer.ggml.merges").is_some()
                || reader.get_string_array_metadata("tokenizer.ggml.bpe_merges").is_some();

            if !has_merges {
                issues.push(
                    "Missing BPE merges (tokenizer.ggml.merges or tokenizer.ggml.bpe_merges)"
                        .to_string(),
                );
            } else {
                // Log merges count for diagnostics
                let merges_count = reader
                    .get_string_array_metadata("tokenizer.ggml.merges")
                    .or_else(|| reader.get_string_array_metadata("tokenizer.ggml.bpe_merges"))
                    .map(|m| m.len())
                    .unwrap_or(0);

                if merges_count == 0 {
                    issues.push("BPE merges array is empty".to_string());
                } else {
                    info!("GPT-2 tokenizer has {} BPE merges", merges_count);
                }
            }
        }

        Ok(issues)
    }

    /// Probe tokenization with a known phrase to verify first-token correctness
    ///
    /// For GPT-2 family tokenizers, "What is 2+2?" should produce first token ID 3923 ("ĠWhat")
    /// This catches common issues with piece→ID mapping and space handling.
    #[cfg(feature = "tokenizer-probe")]
    pub fn probe_tokenizer<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
        use bitnet_tokenizers::RustTokenizer;

        let data = fs::read(path.as_ref())?;
        let reader = GgufReader::new(&data)?;
        let mut warnings = Vec::new();

        let tokenizer_model = reader
            .get_string_metadata("tokenizer.ggml.model")
            .unwrap_or_else(|| "unknown".to_string());

        if tokenizer_model != "gpt2" {
            // Skip probe for non-GPT-2 tokenizers
            return Ok(warnings);
        }

        // Try to encode using bitnet-tokenizers
        match RustTokenizer::from_gguf(&reader) {
            Ok(tokenizer) => {
                // encode(text, add_bos, parse_special)
                match tokenizer.encode("What is 2+2?", false, false) {
                    Ok(ids) => {
                        if let Some(&first_id) = ids.first() {
                            const EXPECTED_FIRST_ID: u32 = 3923; // "ĠWhat"
                            if first_id != EXPECTED_FIRST_ID {
                                warnings.push(format!(
                                    "Tokenizer probe failed: first token ID is {} but expected {} for 'What'",
                                    first_id, EXPECTED_FIRST_ID
                                ));
                                warnings.push(
                                    "This suggests incorrect piece→ID mapping or space handling"
                                        .to_string(),
                                );
                            } else {
                                info!(
                                    "✅ Tokenizer probe passed: first token ID correct ({})",
                                    first_id
                                );
                            }
                        }
                    }
                    Err(e) => {
                        warnings.push(format!("Tokenizer encode failed: {}", e));
                    }
                }
            }
            Err(e) => {
                warnings.push(format!("Could not load tokenizer for probe: {}", e));
            }
        }

        Ok(warnings)
    }

    /// Export a fixed GGUF file with missing metadata (non-destructive)
    pub fn export_fixed<P: AsRef<Path>>(input_path: P, output_path: P) -> Result<()> {
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
            Self::write_stamp(output_path, &issues)?;
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
        meta.push((
            "bitnet.compat.timestamp".into(),
            GGufMetaDataValueType::String,
            enc_str(&chrono::Utc::now().to_rfc3339()),
        ));

        // Write out new GGUF file
        let header =
            GGufFileHeader::new(gguf.header.version, gguf.tensors.len() as _, meta.len() as _);
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
        Self::write_stamp(output_path, &issues)?;
        Ok(())
    }

    fn write_stamp(output_path: &Path, issues: &[String]) -> Result<()> {
        #[derive(Serialize)]
        struct Stamp<'a> {
            timestamp: String,
            version: &'static str,
            fixes_applied: &'a [String],
        }

        let stamp = Stamp {
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION"),
            fixes_applied: issues,
        };

        let stamp_path = output_path.with_extension("gguf.compat.json");
        let json = serde_json::to_string_pretty(&stamp)?;
        fs::write(stamp_path, json)?;
        Ok(())
    }

    /// Check if fixes are idempotent (running twice produces same result)
    pub fn verify_idempotent<P: AsRef<Path>>(path: P) -> Result<bool> {
        let first_issues = Self::diagnose(&path)?;

        if !first_issues.is_empty() {
            return Ok(false);
        }

        // Look for embedded compat flag
        let data = fs::read(path.as_ref())?;
        let reader = GgufReader::new(&data)?;
        Ok(reader.get_bool_metadata("bitnet.compat.fixed").unwrap_or(false))
    }

    /// Print compatibility report with optional tokenizer probe
    pub fn print_report<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
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

        // Optional probe validation (requires tokenizer-probe feature)
        #[cfg(feature = "tokenizer-probe")]
        {
            let probe_warnings = Self::probe_tokenizer(path)?;
            if !probe_warnings.is_empty() {
                println!("\n⚠️  Tokenizer probe warnings:");
                for warning in probe_warnings {
                    println!("  - {}", warning);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::GgufCompatibilityFixer;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn diagnose_detects_missing_metadata() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("mini.gguf");

        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        fs::write(&path, &data).unwrap();

        let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
        assert!(issues.iter().any(|i| i.contains("BOS")));
    }
}
