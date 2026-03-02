use anyhow::Result;
use bitnet_models::formats::gguf::GgufReader;
use tracing::info;

/// Diagnose compatibility issues in GGUF data.
pub fn diagnose_data(data: &[u8]) -> Result<Vec<String>> {
    let reader = GgufReader::new(data)?;
    Ok(diagnose_reader(&reader))
}

/// Diagnose compatibility issues from a parsed GGUF reader.
#[must_use]
pub fn diagnose_reader(reader: &GgufReader<'_>) -> Vec<String> {
    let mut issues = Vec::new();

    let tokenizer_model =
        reader.get_string_metadata("tokenizer.ggml.model").unwrap_or_else(|| "unknown".to_string());

    if tokenizer_model == "gpt2" || tokenizer_model == "llama3" {
        let has_pre = reader.get_string_metadata("tokenizer.ggml.pre").is_some();
        if !has_pre {
            issues.push(format!(
                "Missing tokenizer.ggml.pre for {} tokenizer - llama.cpp will fail",
                tokenizer_model
            ));
        }
    }

    if tokenizer_model == "gpt2" {
        let has_space_prefix =
            reader.get_bool_metadata("tokenizer.ggml.add_space_prefix").is_some();
        if !has_space_prefix {
            issues.push("Missing tokenizer.ggml.add_space_prefix for GPT-2".to_string());
        }
    }

    let bos_id = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
    let eos_id = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");

    if bos_id.is_none() {
        issues.push("Missing BOS token ID".to_string());
    }
    if eos_id.is_none() {
        issues.push("Missing EOS token ID".to_string());
    }

    if reader.get_u32_metadata("tokenizer.ggml.vocab_size").is_none() {
        issues.push("No vocabulary found".to_string());
    }

    if tokenizer_model == "gpt2" {
        if let Some(tokens) = reader.get_string_array_metadata("tokenizer.ggml.tokens") {
            let vocab_len = tokens.len();
            if vocab_len < 50_000 {
                issues.push(format!(
                    "Vocabulary too small: {} tokens (expected >= 50,000 for GPT-2 family)",
                    vocab_len
                ));
            }
        } else {
            issues
                .push("Missing tokenizer.ggml.tokens array - cannot verify vocabulary".to_string());
        }

        let has_merges = reader.get_string_array_metadata("tokenizer.ggml.merges").is_some()
            || reader.get_string_array_metadata("tokenizer.ggml.bpe_merges").is_some();

        if !has_merges {
            issues.push(
                "Missing BPE merges (tokenizer.ggml.merges or tokenizer.ggml.bpe_merges)"
                    .to_string(),
            );
        } else {
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

    issues
}

#[cfg(test)]
mod tests {
    use super::diagnose_data;

    #[test]
    fn diagnose_data_detects_missing_token_ids() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let issues = diagnose_data(&data).expect("minimal gguf parses");
        assert!(issues.iter().any(|issue| issue.contains("BOS")));
        assert!(issues.iter().any(|issue| issue.contains("EOS")));
    }
}
