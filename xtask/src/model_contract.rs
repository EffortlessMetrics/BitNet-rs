use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "all_bitnet_models_supported",
    "all_supported_models_a770_accelerated",
    "llama_a770_supported",
    "qwen_a770_supported",
    "gemma_a770_supported",
    "gemma4_a770_supported",
    "slm_a770_supported",
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
];

const REQUIRED_POINTERS: &[&str] = &[
    "/model_id",
    "/family",
    "/source",
    "/format",
    "/quantization",
    "/architecture",
    "/local_path",
    "/sha256",
    "/tokenizer/path",
    "/tokenizer/sha256",
    "/chat_template/name",
    "/stop_tokens/eos_token_id",
    "/max_context",
];

const REQUIRED_A770_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
];

#[derive(Debug, Serialize)]
struct AssetReport {
    label: &'static str,
    asset_path: String,
    resolved_path: String,
    expected_sha256: String,
    actual_sha256: Option<String>,
    passed: bool,
    missing: bool,
}

#[derive(Debug, Serialize)]
struct ContractReport {
    path: String,
    model_id: Option<String>,
    passed: bool,
    asset_hashes_verified: bool,
    missing: Vec<String>,
    asset_hashes: Vec<AssetReport>,
}

#[derive(Debug, Serialize)]
struct LintReport {
    diagnostic: &'static str,
    producer: &'static str,
    contract_dir: String,
    contract_count: usize,
    passed: bool,
    missing: Vec<String>,
    contracts: Vec<ContractReport>,
    not_claims: &'static [&'static str],
}

pub fn lint_contracts(contract_dir: &Path, format: &str, allow_missing_assets: bool) -> Result<()> {
    let report = build_lint_report(contract_dir, allow_missing_assets)?;
    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "human" => print_human_report(&report),
        other => bail!("unsupported model-contract lint format: {other}"),
    }

    if !report.passed {
        bail!("model contract lint failed: {}", report.missing.join(", "));
    }

    Ok(())
}

fn build_lint_report(contract_dir: &Path, allow_missing_assets: bool) -> Result<LintReport> {
    let mut paths = Vec::new();
    if contract_dir.exists() {
        for entry in fs::read_dir(contract_dir)
            .with_context(|| format!("reading {}", contract_dir.display()))?
        {
            let path = entry?.path();
            if matches!(path.extension().and_then(|ext| ext.to_str()), Some("yaml" | "yml")) {
                paths.push(path);
            }
        }
    }
    paths.sort();

    let mut missing = Vec::new();
    if paths.is_empty() {
        missing.push(format!("no model contracts found under {}", contract_dir.display()));
    }

    let mut contracts = Vec::new();
    for path in paths {
        contracts.push(lint_one_contract(&path, allow_missing_assets)?);
    }

    for contract in &contracts {
        if !contract.passed {
            missing.push(format!("{} failed", contract.path));
        }
    }

    Ok(LintReport {
        diagnostic: "model_contract_lint",
        producer: "cargo xtask model-contract lint",
        contract_dir: contract_dir.display().to_string(),
        contract_count: contracts.len(),
        passed: missing.is_empty(),
        missing,
        contracts,
        not_claims: CRITICAL_NOT_CLAIMS,
    })
}

fn lint_one_contract(path: &Path, allow_missing_assets: bool) -> Result<ContractReport> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let value: Value =
        serde_yaml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;

    let mut missing = Vec::new();
    for pointer in REQUIRED_POINTERS {
        if value.pointer(pointer).map_or(true, Value::is_null) {
            missing.push((*pointer).to_string());
        }
    }

    for not_claim in REQUIRED_A770_NOT_CLAIMS {
        if !array_contains_str(&value, "/a770/not_claims", not_claim) {
            missing.push(format!("/a770/not_claims missing {not_claim}"));
        }
    }

    let root = Path::new(".");
    let mut asset_hashes = Vec::new();
    if let (Some(model_path), Some(model_sha)) =
        (str_at(&value, "/local_path"), str_at(&value, "/sha256"))
    {
        asset_hashes.push(verify_asset(
            "model weights",
            root,
            model_path,
            model_sha,
            allow_missing_assets,
        )?);
    }
    if let (Some(tokenizer_path), Some(tokenizer_sha)) =
        (str_at(&value, "/tokenizer/path"), str_at(&value, "/tokenizer/sha256"))
    {
        asset_hashes.push(verify_asset(
            "tokenizer",
            root,
            tokenizer_path,
            tokenizer_sha,
            allow_missing_assets,
        )?);
    }

    for asset in &asset_hashes {
        if !asset.passed {
            missing.push(format!("{} hash", asset.label));
        }
    }

    let asset_hashes_verified =
        !asset_hashes.is_empty() && asset_hashes.iter().all(|a| !a.missing && a.passed);
    let passed = missing.is_empty() && asset_hashes.iter().all(|a| a.passed);

    Ok(ContractReport {
        path: path.display().to_string(),
        model_id: str_at(&value, "/model_id").map(ToOwned::to_owned),
        passed,
        asset_hashes_verified,
        missing,
        asset_hashes,
    })
}

fn verify_asset(
    label: &'static str,
    root: &Path,
    asset_path: &str,
    expected_sha256: &str,
    allow_missing_assets: bool,
) -> Result<AssetReport> {
    let resolved = root.join(PathBuf::from(asset_path));
    if !resolved.exists() {
        return Ok(AssetReport {
            label,
            asset_path: asset_path.to_string(),
            resolved_path: resolved.display().to_string(),
            expected_sha256: normalize_sha(expected_sha256),
            actual_sha256: None,
            passed: allow_missing_assets,
            missing: true,
        });
    }

    let actual = sha256_file(&resolved)?;
    let expected = normalize_sha(expected_sha256);
    Ok(AssetReport {
        label,
        asset_path: asset_path.to_string(),
        resolved_path: resolved.display().to_string(),
        expected_sha256: expected.clone(),
        actual_sha256: Some(actual.clone()),
        passed: actual == expected,
        missing: false,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

fn array_contains_str(value: &Value, pointer: &str, needle: &str) -> bool {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .is_some_and(|items| items.iter().any(|item| item.as_str() == Some(needle)))
}

fn normalize_sha(value: &str) -> String {
    value.trim().trim_start_matches("sha256:").to_ascii_lowercase()
}

fn print_human_report(report: &LintReport) {
    println!("model contract lint: passed={}", report.passed);
    println!("contract dir: {}", report.contract_dir);
    for contract in &report.contracts {
        println!(
            "- {}: passed={} asset_hashes_verified={}",
            contract.path, contract.passed, contract.asset_hashes_verified
        );
        if let Some(model_id) = &contract.model_id {
            println!("  model_id: {model_id}");
        }
        for asset in &contract.asset_hashes {
            println!(
                "  {}: passed={} missing={} path={}",
                asset.label, asset.passed, asset.missing, asset.asset_path
            );
        }
        if !contract.missing.is_empty() {
            println!("  missing: {}", contract.missing.join(", "));
        }
    }
    if !report.missing.is_empty() {
        println!("missing: {}", report.missing.join(", "));
    }
    println!("not_claims: {}", report.not_claims.join(", "));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_sha_accepts_prefixed_or_plain_values() -> Result<()> {
        assert_eq!(normalize_sha("sha256:ABCD"), "abcd");
        assert_eq!(normalize_sha(" abcd "), "abcd");
        Ok(())
    }

    #[test]
    fn missing_assets_are_not_verified_by_default() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let contract = dir.path().join("model.yaml");
        fs::write(
            &contract,
            r#"
model_id: example/model
family: bitnet
source: test
format: gguf
quantization: i2_s
architecture: bitnet_b1_58
local_path: missing.gguf
sha256: 00
tokenizer:
  path: missing-tokenizer.json
  sha256: 11
a770:
  not_claims:
    - selected_attention_residency
    - resident_kv_decode
    - attention_scores_residency
    - softmax_residency
    - attention_value_mix_residency
    - full_support_op_residency
    - full_device_residency
    - completion
chat_template:
  name: llama3-chat
stop_tokens:
  eos_token_id: 128001
max_context: 4096
"#,
        )?;

        let report = build_lint_report(dir.path(), false)?;
        assert!(!report.passed);
        assert_eq!(report.contract_count, 1);
        assert!(!report.contracts[0].asset_hashes_verified);
        Ok(())
    }

    #[test]
    fn missing_assets_can_be_allowed_without_counting_as_verified() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let contract = dir.path().join("model.yaml");
        fs::write(
            &contract,
            r#"
model_id: example/model
family: bitnet
source: test
format: gguf
quantization: i2_s
architecture: bitnet_b1_58
local_path: missing.gguf
sha256: 00
tokenizer:
  path: missing-tokenizer.json
  sha256: 11
a770:
  not_claims:
    - selected_attention_residency
    - resident_kv_decode
    - attention_scores_residency
    - softmax_residency
    - attention_value_mix_residency
    - full_support_op_residency
    - full_device_residency
    - completion
chat_template:
  name: llama3-chat
stop_tokens:
  eos_token_id: 128001
max_context: 4096
"#,
        )?;

        let report = build_lint_report(dir.path(), true)?;
        assert!(report.passed);
        assert_eq!(report.contract_count, 1);
        assert!(!report.contracts[0].asset_hashes_verified);
        assert!(report.contracts[0].asset_hashes.iter().all(|asset| asset.missing));
        Ok(())
    }

    #[test]
    fn verifies_local_asset_hashes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let model = dir.path().join("model.gguf");
        let tokenizer = dir.path().join("tokenizer.json");
        fs::write(&model, b"model")?;
        fs::write(&tokenizer, b"tokenizer")?;
        let model_sha = sha256_file(&model)?;
        let tokenizer_sha = sha256_file(&tokenizer)?;
        let contract = dir.path().join("model.yaml");
        fs::write(
            &contract,
            format!(
                r#"
model_id: example/model
family: bitnet
source: test
format: gguf
quantization: i2_s
architecture: bitnet_b1_58
local_path: {}
sha256: {}
tokenizer:
  path: {}
  sha256: {}
a770:
  not_claims:
    - selected_attention_residency
    - resident_kv_decode
    - attention_scores_residency
    - softmax_residency
    - attention_value_mix_residency
    - full_support_op_residency
    - full_device_residency
    - completion
chat_template:
  name: llama3-chat
stop_tokens:
  eos_token_id: 128001
max_context: 4096
"#,
                model.display(),
                model_sha,
                tokenizer.display(),
                tokenizer_sha
            ),
        )?;

        let report = build_lint_report(dir.path(), false)?;
        assert!(report.passed);
        assert_eq!(report.contracts[0].asset_hashes.len(), 2);
        assert!(report.contracts[0].asset_hashes_verified);
        Ok(())
    }

    #[test]
    fn missing_required_a770_not_claims_fail_lint() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let model = dir.path().join("model.gguf");
        let tokenizer = dir.path().join("tokenizer.json");
        fs::write(&model, b"model")?;
        fs::write(&tokenizer, b"tokenizer")?;
        let model_sha = sha256_file(&model)?;
        let tokenizer_sha = sha256_file(&tokenizer)?;
        let contract = dir.path().join("model.yaml");
        fs::write(
            &contract,
            format!(
                r#"
model_id: example/model
family: bitnet
source: test
format: gguf
quantization: i2_s
architecture: bitnet_b1_58
local_path: {}
sha256: {}
tokenizer:
  path: {}
  sha256: {}
a770:
  not_claims:
    - selected_attention_residency
chat_template:
  name: llama3-chat
stop_tokens:
  eos_token_id: 128001
max_context: 4096
"#,
                model.display(),
                model_sha,
                tokenizer.display(),
                tokenizer_sha
            ),
        )?;

        let report = build_lint_report(dir.path(), false)?;
        assert!(!report.passed);
        assert!(report.contracts[0].missing.iter().any(|entry| entry.contains("completion")));
        Ok(())
    }

    #[test]
    fn json_shape_contains_policy_not_claims() -> Result<()> {
        let report = LintReport {
            diagnostic: "model_contract_lint",
            producer: "cargo xtask model-contract lint",
            contract_dir: "docs/model-contracts".to_string(),
            contract_count: 0,
            passed: false,
            missing: vec!["no model contracts".to_string()],
            contracts: vec![],
            not_claims: CRITICAL_NOT_CLAIMS,
        };
        let value = serde_json::json!(report);
        assert_eq!(value["diagnostic"], "model_contract_lint");
        let not_claims =
            value["not_claims"].as_array().context("not_claims should serialize as an array")?;
        assert!(not_claims.iter().any(|v| v == "selected_attention_residency"));
        assert!(not_claims.iter().any(|v| v == "completion"));
        Ok(())
    }
}
