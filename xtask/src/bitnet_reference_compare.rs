use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
    "reference_parity_promotion",
    "a770_semantic_quality_proven",
];

#[derive(Debug)]
struct CompareArgs {
    reference: PathBuf,
    cpu: PathBuf,
    a770: PathBuf,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct ReceiptInput {
    path: PathBuf,
    exists: bool,
    read_ok: bool,
    parse_ok: bool,
    error: Option<String>,
    value: Option<Value>,
}

#[derive(Clone, Debug)]
struct OutputSignal {
    token_ids: Option<Vec<i64>>,
    text: Option<String>,
    top_logits: Option<Vec<TopLogit>>,
}

#[derive(Clone, Debug)]
struct TopLogit {
    token_id: i64,
    logit: f64,
}

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-compare") {
        return Ok(false);
    }
    if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(true);
    }
    let opts = parse_args(args)?;
    let report = build_report(&opts);
    if let Some(output) = &opts.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(output, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("writing {}", output.display()))?;
    }
    emit_report(&report, &opts.format)?;
    Ok(true)
}

fn print_help() {
    println!(
        "cargo xtask bitnet-reference-compare --reference <json> --cpu <json> --a770 <json> [--output <json>] [--format human|json]"
    );
}

fn parse_args(args: &[String]) -> Result<CompareArgs> {
    let mut reference: Option<PathBuf> = None;
    let mut cpu: Option<PathBuf> = None;
    let mut a770: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut format = "human".to_string();
    let mut i = 2usize;
    while i < args.len() {
        let key = args[i].as_str();
        i += 1;
        let mut value = || -> Result<String> {
            let value = args.get(i).with_context(|| format!("{key} requires a value"))?.clone();
            i += 1;
            Ok(value)
        };
        match key {
            "--reference" => reference = Some(PathBuf::from(value()?)),
            "--cpu" => cpu = Some(PathBuf::from(value()?)),
            "--a770" => a770 = Some(PathBuf::from(value()?)),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-compare argument: {other}"),
        }
    }
    Ok(CompareArgs {
        reference: reference.context("--reference is required")?,
        cpu: cpu.context("--cpu is required")?,
        a770: a770.context("--a770 is required")?,
        output,
        format,
    })
}

fn build_report(args: &CompareArgs) -> Value {
    let reference = read_receipt(&args.reference);
    let cpu = read_receipt(&args.cpu);
    let a770 = read_receipt(&args.a770);
    let reference_signal = output_signal(reference.value.as_ref());
    let cpu_signal = output_signal(cpu.value.as_ref());
    let a770_signal = output_signal(a770.value.as_ref());
    let comparisons = json!({
        "reference_vs_cpu": compare_pair(&reference_signal, &cpu_signal),
        "reference_vs_a770": compare_pair(&reference_signal, &a770_signal),
        "cpu_vs_a770": compare_pair(&cpu_signal, &a770_signal),
    });

    let mut blocked_reasons = Vec::new();
    push_input_blockers(&mut blocked_reasons, "reference", &reference, &reference_signal);
    push_input_blockers(&mut blocked_reasons, "cpu", &cpu, &cpu_signal);
    push_input_blockers(&mut blocked_reasons, "a770", &a770, &a770_signal);
    if !bool_at(&comparisons, "/cpu_vs_a770/token_ids_exact").unwrap_or(false) {
        blocked_reasons.push("rust_cpu_a770_token_ids_not_proven_exact".to_string());
    }
    if !bool_at(&comparisons, "/reference_vs_cpu/token_ids_exact").unwrap_or(false) {
        blocked_reasons.push("reference_cpu_token_ids_not_proven_exact".to_string());
    }
    if !bool_at(&comparisons, "/reference_vs_a770/token_ids_exact").unwrap_or(false) {
        blocked_reasons.push("reference_a770_token_ids_not_proven_exact".to_string());
    }
    if !bool_at(&comparisons, "/cpu_vs_a770/top_logit_token_ids_exact").unwrap_or(false) {
        blocked_reasons.push("rust_cpu_a770_top_logit_ids_not_proven_exact".to_string());
    }
    if !bool_at(&comparisons, "/reference_vs_cpu/top_logit_token_ids_exact").unwrap_or(false) {
        blocked_reasons.push("reference_cpu_top_logit_ids_not_proven_exact".to_string());
    }
    if !bool_at(&comparisons, "/reference_vs_a770/top_logit_token_ids_exact").unwrap_or(false) {
        blocked_reasons.push("reference_a770_top_logit_ids_not_proven_exact".to_string());
    }
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    json!({
        "schema_version": 1,
        "diagnostic": "bitnet_reference_compare",
        "producer": "cargo xtask bitnet-reference-compare",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "inputs": {
            "reference": input_value(&reference, &reference_signal),
            "cpu": input_value(&cpu, &cpu_signal),
            "a770": input_value(&a770, &a770_signal),
        },
        "comparisons": comparisons,
        "decision": {
            "reference_compare_ready": blocked_reasons.is_empty(),
            "current_blocked_reasons": blocked_reasons,
            "next_when_ready": "inspect token/logit disagreement before changing Rust model math",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    })
}

fn read_receipt(path: &Path) -> ReceiptInput {
    let exists = path.is_file();
    if !exists {
        return ReceiptInput {
            path: path.to_path_buf(),
            exists,
            read_ok: false,
            parse_ok: false,
            error: Some("file_missing".to_string()),
            value: None,
        };
    }
    match fs::read_to_string(path) {
        Ok(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(value) => ReceiptInput {
                path: path.to_path_buf(),
                exists,
                read_ok: true,
                parse_ok: true,
                error: None,
                value: Some(value),
            },
            Err(error) => ReceiptInput {
                path: path.to_path_buf(),
                exists,
                read_ok: true,
                parse_ok: false,
                error: Some(format!("json_parse_error: {error}")),
                value: None,
            },
        },
        Err(error) => ReceiptInput {
            path: path.to_path_buf(),
            exists,
            read_ok: false,
            parse_ok: false,
            error: Some(format!("read_error: {error}")),
            value: None,
        },
    }
}

fn output_signal(value: Option<&Value>) -> OutputSignal {
    let Some(value) = value else {
        return OutputSignal { token_ids: None, text: None, top_logits: None };
    };
    OutputSignal {
        token_ids: token_ids(value),
        text: string_at(value, &["/text", "/generated_text", "/output", "/response"]),
        top_logits: top_logits(value),
    }
}

fn token_ids(value: &Value) -> Option<Vec<i64>> {
    for pointer in [
        "/tokens/ids",
        "/tokens/generated_ids",
        "/generated_tokens",
        "/output_tokens/ids",
        "/output_tokens",
    ] {
        if let Some(ids) = array_i64(value.pointer(pointer)) {
            return Some(ids);
        }
    }
    None
}

fn array_i64(value: Option<&Value>) -> Option<Vec<i64>> {
    let array = value?.as_array()?;
    let mut ids = Vec::with_capacity(array.len());
    for item in array {
        if let Some(id) = item.as_i64() {
            ids.push(id);
        } else if let Some(id) = item.as_u64().and_then(|id| i64::try_from(id).ok()) {
            ids.push(id);
        } else {
            return None;
        }
    }
    Some(ids)
}

fn string_at(value: &Value, pointers: &[&str]) -> Option<String> {
    pointers
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_str).map(ToOwned::to_owned))
}

fn top_logits(value: &Value) -> Option<Vec<TopLogit>> {
    for pointer in
        ["/logits_dump/0/top_logits", "/top_logits", "/logits/top_logits", "/logits/top_k"]
    {
        if let Some(logits) = top_logits_array(value.pointer(pointer)) {
            return Some(logits);
        }
    }
    None
}

fn top_logits_array(value: Option<&Value>) -> Option<Vec<TopLogit>> {
    let array = value?.as_array()?;
    let mut logits = Vec::with_capacity(array.len());
    for item in array {
        let token_id = item
            .pointer("/token_id")
            .and_then(Value::as_i64)
            .or_else(|| item.pointer("/id").and_then(Value::as_i64))
            .or_else(|| item.pointer("/token").and_then(Value::as_i64))?;
        let logit = item
            .pointer("/logit")
            .and_then(Value::as_f64)
            .or_else(|| item.pointer("/value").and_then(Value::as_f64))?;
        logits.push(TopLogit { token_id, logit });
    }
    Some(logits)
}

fn compare_pair(left: &OutputSignal, right: &OutputSignal) -> Value {
    let token_ids_exact =
        left.token_ids.as_ref().zip(right.token_ids.as_ref()).map(|(left, right)| left == right);
    let first_token_exact =
        left.token_ids.as_ref().zip(right.token_ids.as_ref()).and_then(|(left, right)| {
            left.first().zip(right.first()).map(|(left, right)| left == right)
        });
    let text_exact = left.text.as_ref().zip(right.text.as_ref()).map(|(left, right)| left == right);
    let top_logit_token_ids_exact =
        left.top_logits.as_ref().zip(right.top_logits.as_ref()).map(|(left, right)| {
            left.iter().map(|item| item.token_id).collect::<Vec<_>>()
                == right.iter().map(|item| item.token_id).collect::<Vec<_>>()
        });
    let top_logit_max_abs_delta =
        left.top_logits.as_ref().zip(right.top_logits.as_ref()).map(|(left, right)| {
            left.iter()
                .zip(right.iter())
                .map(|(left, right)| (left.logit - right.logit).abs())
                .fold(0.0_f64, f64::max)
        });
    json!({
        "token_ids_available": left.token_ids.is_some() && right.token_ids.is_some(),
        "token_ids_exact": token_ids_exact,
        "first_token_exact": first_token_exact,
        "left_token_count": left.token_ids.as_ref().map(Vec::len),
        "right_token_count": right.token_ids.as_ref().map(Vec::len),
        "text_available": left.text.is_some() && right.text.is_some(),
        "text_exact": text_exact,
        "top_logits_available": left.top_logits.is_some() && right.top_logits.is_some(),
        "top_logit_count": left.top_logits.as_ref().zip(right.top_logits.as_ref()).map(|(left, right)| left.len().min(right.len())),
        "top_logit_token_ids_exact": top_logit_token_ids_exact,
        "top_logit_max_abs_delta": top_logit_max_abs_delta,
    })
}

fn input_value(input: &ReceiptInput, signal: &OutputSignal) -> Value {
    json!({
        "path": input.path.display().to_string(),
        "exists": input.exists,
        "read_ok": input.read_ok,
        "parse_ok": input.parse_ok,
        "error": input.error,
        "token_ids_present": signal.token_ids.is_some(),
        "token_count": signal.token_ids.as_ref().map(Vec::len),
        "text_present": signal.text.is_some(),
        "top_logits_present": signal.top_logits.is_some(),
        "top_logit_count": signal.top_logits.as_ref().map(Vec::len),
    })
}

fn push_input_blockers(
    blocked_reasons: &mut Vec<String>,
    label: &str,
    input: &ReceiptInput,
    signal: &OutputSignal,
) {
    if !input.exists {
        blocked_reasons.push(format!("{label}_receipt_missing"));
        return;
    }
    if !input.read_ok {
        blocked_reasons.push(format!("{label}_receipt_unreadable"));
        return;
    }
    if !input.parse_ok {
        blocked_reasons.push(format!("{label}_receipt_json_invalid"));
        return;
    }
    if signal.token_ids.is_none() {
        blocked_reasons.push(format!("{label}_token_ids_missing"));
    }
    if signal.top_logits.is_none() {
        blocked_reasons.push(format!("{label}_top_logits_missing"));
    }
}

fn bool_at(value: &Value, pointer: &str) -> Option<bool> {
    value.pointer(pointer).and_then(Value::as_bool)
}

fn emit_report(value: &Value, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(value)?),
        "human" => {
            println!("diagnostic: bitnet_reference_compare");
            println!(
                "classification: {}",
                value
                    .pointer("/classification")
                    .and_then(Value::as_str)
                    .unwrap_or("diagnostic_only")
            );
            if let Some(reasons) = value.pointer("/decision/current_blocked_reasons") {
                println!("blocked_reasons: {}", serde_json::to_string(reasons)?);
            }
            println!("not_claims: {}", serde_json::to_string(&value["not_claims"])?);
        }
        other => bail!("unsupported bitnet-reference-compare output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_cli_token_ids() {
        let value = json!({
            "tokens": {
                "ids": [1, 2, 3]
            }
        });
        assert_eq!(token_ids(&value), Some(vec![1, 2, 3]));
    }

    #[test]
    fn compare_pair_reports_token_mismatch() {
        let left = OutputSignal {
            token_ids: Some(vec![1, 2]),
            text: Some("a".to_string()),
            top_logits: Some(vec![
                TopLogit { token_id: 1, logit: 2.0 },
                TopLogit { token_id: 2, logit: 1.0 },
            ]),
        };
        let right = OutputSignal {
            token_ids: Some(vec![1, 3]),
            text: Some("b".to_string()),
            top_logits: Some(vec![
                TopLogit { token_id: 1, logit: 2.25 },
                TopLogit { token_id: 3, logit: 0.75 },
            ]),
        };
        let report = compare_pair(&left, &right);
        assert_eq!(report["token_ids_available"], true);
        assert_eq!(report["token_ids_exact"], false);
        assert_eq!(report["first_token_exact"], true);
        assert_eq!(report["text_exact"], false);
        assert_eq!(report["top_logits_available"], true);
        assert_eq!(report["top_logit_token_ids_exact"], false);
        assert_eq!(report["top_logit_max_abs_delta"], 0.25);
    }

    #[test]
    fn extracts_cli_top_logits() {
        let value = json!({
            "logits_dump": [{
                "top_logits": [
                    {"token_id": 10, "logit": 1.5},
                    {"token_id": 11, "logit": 1.25}
                ]
            }]
        });
        let logits = top_logits(&value).expect("top logits");
        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0].token_id, 10);
        assert_eq!(logits[0].logit, 1.5);
    }

    #[test]
    fn missing_reference_blocks_ready_report() {
        let args = CompareArgs {
            reference: PathBuf::from("target/missing-reference.json"),
            cpu: PathBuf::from("target/missing-cpu.json"),
            a770: PathBuf::from("target/missing-a770.json"),
            output: None,
            format: "json".to_string(),
        };
        let report = build_report(&args);
        assert_eq!(report["claim_allowed"], false);
        let reasons = report["decision"]["current_blocked_reasons"].as_array().unwrap();
        assert!(reasons.iter().any(|reason| reason == "reference_receipt_missing"));
    }
}
