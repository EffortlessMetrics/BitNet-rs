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
    selected_logits: Option<Vec<SelectedLogit>>,
    reference_text_candidate_ids: Option<Vec<i64>>,
    prompt: PromptIdentity,
}

#[derive(Clone, Debug)]
struct TopLogit {
    token_id: i64,
    logit: f64,
}

#[derive(Clone, Debug)]
struct SelectedLogit {
    token_id: i64,
    present: bool,
    logit: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct PromptIdentity {
    template: Option<String>,
    rendered_prompt_sha256: Option<String>,
    prompt_token_ids_sha256: Option<String>,
    prompt_token_count: Option<u64>,
    add_bos_or_bos_policy: Option<bool>,
    parse_special: Option<bool>,
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
    if !bool_at(&comparisons, "/reference_vs_cpu/text_exact").unwrap_or(false) {
        blocked_reasons.push("reference_cpu_text_not_exact".to_string());
    }
    if !bool_at(&comparisons, "/reference_vs_a770/text_exact").unwrap_or(false) {
        blocked_reasons.push("reference_a770_text_not_exact".to_string());
    }
    if !bool_at(&comparisons, "/cpu_vs_a770/text_exact").unwrap_or(false) {
        blocked_reasons.push("rust_cpu_a770_text_not_exact".to_string());
    }
    if !bool_at(&comparisons, "/reference_vs_cpu/prompt_identity_matched").unwrap_or(false) {
        blocked_reasons.push("reference_cpu_prompt_identity_not_matched".to_string());
    }
    if !bool_at(&comparisons, "/reference_vs_a770/prompt_identity_matched").unwrap_or(false) {
        blocked_reasons.push("reference_a770_prompt_identity_not_matched".to_string());
    }
    if !bool_at(&comparisons, "/cpu_vs_a770/prompt_identity_matched").unwrap_or(false) {
        blocked_reasons.push("rust_cpu_a770_prompt_identity_not_matched".to_string());
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
        return OutputSignal {
            token_ids: None,
            text: None,
            top_logits: None,
            selected_logits: None,
            reference_text_candidate_ids: None,
            prompt: PromptIdentity::default(),
        };
    };
    OutputSignal {
        token_ids: token_ids(value),
        text: string_at(value, &["/text", "/generated_text", "/output", "/response"]),
        top_logits: top_logits(value),
        selected_logits: selected_logits(value),
        reference_text_candidate_ids: reference_text_candidate_ids(value),
        prompt: prompt_identity(value),
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

fn selected_logits(value: &Value) -> Option<Vec<SelectedLogit>> {
    for pointer in ["/logits_dump/0/selected_logits", "/selected_logits", "/logits/selected_logits"]
    {
        if let Some(logits) = selected_logits_array(value.pointer(pointer)) {
            return Some(logits);
        }
    }
    None
}

fn selected_logits_array(value: Option<&Value>) -> Option<Vec<SelectedLogit>> {
    let array = value?.as_array()?;
    let mut logits = Vec::with_capacity(array.len());
    for item in array {
        let token_id = item
            .pointer("/token_id")
            .and_then(Value::as_i64)
            .or_else(|| item.pointer("/id").and_then(Value::as_i64))
            .or_else(|| item.pointer("/token").and_then(Value::as_i64))?;
        let present = item.pointer("/present").and_then(Value::as_bool).unwrap_or(true);
        let logit = item
            .pointer("/logit")
            .and_then(Value::as_f64)
            .or_else(|| item.pointer("/value").and_then(Value::as_f64));
        logits.push(SelectedLogit { token_id, present, logit });
    }
    Some(logits)
}

fn reference_text_candidate_ids(value: &Value) -> Option<Vec<i64>> {
    for pointer in [
        "/reference_text_tokenization/selected_logit_probe_ids",
        "/rust_commands/reference_text_tokenization/selected_logit_probe_ids",
        "/plan/reference_text_tokenization/selected_logit_probe_ids",
    ] {
        if let Some(ids) = array_i64(value.pointer(pointer)) {
            return Some(ids);
        }
    }
    None
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
    let top_logit_argmax_token_exact =
        left.top_logits.as_ref().zip(right.top_logits.as_ref()).and_then(|(left, right)| {
            left.first().zip(right.first()).map(|(left, right)| left.token_id == right.token_id)
        });
    let top_logit_argmax_delta =
        left.top_logits.as_ref().zip(right.top_logits.as_ref()).and_then(|(left, right)| {
            left.first().zip(right.first()).map(|(left, right)| (left.logit - right.logit).abs())
        });
    let top_logit_max_abs_delta =
        left.top_logits.as_ref().zip(right.top_logits.as_ref()).map(|(left, right)| {
            left.iter()
                .zip(right.iter())
                .map(|(left, right)| (left.logit - right.logit).abs())
                .fold(0.0_f64, f64::max)
        });
    json!({
        "prompt_identity": compare_prompt_identity(&left.prompt, &right.prompt),
        "prompt_identity_available": left.prompt.available() && right.prompt.available(),
        "prompt_identity_matched": left.prompt.matches(&right.prompt),
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
        "top_logit_argmax_token_exact": top_logit_argmax_token_exact,
        "top_logit_argmax_delta": top_logit_argmax_delta,
        "top_logit_max_abs_delta": top_logit_max_abs_delta,
        "reference_text_candidate_logits": reference_text_candidate_logits(left, right),
    })
}

fn reference_text_candidate_logits(left: &OutputSignal, right: &OutputSignal) -> Value {
    let candidate_ids = left.reference_text_candidate_ids.as_deref();
    let selected_logits = right.selected_logits.as_deref();
    let top_logit = right.top_logits.as_ref().and_then(|logits| logits.first());
    let Some(candidate_ids) = candidate_ids else {
        return json!({
            "diagnostic_only": true,
            "candidate_ids_available": false,
            "selected_logits_available": selected_logits.is_some(),
            "matched_candidate_count": Value::Null,
            "not_claim": "text-tokenized reference candidates are not reference generated token IDs",
        });
    };
    let Some(selected_logits) = selected_logits else {
        return json!({
            "diagnostic_only": true,
            "candidate_ids_available": true,
            "selected_logits_available": false,
            "candidate_count": candidate_ids.len(),
            "matched_candidate_count": Value::Null,
            "not_claim": "text-tokenized reference candidates are not reference generated token IDs",
        });
    };

    let mut best_candidate: Option<(i64, f64)> = None;
    let mut rows = Vec::new();
    for token_id in candidate_ids {
        let selected = selected_logits.iter().find(|item| item.token_id == *token_id);
        let present = selected.is_some_and(|item| item.present && item.logit.is_some());
        let logit = selected.and_then(|item| item.logit);
        if let Some(logit) = logit
            && best_candidate.is_none_or(|(_, best)| logit > best)
        {
            best_candidate = Some((*token_id, logit));
        }
        rows.push(json!({
            "token_id": token_id,
            "present": present,
            "logit": logit,
        }));
    }

    let best_candidate_token_id = best_candidate.map(|(token_id, _)| token_id);
    let best_candidate_logit = best_candidate.map(|(_, logit)| logit);
    let top_token_id = top_logit.map(|item| item.token_id);
    let top_logit_value = top_logit.map(|item| item.logit);
    let best_candidate_to_top_delta =
        best_candidate_logit.zip(top_logit_value).map(|(candidate, top)| top - candidate);

    json!({
        "diagnostic_only": true,
        "candidate_ids_available": true,
        "selected_logits_available": true,
        "candidate_count": candidate_ids.len(),
        "matched_candidate_count": rows.iter().filter(|row| row["present"] == true).count(),
        "candidate_logits": rows,
        "best_candidate_token_id": best_candidate_token_id,
        "best_candidate_logit": best_candidate_logit,
        "right_top_token_id": top_token_id,
        "right_top_logit": top_logit_value,
        "best_candidate_to_top_delta": best_candidate_to_top_delta,
        "not_claim": "text-tokenized reference candidates are not reference generated token IDs",
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
        "selected_logits_present": signal.selected_logits.is_some(),
        "selected_logit_count": signal.selected_logits.as_ref().map(Vec::len),
        "reference_text_candidate_ids_present": signal.reference_text_candidate_ids.is_some(),
        "reference_text_candidate_id_count": signal.reference_text_candidate_ids.as_ref().map(Vec::len),
        "prompt_identity_present": signal.prompt.available(),
        "prompt_identity": prompt_identity_value(&signal.prompt),
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
    if !signal.prompt.available() {
        blocked_reasons.push(format!("{label}_prompt_identity_missing"));
    }
}

fn bool_at(value: &Value, pointer: &str) -> Option<bool> {
    value.pointer(pointer).and_then(Value::as_bool)
}

fn prompt_identity(value: &Value) -> PromptIdentity {
    let prompt =
        value.pointer("/prompt_identity").or_else(|| value.pointer("/plan/prompt_identity"));
    let Some(prompt) = prompt else {
        return PromptIdentity::default();
    };
    PromptIdentity {
        template: string_at(prompt, &["/template", "/prompt_template"]),
        rendered_prompt_sha256: string_at(prompt, &["/rendered_prompt_sha256"]),
        prompt_token_ids_sha256: string_at(prompt, &["/prompt_token_ids_sha256"]),
        prompt_token_count: prompt.pointer("/prompt_token_count").and_then(Value::as_u64),
        add_bos_or_bos_policy: prompt
            .pointer("/add_bos")
            .and_then(Value::as_bool)
            .or_else(|| prompt.pointer("/bos_policy").and_then(Value::as_bool)),
        parse_special: prompt.pointer("/parse_special").and_then(Value::as_bool),
    }
}

impl PromptIdentity {
    fn available(&self) -> bool {
        self.rendered_prompt_sha256.is_some()
            && self.prompt_token_ids_sha256.is_some()
            && self.prompt_token_count.is_some()
    }

    fn matches(&self, other: &Self) -> bool {
        self.available()
            && other.available()
            && self.template == other.template
            && self.rendered_prompt_sha256 == other.rendered_prompt_sha256
            && self.prompt_token_ids_sha256 == other.prompt_token_ids_sha256
            && self.prompt_token_count == other.prompt_token_count
            && self.add_bos_or_bos_policy == other.add_bos_or_bos_policy
            && self.parse_special == other.parse_special
    }
}

fn compare_prompt_identity(left: &PromptIdentity, right: &PromptIdentity) -> Value {
    json!({
        "template_matched": left.template == right.template,
        "rendered_prompt_sha256_matched": left.rendered_prompt_sha256 == right.rendered_prompt_sha256,
        "prompt_token_ids_sha256_matched": left.prompt_token_ids_sha256 == right.prompt_token_ids_sha256,
        "prompt_token_count_matched": left.prompt_token_count == right.prompt_token_count,
        "bos_policy_matched": left.add_bos_or_bos_policy == right.add_bos_or_bos_policy,
        "parse_special_matched": left.parse_special == right.parse_special,
        "left": prompt_identity_value(left),
        "right": prompt_identity_value(right),
    })
}

fn prompt_identity_value(prompt: &PromptIdentity) -> Value {
    json!({
        "template": prompt.template,
        "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
        "prompt_token_ids_sha256": prompt.prompt_token_ids_sha256,
        "prompt_token_count": prompt.prompt_token_count,
        "add_bos_or_bos_policy": prompt.add_bos_or_bos_policy,
        "parse_special": prompt.parse_special,
    })
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
            selected_logits: None,
            reference_text_candidate_ids: None,
            prompt: prompt("llama3-chat", "rendered-a", "ids-a", 17, false, true),
        };
        let right = OutputSignal {
            token_ids: Some(vec![1, 3]),
            text: Some("b".to_string()),
            top_logits: Some(vec![
                TopLogit { token_id: 1, logit: 2.25 },
                TopLogit { token_id: 3, logit: 0.75 },
            ]),
            selected_logits: None,
            reference_text_candidate_ids: None,
            prompt: prompt("llama3-chat", "rendered-a", "ids-a", 17, false, true),
        };
        let report = compare_pair(&left, &right);
        assert_eq!(report["prompt_identity_matched"], true);
        assert_eq!(report["token_ids_available"], true);
        assert_eq!(report["token_ids_exact"], false);
        assert_eq!(report["first_token_exact"], true);
        assert_eq!(report["text_exact"], false);
        assert_eq!(report["top_logits_available"], true);
        assert_eq!(report["top_logit_token_ids_exact"], false);
        assert_eq!(report["top_logit_argmax_token_exact"], true);
        assert_eq!(report["top_logit_argmax_delta"], 0.25);
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
    fn extracts_selected_logits_and_reference_text_candidates() {
        let value = json!({
            "rust_commands": {
                "reference_text_tokenization": {
                    "selected_logit_probe_ids": [17, 10, 17239]
                }
            },
            "logits_dump": [{
                "selected_logits": [
                    {"token_id": 17, "present": true, "logit": 3.5},
                    {"token_id": 10, "present": true, "logit": -0.5}
                ]
            }]
        });

        assert_eq!(reference_text_candidate_ids(&value), Some(vec![17, 10, 17239]));
        let logits = selected_logits(&value).expect("selected logits");
        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0].token_id, 17);
        assert_eq!(logits[0].logit, Some(3.5));
    }

    #[test]
    fn compare_pair_reports_reference_text_candidate_logit_gap() {
        let left = OutputSignal {
            token_ids: None,
            text: Some("2+2 equals 4.".to_string()),
            top_logits: None,
            selected_logits: None,
            reference_text_candidate_ids: Some(vec![17, 10]),
            prompt: prompt("llama3-chat", "rendered-a", "ids-a", 17, false, true),
        };
        let right = OutputSignal {
            token_ids: Some(vec![54864]),
            text: Some("-fixed".to_string()),
            top_logits: Some(vec![TopLogit { token_id: 54864, logit: 10.75 }]),
            selected_logits: Some(vec![
                SelectedLogit { token_id: 17, present: true, logit: Some(3.5) },
                SelectedLogit { token_id: 10, present: true, logit: Some(-0.5) },
            ]),
            reference_text_candidate_ids: None,
            prompt: prompt("llama3-chat", "rendered-a", "ids-a", 17, false, true),
        };

        let report = compare_pair(&left, &right);
        let candidate_logits = &report["reference_text_candidate_logits"];
        assert_eq!(candidate_logits["diagnostic_only"], true);
        assert_eq!(candidate_logits["candidate_ids_available"], true);
        assert_eq!(candidate_logits["selected_logits_available"], true);
        assert_eq!(candidate_logits["best_candidate_token_id"], 17);
        assert_eq!(candidate_logits["best_candidate_logit"], 3.5);
        assert_eq!(candidate_logits["right_top_token_id"], 54864);
        assert_eq!(candidate_logits["best_candidate_to_top_delta"], 7.25);
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

    #[test]
    fn reference_text_mismatch_is_explicit_blocker() {
        let dir = tempfile::tempdir().unwrap();
        let reference_path = dir.path().join("reference.json");
        let cpu_path = dir.path().join("cpu.json");
        let a770_path = dir.path().join("a770.json");
        let prompt = json!({
            "prompt_template": "llama3-chat",
            "rendered_prompt_sha256": "rendered",
            "prompt_token_ids_sha256": "ids",
            "prompt_token_count": 17,
            "add_bos": false,
            "parse_special": true
        });
        let top_logits = json!([{"token_id": 123, "logit": 4.0}]);
        std::fs::write(
            &reference_path,
            serde_json::to_vec(&json!({
                "generated_tokens": [123],
                "generated_text": "2+2 equals 4.",
                "top_logits": top_logits,
                "prompt_identity": prompt,
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            &cpu_path,
            serde_json::to_vec(&json!({
                "generated_tokens": [123],
                "text": ".ps",
                "top_logits": top_logits,
                "prompt_identity": prompt,
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            &a770_path,
            serde_json::to_vec(&json!({
                "generated_tokens": [123],
                "text": ".ps",
                "top_logits": top_logits,
                "prompt_identity": prompt,
            }))
            .unwrap(),
        )
        .unwrap();

        let report = build_report(&CompareArgs {
            reference: reference_path,
            cpu: cpu_path,
            a770: a770_path,
            output: None,
            format: "json".to_string(),
        });

        assert_eq!(report["comparisons"]["reference_vs_cpu"]["token_ids_exact"], true);
        assert_eq!(report["comparisons"]["reference_vs_cpu"]["top_logit_token_ids_exact"], true);
        assert_eq!(report["comparisons"]["reference_vs_cpu"]["text_exact"], false);
        assert_eq!(report["comparisons"]["reference_vs_a770"]["text_exact"], false);
        assert_eq!(report["comparisons"]["cpu_vs_a770"]["text_exact"], true);
        let reasons = report["decision"]["current_blocked_reasons"].as_array().unwrap();
        assert!(reasons.iter().any(|reason| reason == "reference_cpu_text_not_exact"));
        assert!(reasons.iter().any(|reason| reason == "reference_a770_text_not_exact"));
        assert!(!reasons.iter().any(|reason| reason == "rust_cpu_a770_text_not_exact"));
        assert!(!reasons.iter().any(|reason| reason == "reference_cpu_token_ids_not_proven_exact"));
        assert!(
            !reasons.iter().any(|reason| reason == "reference_cpu_top_logit_ids_not_proven_exact")
        );
    }

    #[test]
    fn prompt_identity_mismatch_blocks_comparison_ready() {
        let left = OutputSignal {
            token_ids: Some(vec![1]),
            text: Some("a".to_string()),
            top_logits: Some(vec![TopLogit { token_id: 1, logit: 1.0 }]),
            selected_logits: None,
            reference_text_candidate_ids: None,
            prompt: prompt("llama3-chat", "rendered-a", "ids-a", 17, false, true),
        };
        let right = OutputSignal {
            token_ids: Some(vec![1]),
            text: Some("a".to_string()),
            top_logits: Some(vec![TopLogit { token_id: 1, logit: 1.0 }]),
            selected_logits: None,
            reference_text_candidate_ids: None,
            prompt: prompt("raw", "rendered-b", "ids-b", 8, true, false),
        };
        let report = compare_pair(&left, &right);

        assert_eq!(report["token_ids_exact"], true);
        assert_eq!(report["top_logit_token_ids_exact"], true);
        assert_eq!(report["prompt_identity_matched"], false);
        assert_eq!(report["prompt_identity"]["prompt_token_count_matched"], false);
    }

    #[test]
    fn extracts_reference_run_plan_prompt_identity() {
        let value = json!({
            "plan": {
                "prompt_identity": {
                    "prompt_template": "llama3-chat",
                    "rendered_prompt_sha256": "rendered",
                    "prompt_token_ids_sha256": "ids",
                    "prompt_token_count": 17,
                    "add_bos": false,
                    "parse_special": true
                }
            }
        });
        let identity = prompt_identity(&value);

        assert!(identity.available());
        assert_eq!(identity.template.as_deref(), Some("llama3-chat"));
        assert_eq!(identity.prompt_token_count, Some(17));
        assert_eq!(identity.add_bos_or_bos_policy, Some(false));
        assert_eq!(identity.parse_special, Some(true));
    }

    fn prompt(
        template: &str,
        rendered_prompt_sha256: &str,
        prompt_token_ids_sha256: &str,
        prompt_token_count: u64,
        bos_policy: bool,
        parse_special: bool,
    ) -> PromptIdentity {
        PromptIdentity {
            template: Some(template.to_string()),
            rendered_prompt_sha256: Some(rendered_prompt_sha256.to_string()),
            prompt_token_ids_sha256: Some(prompt_token_ids_sha256.to_string()),
            prompt_token_count: Some(prompt_token_count),
            add_bos_or_bos_policy: Some(bos_policy),
            parse_special: Some(parse_special),
        }
    }
}
