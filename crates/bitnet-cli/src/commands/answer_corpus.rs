//! Answer corpus runner for CPU-first and Apple M4 local-answer baselines.

use anyhow::{Context, Result};
use clap::Args;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Run the fixed answer corpus through the existing `bitnet run` surface.
#[derive(Args, Debug)]
pub struct AnswerCorpusCommand {
    /// Path to the answer corpus YAML.
    #[arg(long, default_value = "ci/quality/bitnet-answer-corpus.yaml", value_name = "PATH")]
    pub corpus: PathBuf,

    /// Official BitNet GGUF model path.
    #[arg(long, value_name = "PATH")]
    pub model: PathBuf,

    /// Explicit tokenizer path. If omitted, the run path must resolve one strictly.
    #[arg(long, value_name = "PATH")]
    pub tokenizer: Option<PathBuf>,

    /// Backend label for this baseline.
    #[arg(long, value_name = "BACKEND")]
    pub device: Option<String>,

    /// Output aggregate answer-corpus receipt.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "target/bitnet/receipts/cpu-answer-corpus.json"
    )]
    pub json_out: PathBuf,

    /// Do not invoke model generation; validate corpus shape and emit not_run rows.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,

    /// Per-prompt timeout for child `bitnet run` invocations.
    #[arg(long, value_name = "SECONDS")]
    pub per_prompt_timeout_seconds: Option<u64>,

    /// Fail the command if any executed prompt fails its quality gate.
    #[arg(long, default_value_t = false)]
    pub fail_on_quality: bool,
}

impl AnswerCorpusCommand {
    /// Execute the answer corpus runner.
    pub async fn execute(&self, default_device: &str) -> Result<()> {
        let corpus = AnswerCorpus::load(&self.corpus)?;
        let device =
            normalize_answer_corpus_device(self.device.as_deref().unwrap_or(default_device));
        if !matches!(device.as_str(), "cpu" | "apple-m4-cpu-neon") {
            anyhow::bail!(
                "answer-corpus only accepts --device cpu or --device apple-m4-cpu-neon; got {device}"
            );
        }
        let artifact_kind = answer_corpus_artifact_kind(&device);
        let default_timeout_seconds = effective_default_timeout_seconds(
            self.per_prompt_timeout_seconds,
            corpus.defaults.per_prompt_timeout_seconds,
        );

        let receipt_dir = self
            .json_out
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
            .join("answer-corpus-runs");
        fs::create_dir_all(&receipt_dir)?;

        let exe = std::env::current_exe().context("failed to resolve current bitnet executable")?;
        let mut rows = Vec::with_capacity(corpus.cases.len());
        for case in &corpus.cases {
            let row = if self.dry_run {
                self.not_run_row(case, "dry_run_requested")
            } else {
                self.run_case(&exe, &receipt_dir, &corpus, case, &device, default_timeout_seconds)?
            };
            rows.push(row);
        }

        let total = rows.len();
        let passed = rows.iter().filter(|row| row["quality"]["passed"] == true).count();
        let failed = rows
            .iter()
            .filter(|row| row["status"] == "quality_failed" || row["status"] == "command_failed")
            .count();
        let timed_out = rows.iter().filter(|row| row["status"] == "timeout").count();
        let not_run = rows.iter().filter(|row| row["status"] == "not_run").count();

        let receipt = json!({
            "schema_version": "1.0.0",
            "artifact_kind": artifact_kind,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "corpus": {
                "path": self.corpus.display().to_string(),
                "name": corpus.name,
                "description": corpus.description,
                "case_count": corpus.cases.len(),
            },
            "model": {
                "repo": corpus.model.repo,
                "file": corpus.model.file,
                "path": self.model.display().to_string(),
                "loader_mode": "real_gguf",
                "fallback_loader_used": false,
                "tokenizer": "llama3",
                "tokenizer_path": self.tokenizer.as_ref().map(|path| path.display().to_string()),
            },
            "backend": {
                "requested_backend": device.as_str(),
                "selected_backend": device.as_str(),
                "runtime_api": "cpu",
                "fallback_used": false,
            },
            "prompt_template": {
                "family": corpus.defaults.prompt_template,
            },
            "generation": {
                "mode": if corpus.defaults.greedy { "greedy" } else { "sampling" },
                "temperature": corpus.defaults.temperature,
                "deterministic": corpus.defaults.deterministic,
                "strict_loader": corpus.defaults.strict_loader,
                "default_max_new_tokens": corpus.defaults.max_new_tokens,
                "per_prompt_timeout_seconds": default_timeout_seconds,
            },
            "quality_summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "timeout": timed_out,
                "not_run": not_run,
            },
            "claim_boundary": {
                "local_answer_path": device.as_str() == "apple-m4-cpu-neon",
                "full_metal_inference_claimed": false,
                "qk256_apple_claimed": false,
                "neural_engine_claimed": false,
                "broad_performance_claimed": false,
            },
            "cases": rows,
            "speedup_claim": false,
        });

        if let Some(parent) = self.json_out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.json_out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("answer corpus receipt written to {}", self.json_out.display());

        if self.fail_on_quality && (failed > 0 || timed_out > 0) {
            anyhow::bail!("answer corpus quality failed: {failed} failed, {timed_out} timed out");
        }
        Ok(())
    }

    fn run_case(
        &self,
        exe: &Path,
        receipt_dir: &Path,
        corpus: &AnswerCorpus,
        case: &AnswerCase,
        device: &str,
        default_timeout_seconds: u64,
    ) -> Result<Value> {
        let case_receipt = receipt_dir.join(format!("{}.json", sanitize_file_stem(&case.id)));
        let max_new_tokens = case.max_new_tokens.unwrap_or(corpus.defaults.max_new_tokens);
        let timeout_seconds = case.timeout_seconds.unwrap_or(default_timeout_seconds).max(1);

        let mut args: Vec<OsString> = vec![
            "--device".into(),
            device.into(),
            "run".into(),
            "--model".into(),
            self.model.as_os_str().to_owned(),
            "--prompt".into(),
            case.question.clone().into(),
            "--max-new-tokens".into(),
            max_new_tokens.to_string().into(),
            "--temperature".into(),
            corpus.defaults.temperature.to_string().into(),
            "--prompt-template".into(),
            corpus.defaults.prompt_template.clone().into(),
            "--json-out".into(),
            case_receipt.as_os_str().to_owned(),
        ];
        if let Some(tokenizer) = &self.tokenizer {
            args.push("--tokenizer".into());
            args.push(tokenizer.as_os_str().to_owned());
        }
        if corpus.defaults.greedy {
            args.push("--greedy".into());
        }
        if corpus.defaults.deterministic {
            args.push("--deterministic".into());
        }
        if corpus.defaults.strict_loader {
            args.push("--strict-loader".into());
            args.push("--strict-tokenizer".into());
        }

        let run = run_child_with_timeout(exe, &args, Duration::from_secs(timeout_seconds))?;
        if run.timed_out {
            return Ok(json!({
                "id": case.id,
                "question": case.question,
                "status": "timeout",
                "timeout_seconds": timeout_seconds,
                "run_receipt_path": case_receipt.display().to_string(),
                "quality": {
                    "passed": false,
                    "failed_rules": ["timeout"],
                },
                "stderr_tail": tail_string(&run.stderr, 4096),
            }));
        }
        if !run.success {
            return Ok(json!({
                "id": case.id,
                "question": case.question,
                "status": "command_failed",
                "exit_code": run.exit_code,
                "run_receipt_path": case_receipt.display().to_string(),
                "quality": {
                    "passed": false,
                    "failed_rules": ["command_failed"],
                },
                "stdout_tail": tail_string(&run.stdout, 4096),
                "stderr_tail": tail_string(&run.stderr, 4096),
            }));
        }

        let run_receipt: Value = serde_json::from_slice(
            &fs::read(&case_receipt)
                .with_context(|| format!("missing run receipt {}", case_receipt.display()))?,
        )
        .with_context(|| format!("invalid run receipt {}", case_receipt.display()))?;
        let answer = run_receipt["text"].as_str().unwrap_or_default().to_string();
        let token_ids = generated_token_ids(&run_receipt);
        let generated_token_count = run_receipt["tokens"]["generated"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(token_ids.len());
        let min_generated_tokens =
            case.min_generated_tokens.or(corpus.defaults.min_generated_tokens);
        let min_distinct_generated_tokens =
            case.min_distinct_generated_tokens.or(corpus.defaults.min_distinct_generated_tokens);
        let mut quality = evaluate_quality(
            &answer,
            &case.gate,
            Some(&token_ids),
            min_generated_tokens,
            min_distinct_generated_tokens,
        );
        quality.failed_rules.extend(answer_receipt_failed_rules(&run_receipt, device));
        quality.passed = quality.failed_rules.is_empty();
        let status = if quality.passed { "passed" } else { "quality_failed" };

        Ok(json!({
            "id": case.id,
            "question": case.question,
            "status": status,
            "run_receipt_path": case_receipt.display().to_string(),
            "answer": answer,
            "tokens": run_receipt.get("tokens").cloned().unwrap_or(Value::Null),
            "token_ids": {
                "prompt": run_receipt["tokens"]["prompt_ids"].clone(),
                "generated": run_receipt["tokens"]["generated_ids"].clone(),
            },
            "prompt_template": corpus.defaults.prompt_template,
            "quality": {
                "passed": quality.passed,
                "printable_utf8": quality.printable_utf8,
                "non_empty_answer": quality.non_empty_answer,
                "no_replacement_chars": quality.no_replacement_chars,
                "no_raw_special_tokens": quality.no_raw_special_tokens,
                "mostly_text": quality.mostly_text,
                "generated_tokens": generated_token_count,
                "distinct_generated_tokens": quality.distinct_generated_tokens,
                "min_generated_tokens": min_generated_tokens,
                "min_distinct_generated_tokens": min_distinct_generated_tokens,
                "gate_kind": case.gate.kind,
                "failed_rules": quality.failed_rules,
            },
            "backend": {
                "requested_backend": run_receipt["requested_backend"].clone(),
                "selected_backend": run_receipt["selected_backend"].clone(),
                "runtime_api": run_receipt["runtime_api"].clone(),
                "fallback_used": run_receipt["fallback_used"].clone(),
            },
            "kernel": {
                "selected_kernel": run_receipt["kernel"]["kernel_id"].clone(),
                "family": run_receipt["kernel"]["family"].clone(),
            },
            "loader": {
                "mode": run_receipt["loader"]["mode"].clone(),
            },
            "tokenizer": {
                "source": run_receipt["tokenizer"]["source"].clone(),
                "strict": run_receipt["tokenizer"]["strict"].clone(),
            }
        }))
    }

    fn not_run_row(&self, case: &AnswerCase, reason: &str) -> Value {
        json!({
            "id": case.id,
            "question": case.question,
            "status": "not_run",
            "reason": reason,
            "quality": {
                "passed": false,
                "failed_rules": ["not_run"],
            }
        })
    }
}

#[derive(Debug, Deserialize)]
struct AnswerCorpus {
    schema: u32,
    artifact_kind: String,
    name: String,
    description: String,
    model: CorpusModel,
    defaults: CorpusDefaults,
    cases: Vec<AnswerCase>,
}

impl AnswerCorpus {
    fn load(path: &Path) -> Result<Self> {
        let corpus: Self = serde_yaml::from_slice(
            &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))?;
        if corpus.schema != 1 {
            anyhow::bail!("unsupported answer corpus schema {}", corpus.schema);
        }
        if corpus.artifact_kind != "bitnet_answer_corpus" {
            anyhow::bail!("unexpected answer corpus artifact_kind {}", corpus.artifact_kind);
        }
        if corpus.cases.is_empty() {
            anyhow::bail!("answer corpus must contain at least one case");
        }
        Ok(corpus)
    }
}

#[derive(Debug, Deserialize)]
struct CorpusModel {
    repo: String,
    file: String,
}

#[derive(Debug, Deserialize)]
struct CorpusDefaults {
    prompt_template: String,
    max_new_tokens: usize,
    greedy: bool,
    deterministic: bool,
    strict_loader: bool,
    temperature: f32,
    per_prompt_timeout_seconds: Option<u64>,
    min_generated_tokens: Option<usize>,
    min_distinct_generated_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct AnswerCase {
    id: String,
    question: String,
    max_new_tokens: Option<usize>,
    timeout_seconds: Option<u64>,
    min_generated_tokens: Option<usize>,
    min_distinct_generated_tokens: Option<usize>,
    gate: AnswerGate,
}

#[derive(Debug, Deserialize)]
struct AnswerGate {
    kind: String,
    expected: Option<String>,
    contains_any: Option<Vec<String>>,
    starts_with_any: Option<Vec<String>>,
    min_words: Option<usize>,
}

struct QualityResult {
    passed: bool,
    printable_utf8: bool,
    non_empty_answer: bool,
    no_replacement_chars: bool,
    no_raw_special_tokens: bool,
    mostly_text: bool,
    distinct_generated_tokens: usize,
    failed_rules: Vec<String>,
}

fn normalize_answer_corpus_device(device: &str) -> String {
    match device.trim() {
        "auto" => "cpu".to_string(),
        other => other.to_string(),
    }
}

fn answer_corpus_artifact_kind(device: &str) -> &'static str {
    match device {
        "apple-m4-cpu-neon" => "bitnet_apple_m4_local_answer_corpus",
        _ => "bitnet_cpu_answer_corpus",
    }
}

fn effective_default_timeout_seconds(cli: Option<u64>, corpus: Option<u64>) -> u64 {
    cli.or(corpus).unwrap_or(300).max(1)
}

fn evaluate_quality(
    answer: &str,
    gate: &AnswerGate,
    generated_token_ids: Option<&[u32]>,
    min_generated_tokens: Option<usize>,
    min_distinct_generated_tokens: Option<usize>,
) -> QualityResult {
    let normalized = strip_special_markers(answer).trim().to_string();
    let non_empty_answer = !normalized.is_empty();
    let no_replacement_chars = !normalized.contains('\u{FFFD}');
    let no_raw_special_tokens = !contains_raw_special_token(&normalized);
    let mostly_text = mostly_text(&normalized);
    let printable_utf8 = normalized.chars().all(|ch| ch == '\n' || ch == '\t' || !ch.is_control());
    let generated_token_count = generated_token_ids.map(|tokens| tokens.len()).unwrap_or(0);
    let distinct_generated_tokens = generated_token_ids
        .map(|tokens| tokens.iter().copied().collect::<std::collections::BTreeSet<_>>().len())
        .unwrap_or(0);

    let mut failed_rules = Vec::new();
    if !non_empty_answer {
        failed_rules.push("empty_answer".to_string());
    }
    if !no_replacement_chars {
        failed_rules.push("replacement_chars".to_string());
    }
    if !no_raw_special_tokens {
        failed_rules.push("raw_special_tokens".to_string());
    }
    if !mostly_text {
        failed_rules.push("mostly_text".to_string());
    }
    if !printable_utf8 {
        failed_rules.push("printable_utf8".to_string());
    }

    if !gate_passed(&normalized, gate) {
        failed_rules.push(format!("gate_{}", gate.kind));
    }
    if let Some(minimum) = min_generated_tokens
        && generated_token_count < minimum
    {
        failed_rules.push("generated_token_min".to_string());
    }
    if let Some(minimum) = min_distinct_generated_tokens
        && distinct_generated_tokens < minimum
    {
        failed_rules.push("generated_token_variation".to_string());
    }

    QualityResult {
        passed: failed_rules.is_empty(),
        printable_utf8,
        non_empty_answer,
        no_replacement_chars,
        no_raw_special_tokens,
        mostly_text,
        distinct_generated_tokens,
        failed_rules,
    }
}

fn answer_receipt_failed_rules(run_receipt: &Value, expected_backend: &str) -> Vec<String> {
    let mut failed = Vec::new();
    let requested_backend = run_receipt["requested_backend"].as_str().unwrap_or_default();
    let selected_backend = run_receipt["selected_backend"].as_str().unwrap_or_default();
    let runtime_api = run_receipt["runtime_api"].as_str().unwrap_or_default();
    let fallback_used = run_receipt["fallback_used"].as_bool().unwrap_or(true);
    if requested_backend != expected_backend {
        failed.push(format!("requested_backend_{expected_backend}"));
    }
    let selected_backend_valid = match expected_backend {
        "cpu" => matches!(selected_backend, "cpu" | "cpu-rust"),
        "apple-m4-cpu-neon" => selected_backend == "apple-m4-cpu-neon",
        _ => false,
    };
    if !selected_backend_valid {
        failed.push(format!("selected_backend_{expected_backend}"));
    }
    if runtime_api != "cpu" {
        failed.push("runtime_api_cpu".to_string());
    }
    if fallback_used {
        failed.push("fallback_false".to_string());
    }

    let loader_mode = run_receipt["loader"]["mode"]
        .as_str()
        .or_else(|| run_receipt["model"]["loader_mode"].as_str())
        .unwrap_or_default();
    if loader_mode != "real_gguf" {
        failed.push("loader_real_gguf".to_string());
    }

    let tokenizer_source = run_receipt["tokenizer"]["source"].as_str().unwrap_or_default();
    let tokenizer_strict = run_receipt["tokenizer"]["strict"].as_bool().unwrap_or(false);
    if tokenizer_source.is_empty() || tokenizer_source == "unknown" {
        failed.push("tokenizer_source_recorded".to_string());
    }
    if !tokenizer_strict {
        failed.push("tokenizer_strict".to_string());
    }

    let selected_kernel = run_receipt["kernel"]["kernel_id"].as_str().unwrap_or_default();
    if selected_kernel.is_empty() {
        failed.push("selected_kernel_recorded".to_string());
    }
    if selected_kernel.contains("mock") || selected_kernel.contains("diagnostic") {
        failed.push("selected_kernel_production".to_string());
    }

    if !run_receipt["tokens"]["prompt_ids"].is_array() {
        failed.push("prompt_token_ids_recorded".to_string());
    }
    if generated_token_ids(run_receipt).is_empty() {
        failed.push("generated_token_ids_recorded".to_string());
    }
    failed
}

fn generated_token_ids(receipt: &Value) -> Vec<u32> {
    receipt["tokens"]["generated_ids"]
        .as_array()
        .or_else(|| receipt["tokens"]["ids"].as_array())
        .map(|ids| {
            ids.iter()
                .filter_map(|value| value.as_u64().and_then(|id| u32::try_from(id).ok()))
                .collect()
        })
        .unwrap_or_default()
}

fn gate_passed(answer: &str, gate: &AnswerGate) -> bool {
    match gate.kind.as_str() {
        "exact_trimmed" => {
            let Some(expected) = &gate.expected else {
                return false;
            };
            answer.trim().eq_ignore_ascii_case(expected.trim())
        }
        "contains_any" => {
            let lower = answer.to_ascii_lowercase();
            gate.contains_any.as_ref().is_some_and(|items| {
                items.iter().any(|needle| lower.contains(&needle.to_ascii_lowercase()))
            })
        }
        "starts_with_any" => {
            let lower = answer.trim_start().to_ascii_lowercase();
            gate.starts_with_any.as_ref().is_some_and(|items| {
                items.iter().any(|needle| lower.starts_with(&needle.to_ascii_lowercase()))
            })
        }
        "readable" => word_count(answer) >= gate.min_words.unwrap_or(1),
        _ => false,
    }
}

fn strip_special_markers(answer: &str) -> String {
    answer.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "").replace("<|eot_id|>", "")
}

fn contains_raw_special_token(answer: &str) -> bool {
    answer.contains("<|") || answer.contains("|>")
}

fn mostly_text(answer: &str) -> bool {
    let mut meaningful = 0usize;
    let mut punctuation_or_control = 0usize;
    for ch in answer.chars() {
        if ch.is_alphanumeric() || ch.is_whitespace() {
            meaningful += 1;
        } else if ch.is_ascii_punctuation() || ch.is_control() {
            punctuation_or_control += 1;
        }
    }
    meaningful > 0 && punctuation_or_control <= meaningful.saturating_mul(2)
}

fn word_count(answer: &str) -> usize {
    answer.split_whitespace().filter(|word| word.chars().any(char::is_alphanumeric)).count()
}

struct ChildRun {
    success: bool,
    timed_out: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn run_child_with_timeout(exe: &Path, args: &[OsString], timeout: Duration) -> Result<ChildRun> {
    let child_rust_log =
        std::env::var("BITNET_ANSWER_CORPUS_CHILD_RUST_LOG").unwrap_or_else(|_| "warn".into());
    let stdout_path = child_capture_path("stdout");
    let stderr_path = child_capture_path("stderr");
    let stdout_file = File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr_file = File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let mut child = Command::new(exe)
        .args(args)
        .env("RUST_LOG", child_rust_log)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .with_context(|| format!("failed to spawn {}", exe.display()))?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            let stdout = read_child_capture(&stdout_path);
            let stderr = read_child_capture(&stderr_path);
            remove_child_capture(&stdout_path);
            remove_child_capture(&stderr_path);
            return Ok(ChildRun {
                success: status.success(),
                timed_out: false,
                exit_code: status.code(),
                stdout,
                stderr,
            });
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let status = child.wait()?;
            let stdout = read_child_capture(&stdout_path);
            let stderr = read_child_capture(&stderr_path);
            remove_child_capture(&stdout_path);
            remove_child_capture(&stderr_path);
            return Ok(ChildRun {
                success: false,
                timed_out: true,
                exit_code: status.code(),
                stdout,
                stderr,
            });
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn child_capture_path(kind: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir()
        .join(format!("bitnet-answer-corpus-{}-{nanos}-{sequence}-{kind}.log", std::process::id()))
}

fn read_child_capture(path: &Path) -> String {
    fs::read(path).map(|bytes| String::from_utf8_lossy(&bytes).into_owned()).unwrap_or_default()
}

fn remove_child_capture(path: &Path) {
    let _ = fs::remove_file(path);
}

fn tail_string(value: &str, max_chars: usize) -> String {
    let len = value.chars().count();
    if len <= max_chars { value.to_string() } else { value.chars().skip(len - max_chars).collect() }
}

fn sanitize_file_stem(id: &str) -> String {
    id.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' { ch } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(kind: &str) -> AnswerGate {
        AnswerGate {
            kind: kind.to_string(),
            expected: None,
            contains_any: None,
            starts_with_any: None,
            min_words: None,
        }
    }

    #[test]
    fn exact_gate_accepts_trimmed_answer() {
        let gate = AnswerGate { expected: Some("4".to_string()), ..gate("exact_trimmed") };
        let quality = evaluate_quality(" 4\n", &gate, None, None, None);
        assert!(quality.passed);
    }

    #[test]
    fn quality_rejects_raw_special_tokens() {
        let quality =
            evaluate_quality("<|start_header_id|>assistant", &gate("readable"), None, None, None);
        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"raw_special_tokens".to_string()));
    }

    #[test]
    fn quality_rejects_punctuation_noise() {
        let quality = evaluate_quality("!!!,,,!!!", &gate("readable"), None, None, None);
        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"mostly_text".to_string()));
    }

    #[test]
    fn apple_m4_quality_rejects_short_or_degenerate_token_output() {
        let gate = AnswerGate { min_words: Some(2), ..gate("readable") };
        let quality = evaluate_quality("short answer", &gate, Some(&[7, 7, 7]), Some(4), Some(2));
        assert!(!quality.passed);
        assert!(quality.failed_rules.contains(&"generated_token_min".to_string()));
        assert!(quality.failed_rules.contains(&"generated_token_variation".to_string()));
        assert_eq!(quality.distinct_generated_tokens, 1);
    }

    #[test]
    fn cli_timeout_overrides_corpus_default() {
        assert_eq!(effective_default_timeout_seconds(Some(1), Some(300)), 1);
        assert_eq!(effective_default_timeout_seconds(None, Some(120)), 120);
        assert_eq!(effective_default_timeout_seconds(None, None), 300);
        assert_eq!(effective_default_timeout_seconds(Some(0), Some(300)), 1);
    }

    #[test]
    fn cpu_answer_receipt_accepts_strict_cpu_truth() {
        let receipt = json!({
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": { "source": "gguf_metadata", "strict": true },
            "kernel": { "kernel_id": "i2_s-avx2-reference" },
            "tokens": {
                "prompt_ids": [1, 2, 3],
                "generated_ids": [4]
            }
        });

        assert!(answer_receipt_failed_rules(&receipt, "cpu").is_empty());
    }

    #[test]
    fn answer_receipt_accepts_strict_apple_m4_cpu_neon_truth() {
        let receipt = json!({
            "requested_backend": "apple-m4-cpu-neon",
            "selected_backend": "apple-m4-cpu-neon",
            "runtime_api": "cpu",
            "fallback_used": false,
            "loader": { "mode": "real_gguf" },
            "tokenizer": { "source": "gguf_metadata", "strict": true },
            "kernel": { "kernel_id": "i2_s-scalar-reference" },
            "tokens": {
                "prompt_ids": [1, 2, 3],
                "ids": [4]
            }
        });

        assert!(answer_receipt_failed_rules(&receipt, "apple-m4-cpu-neon").is_empty());
    }

    #[test]
    fn cpu_answer_receipt_rejects_hidden_fallback_and_missing_ids() {
        let receipt = json!({
            "requested_backend": "cpu",
            "selected_backend": "cpu-rust",
            "runtime_api": "cpu",
            "fallback_used": true,
            "loader": { "mode": "minimal_compatibility" },
            "tokenizer": { "source": "unknown", "strict": false },
            "kernel": { "kernel_id": "mock-diagnostic" },
            "tokens": {}
        });

        let failed = answer_receipt_failed_rules(&receipt, "cpu");
        assert!(failed.contains(&"fallback_false".to_string()));
        assert!(failed.contains(&"loader_real_gguf".to_string()));
        assert!(failed.contains(&"tokenizer_source_recorded".to_string()));
        assert!(failed.contains(&"tokenizer_strict".to_string()));
        assert!(failed.contains(&"selected_kernel_production".to_string()));
        assert!(failed.contains(&"prompt_token_ids_recorded".to_string()));
        assert!(failed.contains(&"generated_token_ids_recorded".to_string()));
    }
}
