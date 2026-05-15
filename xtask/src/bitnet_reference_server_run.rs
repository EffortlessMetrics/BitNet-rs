use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const CRITICAL_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
    "reference_generated_token_ids",
    "reference_top_logits",
    "rust_reference_parity_proven",
    "a770_semantic_quality_proven",
];

#[derive(Debug)]
struct ReferenceServerRunArgs {
    plan: PathBuf,
    reference_run_receipt: PathBuf,
    server_exe: Option<PathBuf>,
    output: Option<PathBuf>,
    format: String,
    port: Option<u16>,
    timeout_seconds: u64,
    max_new_tokens: u64,
    n_probs: u64,
}

struct ServerGuard {
    child: Child,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-server-run") {
        return Ok(false);
    }
    if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(true);
    }
    let opts = parse_args(args)?;
    let report = run_reference_server(&opts)?;
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
        "Run the BitNet C++ reference server completion endpoint and emit a diagnostic receipt\n\nUsage: xtask.exe bitnet-reference-server-run [OPTIONS]\n\nOptions:\n      --plan <PATH>                  Reference plan JSON [default: target/a770-diagnostic/bitnet-reference-plan.json]\n      --reference-run-receipt <PATH> Reference run receipt with verified prompt token IDs [default: target/a770-diagnostic/bitnet-reference-run.json]\n      --server-exe <PATH>            Explicit llama-server executable path\n      --output <PATH>                Output reference server run JSON [default: target/a770-diagnostic/bitnet-reference-server-run.json]\n      --format <FORMAT>              Output format: human or json [default: human]\n      --port <PORT>                  Local server port [default: auto]\n      --timeout-seconds <SECONDS>    Startup/request timeout [default: 120]\n      --max-new-tokens <N>           Tokens to request from the server [default: 1]\n      --n-probs <N>                  Top probability strings to request [default: 10]\n  -h, --help                         Print help"
    );
}

fn parse_args(args: &[String]) -> Result<ReferenceServerRunArgs> {
    let mut plan = PathBuf::from("target/a770-diagnostic/bitnet-reference-plan.json");
    let mut reference_run_receipt =
        PathBuf::from("target/a770-diagnostic/bitnet-reference-run.json");
    let mut server_exe = None;
    let mut output = Some(PathBuf::from("target/a770-diagnostic/bitnet-reference-server-run.json"));
    let mut format = "human".to_string();
    let mut port = None;
    let mut timeout_seconds = 120u64;
    let mut max_new_tokens = 1u64;
    let mut n_probs = 10u64;
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
            "--plan" => plan = PathBuf::from(value()?),
            "--reference-run-receipt" => reference_run_receipt = PathBuf::from(value()?),
            "--server-exe" => server_exe = Some(PathBuf::from(value()?)),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            "--port" => {
                let raw = value()?;
                port = Some(raw.parse().with_context(|| format!("parsing --port {raw}"))?);
            }
            "--timeout-seconds" => {
                let raw = value()?;
                timeout_seconds =
                    raw.parse().with_context(|| format!("parsing --timeout-seconds {raw}"))?;
            }
            "--max-new-tokens" => {
                let raw = value()?;
                max_new_tokens =
                    raw.parse().with_context(|| format!("parsing --max-new-tokens {raw}"))?;
            }
            "--n-probs" => {
                let raw = value()?;
                n_probs = raw.parse().with_context(|| format!("parsing --n-probs {raw}"))?;
            }
            other => bail!("unknown bitnet-reference-server-run option {other}"),
        }
    }
    Ok(ReferenceServerRunArgs {
        plan,
        reference_run_receipt,
        server_exe,
        output,
        format,
        port,
        timeout_seconds,
        max_new_tokens,
        n_probs,
    })
}

fn run_reference_server(args: &ReferenceServerRunArgs) -> Result<Value> {
    let plan = read_json(&args.plan)?;
    let reference_run_receipt = read_json(&args.reference_run_receipt)?;
    let prompt_token_ids = prompt_token_ids_from_reference_run(&reference_run_receipt)?;
    let server_exe =
        args.server_exe.clone().or_else(|| server_executable_from_plan(&plan)).context(
            "server executable missing; pass --server-exe or run bitnet-reference-plan first",
        )?;
    let model_path =
        str_at(&plan, "/model/model_path").context("plan missing /model/model_path")?;
    let port = match args.port {
        Some(port) => port,
        None => available_local_port()?,
    };
    let mut guard = spawn_server(&server_exe, model_path, port)?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(args.timeout_seconds))
        .build()
        .context("building reference server HTTP client")?;
    wait_for_health(&client, port, &mut guard, Duration::from_secs(args.timeout_seconds))?;
    let request = json!({
        "prompt": prompt_token_ids,
        "n_predict": args.max_new_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": false,
        "n_probs": args.n_probs,
        "cache_prompt": false,
    });
    let started = Instant::now();
    let response: Value = client
        .post(format!("http://127.0.0.1:{port}/completion"))
        .json(&request)
        .send()
        .context("posting reference server completion request")?
        .error_for_status()
        .context("reference server completion request failed")?
        .json()
        .context("parsing reference server completion JSON")?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(build_receipt(
        &args.plan,
        &args.reference_run_receipt,
        &plan,
        &reference_run_receipt,
        &server_exe,
        port,
        &request,
        &response,
        elapsed_ms,
    ))
}

fn spawn_server(server_exe: &Path, model_path: &str, port: u16) -> Result<ServerGuard> {
    let mut command = Command::new(server_exe);
    command
        .args([
            "-m",
            model_path,
            "--override-kv",
            "tokenizer.ggml.add_bos_token=bool:false",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "-c",
            "4096",
            "-n",
            "1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    hide_child_window(&mut command);
    let child = command
        .spawn()
        .with_context(|| format!("starting reference server {}", server_exe.display()))?;
    Ok(ServerGuard { child })
}

#[cfg(windows)]
fn hide_child_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_child_window(_command: &mut Command) {}

fn wait_for_health(
    client: &reqwest::blocking::Client,
    port: u16,
    guard: &mut ServerGuard,
    timeout: Duration,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(Some(status)) = guard.child.try_wait() {
            bail!("reference server exited early with status {status}");
        }
        if let Ok(response) = client.get(format!("http://127.0.0.1:{port}/health")).send()
            && response.status().is_success()
        {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(250));
    }
    bail!("reference server did not become healthy within {}s", timeout.as_secs())
}

#[allow(clippy::too_many_arguments)]
fn build_receipt(
    plan_path: &Path,
    reference_run_receipt_path: &Path,
    plan: &Value,
    reference_run_receipt: &Value,
    server_exe: &Path,
    port: u16,
    request: &Value,
    response: &Value,
    elapsed_ms: f64,
) -> Value {
    let prompt_token_ids =
        array_u32(reference_run_receipt, "/reference_prompt_tokenization/token_ids");
    let prompt_token_ids_sha256 = sha256_token_ids(&prompt_token_ids).ok();
    let plan_prompt_token_ids_sha256 = str_at(plan, "/prompt_identity/prompt_token_ids_sha256");
    let prompt_token_ids_sha256_matches_plan =
        prompt_token_ids_sha256.as_deref().zip(plan_prompt_token_ids_sha256).map(|(a, b)| a == b);
    let plan_prompt_token_count = u64_at(plan, "/prompt_identity/prompt_token_count");
    let prompt_token_count_matches_plan =
        plan_prompt_token_count.map(|count| count as usize == prompt_token_ids.len());
    let completion_probabilities =
        response.pointer("/completion_probabilities").cloned().unwrap_or(Value::Null);
    let first_completion = response.pointer("/completion_probabilities/0");
    let first_selected_content =
        first_completion.and_then(|value| value.pointer("/content")).and_then(Value::as_str);
    let first_top_probability_strings = first_completion
        .and_then(|value| value.pointer("/probs"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    json!({
                        "tok_str": item.pointer("/tok_str").and_then(Value::as_str),
                        "prob": item.pointer("/prob").and_then(Value::as_f64),
                        "token_id_present": item.pointer("/id")
                            .or_else(|| item.pointer("/token_id"))
                            .is_some(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let top_probability_token_ids_present = first_top_probability_strings
        .iter()
        .any(|item| item.pointer("/token_id_present").and_then(Value::as_bool) == Some(true));
    let mut blocked_reasons = Vec::new();
    if response.pointer("/content").and_then(Value::as_str).unwrap_or("").is_empty() {
        blocked_reasons.push("reference_server_content_missing".to_string());
    }
    if first_selected_content.is_none() {
        blocked_reasons.push("reference_server_selected_content_missing".to_string());
    }
    if prompt_token_ids_sha256_matches_plan == Some(false) {
        blocked_reasons.push("reference_server_prompt_token_ids_sha256_mismatch".to_string());
    }
    if prompt_token_count_matches_plan == Some(false) {
        blocked_reasons.push("reference_server_prompt_token_count_mismatch".to_string());
    }

    json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_server_run",
        "diagnostic": "bitnet_reference_server_run",
        "producer": "cargo xtask bitnet-reference-server-run",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "plan": {
            "path": plan_path.display().to_string(),
            "diagnostic": str_at(plan, "/diagnostic").unwrap_or(""),
            "prompt_identity": plan.pointer("/prompt_identity").cloned().unwrap_or(Value::Null),
            "model": plan.pointer("/model").cloned().unwrap_or(Value::Null),
        },
        "reference_run_receipt": {
            "path": reference_run_receipt_path.display().to_string(),
            "receipt_type": str_at(reference_run_receipt, "/receipt_type").unwrap_or(""),
            "reference_prompt_tokenization": reference_run_receipt
                .pointer("/reference_prompt_tokenization")
                .cloned()
                .unwrap_or(Value::Null),
        },
        "server": {
            "executable": server_exe.display().to_string(),
            "backend": "llama-server_completion_endpoint",
            "port": port,
        },
        "request": {
            "elapsed_ms": elapsed_ms,
            "prompt_token_count": prompt_token_ids.len(),
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "plan_prompt_token_count": plan_prompt_token_count,
            "plan_prompt_token_ids_sha256": plan_prompt_token_ids_sha256,
            "prompt_token_count_matches_plan": prompt_token_count_matches_plan,
            "prompt_token_ids_sha256_matches_plan": prompt_token_ids_sha256_matches_plan,
            "body_sha256": sha256_json(request),
            "n_predict": request.pointer("/n_predict").and_then(Value::as_u64),
            "n_probs": request.pointer("/n_probs").and_then(Value::as_u64),
            "temperature": request.pointer("/temperature").and_then(Value::as_f64),
            "seed": request.pointer("/seed").and_then(Value::as_u64),
            "stream": request.pointer("/stream").and_then(Value::as_bool),
            "cache_prompt": request.pointer("/cache_prompt").and_then(Value::as_bool),
        },
        "response": {
            "content": response.pointer("/content").cloned().unwrap_or(Value::Null),
            "tokens_predicted": response.pointer("/tokens_predicted").cloned().unwrap_or(Value::Null),
            "tokens_evaluated": response.pointer("/tokens_evaluated").cloned().unwrap_or(Value::Null),
            "stopped_limit": response.pointer("/stopped_limit").cloned().unwrap_or(Value::Null),
            "stopped_eos": response.pointer("/stopped_eos").cloned().unwrap_or(Value::Null),
            "timings": response.pointer("/timings").cloned().unwrap_or(Value::Null),
            "completion_probabilities": completion_probabilities,
            "response_sha256": sha256_json(response),
        },
        "reference_probability_summary": {
            "source": "llama-server_completion_probabilities",
            "diagnostic_only": true,
            "claimable": false,
            "selected_content": first_selected_content,
            "top_probability_strings": first_top_probability_strings,
            "top_probability_token_ids_present": top_probability_token_ids_present,
            "policy": "llama-server completion_probabilities expose selected text and probability strings in this build; token ids and raw logits are not present and are not inferred",
            "not_claims": [
                "reference_probability_strings_prove_reference_generated_token_ids",
                "reference_probability_strings_prove_reference_top_logits",
                "reference_probability_strings_promote_reference_parity",
                "reference_probability_strings_promote_a770_semantic_quality"
            ],
        },
        "signals": {
            "server_completion_present": response.pointer("/content").and_then(Value::as_str).is_some_and(|text| !text.is_empty()),
            "completion_probabilities_present": response.pointer("/completion_probabilities").is_some(),
            "selected_content_present": first_selected_content.is_some(),
            "top_probability_strings_present": !first_top_probability_strings.is_empty(),
            "actual_generated_token_ids_present": false,
            "top_logits_present": false,
            "policy": "reference server receipts capture exact prompt-token input and probability strings only; generated token ids and top logits remain unclaimed unless the reference endpoint provides token ids/logits explicitly",
        },
        "decision": {
            "reference_server_execution_ready": blocked_reasons.is_empty(),
            "current_blocked_reasons": blocked_reasons,
            "next_when_ready": "compare reference selected text/probability strings against Rust CPU and strict A770 surfaces; use a deeper reference hook before token-id/logit parity claims",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    })
}

fn server_executable_from_plan(plan: &Value) -> Option<PathBuf> {
    let cli = str_at(plan, "/reference/selected_executable")?;
    let cli_path = Path::new(cli);
    let mut server = cli_path.to_path_buf();
    let file_name = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
    server.set_file_name(file_name);
    Some(server)
}

fn prompt_token_ids_from_reference_run(reference_run_receipt: &Value) -> Result<Vec<u32>> {
    let ids = array_u32(reference_run_receipt, "/reference_prompt_tokenization/token_ids");
    if ids.is_empty() {
        bail!("reference run receipt missing /reference_prompt_tokenization/token_ids");
    }
    Ok(ids)
}

fn available_local_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("binding temporary local port")?;
    Ok(listener.local_addr()?.port())
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

fn u64_at(value: &Value, pointer: &str) -> Option<u64> {
    value.pointer(pointer).and_then(Value::as_u64)
}

fn array_u32(value: &Value, pointer: &str) -> Vec<u32> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .map(|items| {
            items.iter().filter_map(Value::as_u64).filter_map(|id| u32::try_from(id).ok()).collect()
        })
        .unwrap_or_default()
}

fn sha256_json<T: serde::Serialize + ?Sized>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    sha256_bytes(&bytes)
}

fn sha256_token_ids(tokens: &[u32]) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(tokens)?))
}

fn sha256_bytes(value: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value);
    format!("{:x}", hasher.finalize())
}

fn emit_report(value: &Value, format: &str) -> Result<()> {
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(value)?),
        "human" => {
            println!("diagnostic: bitnet_reference_server_run");
            println!(
                "classification: {}",
                value
                    .pointer("/classification")
                    .and_then(Value::as_str)
                    .unwrap_or("diagnostic_only")
            );
            println!(
                "reference_server_execution_ready: {}",
                value
                    .pointer("/decision/reference_server_execution_ready")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            );
            if let Some(content) = value.pointer("/response/content").and_then(Value::as_str) {
                println!("content: {content}");
            }
            if let Some(selected) = value
                .pointer("/reference_probability_summary/selected_content")
                .and_then(Value::as_str)
            {
                println!("selected_content: {selected}");
            }
            if let Some(reasons) = value.pointer("/decision/current_blocked_reasons") {
                println!("blocked_reasons: {}", serde_json::to_string(reasons)?);
            }
            println!("not_claims: {}", serde_json::to_string(&value["not_claims"])?);
        }
        other => bail!("unsupported bitnet-reference-server-run output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn derives_server_executable_next_to_cli() {
        let plan = json!({
            "reference": {
                "selected_executable": "target/external/BitNet-reference/build/bin/llama-cli.exe"
            }
        });

        let server = server_executable_from_plan(&plan).unwrap();

        assert!(server.to_string_lossy().ends_with("llama-server.exe"));
    }

    #[test]
    fn server_receipt_records_probability_strings_without_promoting_token_ids() {
        let prompt_ids = vec![128000, 128006, 882, 128007];
        let prompt_hash = sha256_token_ids(&prompt_ids).unwrap();
        let plan = json!({
            "diagnostic": "bitnet_reference_plan",
            "model": {"model_path": "model.gguf"},
            "prompt_identity": {
                "prompt_token_count": prompt_ids.len(),
                "prompt_token_ids_sha256": prompt_hash,
            },
            "reference": {
                "selected_executable": "target/external/BitNet-reference/build/bin/llama-cli.exe"
            }
        });
        let reference_run_receipt = json!({
            "receipt_type": "bitnet_reference_run",
            "reference_prompt_tokenization": {
                "token_ids": prompt_ids,
            }
        });
        let request = json!({
            "prompt": [128000, 128006, 882, 128007],
            "n_predict": 1,
            "temperature": 0.0,
            "seed": 0,
            "stream": false,
            "n_probs": 2,
            "cache_prompt": false,
        });
        let response = json!({
            "content": "2",
            "tokens_predicted": 1,
            "tokens_evaluated": 4,
            "stopped_limit": true,
            "completion_probabilities": [
                {
                    "content": "2",
                    "probs": [
                        {"tok_str": "2", "prob": 0.75},
                        {"tok_str": "The", "prob": 0.12}
                    ]
                }
            ]
        });

        let receipt = build_receipt(
            Path::new("plan.json"),
            Path::new("reference-run.json"),
            &plan,
            &reference_run_receipt,
            Path::new("llama-server.exe"),
            18081,
            &request,
            &response,
            5.0,
        );

        assert_eq!(receipt["claim_allowed"], false);
        assert_eq!(receipt["decision"]["reference_server_execution_ready"], true);
        assert_eq!(receipt["request"]["prompt_token_count_matches_plan"], true);
        assert_eq!(receipt["request"]["prompt_token_ids_sha256_matches_plan"], true);
        assert_eq!(receipt["reference_probability_summary"]["selected_content"], json!("2"));
        assert_eq!(receipt["signals"]["selected_content_present"], true);
        assert_eq!(receipt["signals"]["top_probability_strings_present"], true);
        assert_eq!(receipt["signals"]["actual_generated_token_ids_present"], false);
        assert_eq!(receipt["signals"]["top_logits_present"], false);
        assert_eq!(
            receipt["reference_probability_summary"]["top_probability_token_ids_present"],
            false
        );
        let not_claims = receipt["not_claims"].as_array().unwrap();
        assert!(not_claims.contains(&json!("reference_generated_token_ids")));
        assert!(not_claims.contains(&json!("reference_top_logits")));
        assert!(not_claims.contains(&json!("rust_reference_parity_proven")));
        assert!(not_claims.contains(&json!("a770_semantic_quality_proven")));
    }

    #[test]
    fn server_receipt_blocks_prompt_hash_mismatch() {
        let plan = json!({
            "prompt_identity": {
                "prompt_token_count": 2,
                "prompt_token_ids_sha256": sha256_token_ids(&[1, 2]).unwrap(),
            },
            "reference": {
                "selected_executable": "target/external/BitNet-reference/build/bin/llama-cli.exe"
            }
        });
        let reference_run_receipt = json!({
            "reference_prompt_tokenization": {
                "token_ids": [1, 3],
            }
        });
        let receipt = build_receipt(
            Path::new("plan.json"),
            Path::new("reference-run.json"),
            &plan,
            &reference_run_receipt,
            Path::new("llama-server.exe"),
            18081,
            &json!({"prompt": [1, 3], "n_predict": 1, "n_probs": 1}),
            &json!({
                "content": "x",
                "completion_probabilities": [{"content": "x", "probs": []}]
            }),
            5.0,
        );

        assert_eq!(receipt["decision"]["reference_server_execution_ready"], false);
        let reasons = receipt["decision"]["current_blocked_reasons"].as_array().unwrap();
        assert!(reasons.contains(&json!("reference_server_prompt_token_ids_sha256_mismatch")));
    }
}
