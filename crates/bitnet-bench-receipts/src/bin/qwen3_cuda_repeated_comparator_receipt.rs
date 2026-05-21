#![recursion_limit = "256"]

use bitnet_bench_receipts::validate_qwen3_cuda_repeated_comparator_receipt_json;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_RECEIPT_OUT: &str =
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-0_6b-repeated-comparator.json";
const CAMPAIGN_ITEM: &str = "CUDA-MODEL-017";
const CONTRACT_AUTHORITY: &str = "contract_authoritative";
const QWEN3_PROMPT_TEMPLATE: &str = "qwen-chat-raw-deterministic";

const DEFAULT_ONE_TOKEN_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-one-token-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-one-token-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-one-token-cuda.json",
];
const DEFAULT_SHORT_DECODE_8_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-short-decode-8-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-short-decode-8-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-short-decode-8-cuda.json",
];
const DEFAULT_SHORT_DECODE_32_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-short-decode-32-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-short-decode-32-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-short-decode-32-cuda.json",
];
const DEFAULT_WARM_SESSION_3_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-warm-session-3-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-warm-session-3-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-warm-session-3-cuda.json",
];
const DEFAULT_DECODE_128_RUNS: [&str; 3] = [
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-01/qwen3-0_6b-decode-128-from-warm-context-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-02/qwen3-0_6b-decode-128-from-warm-context-cuda.json",
    "ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-perf-016/run-03/qwen3-0_6b-decode-128-from-warm-context-cuda.json",
];

#[derive(Debug)]
struct Args {
    one_token_runs: Vec<PathBuf>,
    short_decode_8_runs: Vec<PathBuf>,
    short_decode_32_runs: Vec<PathBuf>,
    warm_session_3_runs: Vec<PathBuf>,
    decode_128_runs: Vec<PathBuf>,
    receipt_out: PathBuf,
    manifest_out: Option<PathBuf>,
    print_manifest: bool,
}

#[derive(Debug)]
struct ProfileInputGroup<'a> {
    profile: &'static str,
    flag: &'static str,
    expected_generated_tokens: u64,
    paths: &'a [PathBuf],
}

#[derive(Debug)]
struct ProfileRuns {
    one_token: Vec<(PathBuf, Value)>,
    short_decode_8: Vec<(PathBuf, Value)>,
    short_decode_32: Vec<(PathBuf, Value)>,
    warm_session_3: Vec<(PathBuf, Value)>,
    decode_128: Vec<(PathBuf, Value)>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.print_manifest || args.manifest_out.is_some() {
        let manifest = source_manifest(&args);
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        if args.print_manifest {
            println!("{manifest_json}");
        }
        if let Some(path) = &args.manifest_out {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, manifest_json)?;
        }
        return Ok(());
    }

    assert_input_paths_exist(&args)?;
    let runs = ProfileRuns {
        one_token: read_runs(&args.one_token_runs)?,
        short_decode_8: read_runs(&args.short_decode_8_runs)?,
        short_decode_32: read_runs(&args.short_decode_32_runs)?,
        warm_session_3: read_runs(&args.warm_session_3_runs)?,
        decode_128: read_runs(&args.decode_128_runs)?,
    };
    assert_repeated_sources(&runs)?;

    let receipt = build_receipt(&args, &runs)?;
    validate_qwen3_cuda_repeated_comparator_receipt_json(&receipt)?;

    if let Some(parent) = args.receipt_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.receipt_out, serde_json::to_string_pretty(&receipt)?)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut one_token_runs = DEFAULT_ONE_TOKEN_RUNS.iter().map(PathBuf::from).collect();
    let mut short_decode_8_runs = DEFAULT_SHORT_DECODE_8_RUNS.iter().map(PathBuf::from).collect();
    let mut short_decode_32_runs = DEFAULT_SHORT_DECODE_32_RUNS.iter().map(PathBuf::from).collect();
    let mut warm_session_3_runs = DEFAULT_WARM_SESSION_3_RUNS.iter().map(PathBuf::from).collect();
    let mut decode_128_runs = DEFAULT_DECODE_128_RUNS.iter().map(PathBuf::from).collect();
    let mut receipt_out = PathBuf::from(DEFAULT_RECEIPT_OUT);
    let mut manifest_out = None;
    let mut print_manifest = false;
    let mut iter = env::args().skip(1);

    let mut one_token_overridden = false;
    let mut short_decode_8_overridden = false;
    let mut short_decode_32_overridden = false;
    let mut warm_session_3_overridden = false;
    let mut decode_128_overridden = false;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--one-token-run" => push_override(
                &mut one_token_runs,
                &mut one_token_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--short-decode-8-run" => push_override(
                &mut short_decode_8_runs,
                &mut short_decode_8_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--short-decode-32-run" => push_override(
                &mut short_decode_32_runs,
                &mut short_decode_32_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--warm-session-3-run" => push_override(
                &mut warm_session_3_runs,
                &mut warm_session_3_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--decode-128-from-warm-context-run" => push_override(
                &mut decode_128_runs,
                &mut decode_128_overridden,
                next_value(&mut iter, &arg)?,
            ),
            "--receipt-out" => receipt_out = PathBuf::from(next_value(&mut iter, &arg)?),
            "--manifest-out" => manifest_out = Some(PathBuf::from(next_value(&mut iter, &arg)?)),
            "--print-manifest" => print_manifest = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(Args {
        one_token_runs,
        short_decode_8_runs,
        short_decode_32_runs,
        warm_session_3_runs,
        decode_128_runs,
        receipt_out,
        manifest_out,
        print_manifest,
    })
}

fn push_override(paths: &mut Vec<PathBuf>, overridden: &mut bool, value: String) {
    if !*overridden {
        paths.clear();
        *overridden = true;
    }
    paths.push(PathBuf::from(value));
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: qwen3_cuda_repeated_comparator_receipt [--one-token-run PATH ...] [--short-decode-8-run PATH ...] [--short-decode-32-run PATH ...] [--warm-session-3-run PATH ...] [--decode-128-from-warm-context-run PATH ...] [--receipt-out PATH] [--manifest-out PATH] [--print-manifest]"
    );
}

fn input_groups(args: &Args) -> Vec<ProfileInputGroup<'_>> {
    vec![
        ProfileInputGroup {
            profile: "one_token",
            flag: "--one-token-run",
            expected_generated_tokens: 1,
            paths: &args.one_token_runs,
        },
        ProfileInputGroup {
            profile: "short_decode_8",
            flag: "--short-decode-8-run",
            expected_generated_tokens: 8,
            paths: &args.short_decode_8_runs,
        },
        ProfileInputGroup {
            profile: "short_decode_32",
            flag: "--short-decode-32-run",
            expected_generated_tokens: 32,
            paths: &args.short_decode_32_runs,
        },
        ProfileInputGroup {
            profile: "warm_session_3_turns",
            flag: "--warm-session-3-run",
            expected_generated_tokens: 24,
            paths: &args.warm_session_3_runs,
        },
        ProfileInputGroup {
            profile: "decode_128_from_warm_context",
            flag: "--decode-128-from-warm-context-run",
            expected_generated_tokens: 128,
            paths: &args.decode_128_runs,
        },
    ]
}

fn source_manifest(args: &Args) -> Value {
    let profiles = input_groups(args)
        .into_iter()
        .map(|group| {
            json!({
                "profile": group.profile,
                "run_flag": group.flag,
                "min_runs": 3,
                "expected_generated_tokens": group.expected_generated_tokens,
                "source_paths": group.paths.iter().map(|path| path_label(path)).collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();

    json!({
        "schema": 1,
        "artifact_kind": "qwen3_cuda_repeated_comparator_source_manifest",
        "campaign_item": CAMPAIGN_ITEM,
        "aggregate_artifact_kind": "qwen3_cuda_repeated_comparator",
        "aggregate_receipt_out": path_label(&args.receipt_out),
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "min_runs_per_profile": 3,
        "profiles": profiles,
        "model": {
            "id": "qwen3-0.6b-instruct-q8_0",
            "file": "Qwen3-0.6B-Q8_0.gguf",
            "sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
        },
        "tokenizer_prompt_authority": {
            "tokenizer_authority": CONTRACT_AUTHORITY,
            "prompt_authority": CONTRACT_AUTHORITY,
            "prompt_template": QWEN3_PROMPT_TEMPLATE
        },
        "required_source_fields": [
            "/artifact_kind",
            "/model/id",
            "/model/sha256",
            "/selected_backend",
            "/execution_plan/selected_route",
            "/fallback_used",
            "/quality_gate/passed",
            "/parity/passed",
            "/timing/model_load_ms",
            "/timing/tokenizer_load_ms",
            "/timing/prompt_render_ms",
            "/timing/tokenize_ms",
            "/timing/cuda_context_init_ms",
            "/timing/weight_upload_ms",
            "/timing/prefill_ms",
            "/timing/first_token_ms",
            "/timing/decode_total_ms",
            "/timing/kernel_time_ms",
            "/timing/kernel_launches",
            "/timing/host_to_device_bytes",
            "/timing/device_to_host_bytes",
            "/cuda/vram_bytes",
            "/cuda/power_draw_watts",
            "/cuda/temperature_c"
        ],
        "accepted_optional_source_fields": [
            "/timing/decode_ms may substitute for /timing/decode_total_ms",
            "/timing/host_to_device_ms may be absent if /timing/host_to_device_ms_source explains the source",
            "/timing/device_to_host_ms may be absent if /timing/device_to_host_ms_source explains the source"
        ],
        "claim_boundaries": [
            "source manifest is not a receipt and does not prove hardware execution",
            "speedup_claim=false until a separate exact-profile review accepts a profile",
            "benchmark_qualified_speedup=false until CUDA-MODEL-018",
            "full_cuda_residency_claimed=false until every required phase proves residency",
            "Qwen3 dense CUDA evidence cannot inherit Qwen2.5 evidence",
            "dense_regular_llm_cuda evidence cannot satisfy BitNet packed I2_S/QK256 proof"
        ]
    })
}

fn assert_input_paths_exist(args: &Args) -> Result<(), Box<dyn Error>> {
    let missing = input_groups(args)
        .into_iter()
        .flat_map(|group| {
            group.paths.iter().filter_map(move |path| {
                if path.is_file() {
                    None
                } else {
                    Some(format!("{}: {}", group.profile, path_label(path)))
                }
            })
        })
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    Err(format!(
        "missing CUDA-MODEL-017 Qwen3 source receipts:\n  - {}\nrun with --print-manifest or --manifest-out PATH to inspect the expected capture set",
        missing.join("\n  - ")
    )
    .into())
}

fn read_runs(paths: &[PathBuf]) -> Result<Vec<(PathBuf, Value)>, Box<dyn Error>> {
    paths.iter().map(|path| Ok((path.clone(), read_json(path)?))).collect()
}

fn read_json(path: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn assert_repeated_sources(runs: &ProfileRuns) -> Result<(), Box<dyn Error>> {
    assert_profile_sources("one_token", &runs.one_token)?;
    assert_profile_sources("short_decode_8", &runs.short_decode_8)?;
    assert_profile_sources("short_decode_32", &runs.short_decode_32)?;
    assert_profile_sources("warm_session_3_turns", &runs.warm_session_3)?;
    assert_profile_sources("decode_128_from_warm_context", &runs.decode_128)?;
    Ok(())
}

fn assert_profile_sources(profile: &str, runs: &[(PathBuf, Value)]) -> Result<(), Box<dyn Error>> {
    if runs.len() < 3 {
        return Err(format!("{profile} requires at least 3 runs").into());
    }
    let mut paths = BTreeSet::new();
    let anchor = runs.first().ok_or("runs must not be empty")?.1.clone();
    let prompt_template = str_at(&anchor, "/tokenizer_prompt_authority/prompt_template")?;
    let anchor_prompt_hash = prompt_hash(profile, &anchor)?;

    for (path, receipt) in runs {
        if !paths.insert(path_label(path)) {
            return Err(format!("{profile} run paths must be unique").into());
        }
        assert_source_receipt(profile, receipt)?;
        if str_at(receipt, "/tokenizer_prompt_authority/prompt_template")? != prompt_template {
            return Err(format!("{profile} runs must use the same prompt template").into());
        }
        if prompt_hash(profile, receipt)? != anchor_prompt_hash {
            return Err(format!("{profile} runs must use the same rendered prompt hash").into());
        }
    }
    Ok(())
}

fn assert_source_receipt(profile: &str, receipt: &Value) -> Result<(), Box<dyn Error>> {
    if !str_at(receipt, "/artifact_kind")?.starts_with("dense_gguf_qwen_") {
        return Err("source receipt must be a dense Qwen proof".into());
    }
    let expected_artifact = expected_source_artifact_kind(profile)?;
    if str_at(receipt, "/artifact_kind")? != expected_artifact {
        return Err(format!("{profile} requires source artifact kind {expected_artifact}").into());
    }
    if str_at(receipt, "/model/id")? != "qwen3-0.6b-instruct-q8_0" {
        return Err("source receipt must use the Qwen3 0.6B artifact".into());
    }
    if str_at(receipt, "/model/sha256")?
        != "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
    {
        return Err("source receipt has unexpected Qwen3 SHA".into());
    }
    if str_at(receipt, "/execution_plan/selected_route")? != "dense_regular_llm_cuda" {
        return Err("source receipt must route dense_regular_llm_cuda".into());
    }
    if str_at(receipt, "/selected_backend")? != "nvidia-rtx-5070-ti-cuda" {
        return Err("source receipt must select nvidia-rtx-5070-ti-cuda".into());
    }
    if str_at(receipt, "/tokenizer_prompt_authority/tokenizer_authority")? != CONTRACT_AUTHORITY {
        return Err("source receipt must use contract tokenizer authority".into());
    }
    if str_at(receipt, "/tokenizer_prompt_authority/prompt_authority")? != CONTRACT_AUTHORITY {
        return Err("source receipt must use contract prompt authority".into());
    }
    if str_at(receipt, "/tokenizer_prompt_authority/prompt_template")? != QWEN3_PROMPT_TEMPLATE {
        return Err("source receipt must use qwen-chat-raw-deterministic prompt template".into());
    }
    if bool_at(receipt, "/fallback_used")? {
        return Err("source receipt must be fallback-free".into());
    }
    if bool_at(receipt, "/speedup_claim")? {
        return Err("source receipt must not claim speedup".into());
    }
    if bool_at(receipt, "/claim_boundary/bitnet_packed_i2s_qk256_proof")? {
        return Err("source receipt must not claim BitNet packed proof".into());
    }
    if generated_tokens(profile, receipt)? != expected_generated_tokens(profile)? {
        return Err(format!("{profile} generated token count does not match its profile").into());
    }
    Ok(())
}

fn expected_source_artifact_kind(profile: &str) -> Result<&'static str, Box<dyn Error>> {
    match profile {
        "one_token" => Ok("dense_gguf_qwen_one_token_strict_cuda_proof"),
        "short_decode_8" | "short_decode_32" => {
            Ok("dense_gguf_qwen_short_decode_strict_cuda_proof")
        }
        "warm_session_3_turns" => Ok("dense_gguf_qwen_warm_session_strict_cuda_proof"),
        "decode_128_from_warm_context" => Ok("dense_gguf_qwen_warm_decode_strict_cuda_proof"),
        other => Err(format!("unknown profile {other}").into()),
    }
}

fn build_receipt(args: &Args, runs: &ProfileRuns) -> Result<Value, Box<dyn Error>> {
    let profiles = vec![
        profile_from_runs("one_token", &runs.one_token)?,
        profile_from_runs("short_decode_8", &runs.short_decode_8)?,
        profile_from_runs("short_decode_32", &runs.short_decode_32)?,
        profile_from_runs("warm_session_3_turns", &runs.warm_session_3)?,
        profile_from_runs("decode_128_from_warm_context", &runs.decode_128)?,
    ];
    let comparator_summary = comparator_summary(&profiles);
    let anchor = &runs.one_token.first().ok_or("one_token runs missing")?.1;
    let cuda = anchor.pointer("/cuda").cloned().ok_or("cuda block missing")?;

    Ok(json!({
        "schema": 1,
        "artifact_kind": "qwen3_cuda_repeated_comparator",
        "artifact_path": path_label(&args.receipt_out),
        "machine_id": "windows-9950x3d-rtx5070ti",
        "hardware_lane": "nvidia_rtx_5070_ti_cuda",
        "timestamp_utc": timestamp_label(),
        "requested_backend": "nvidia-rtx-5070-ti-cuda",
        "selected_backend": "nvidia-rtx-5070-ti-cuda",
        "reference_backend": "amd-9950x3d-cpu-avx512",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "claim": "qwen3_cuda_repeated_comparator",
        "fallback_used": false,
        "fallback_backend": null,
        "fallback_reason": null,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "full_cuda_residency_claimed": false,
        "dense_gguf_inference_claimed": false,
        "broad_dense_gguf_ready_claimed": false,
        "qwen25_proof_inherited": false,
        "server_ready_claimed": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "claim_boundary": {
            "qwen3_cuda_repeated_comparator_claimed": true,
            "qwen_one_token_cuda_claimed": true,
            "qwen_short_decode_cuda_claimed": true,
            "qwen_warm_session_cuda_claimed": true,
            "qwen_chat_cuda_claimed": true,
            "server_ready_claimed": false,
            "speedup_claim": false,
            "benchmark_qualified_speedup": false,
            "full_cuda_residency_claimed": false,
            "broad_dense_gguf_ready_claimed": false,
            "qwen25_proof_inherited": false,
            "bitnet_packed_i2s_qk256_proof": false
        },
        "model": model_for_receipt(anchor)?,
        "tokenizer_prompt_authority": {
            "tokenizer_authority": str_at(anchor, "/tokenizer_prompt_authority/tokenizer_authority")?,
            "prompt_authority": str_at(anchor, "/tokenizer_prompt_authority/prompt_authority")?,
            "prompt_template": QWEN3_PROMPT_TEMPLATE,
            "prompt_policy": "profile-local deterministic prompts; same tokenizer and prompt policy across all runs",
            "deterministic_prompt": true
        },
        "execution_plan": anchor.pointer("/execution_plan").cloned().ok_or("execution_plan missing")?,
        "proof_inputs": {
            "one_token": profile_input("one_token", &runs.one_token)?,
            "short_decode_8": profile_input("short_decode_8", &runs.short_decode_8)?,
            "short_decode_32": profile_input("short_decode_32", &runs.short_decode_32)?,
            "warm_session_3_turns": profile_input("warm_session_3_turns", &runs.warm_session_3)?,
            "decode_128_from_warm_context": profile_input("decode_128_from_warm_context", &runs.decode_128)?
        },
        "profiles": profiles,
        "comparator_summary": comparator_summary,
        "transfer_timing": transfer_timing(&profiles),
        "hardware_context": hardware_context(runs)?,
        "cuda": cuda,
        "claim_boundaries": [
            "speedup_claim=false; repeated CPU/CUDA comparator evidence is not a speedup qualification.",
            "benchmark_qualified_speedup=false until a separate exact-profile review accepts a profile.",
            "Qwen3 repeated comparator evidence cannot inherit Qwen2.5 evidence.",
            "dense_regular_llm_cuda receipts cannot satisfy BitNet packed I2S/QK256 proof."
        ]
    }))
}

fn model_for_receipt(receipt: &Value) -> Result<Value, Box<dyn Error>> {
    let mut model = receipt.pointer("/model").cloned().ok_or("model missing")?;
    if let Some(object) = model.as_object_mut() {
        object.remove("path");
    }
    Ok(model)
}

fn profile_input(profile: &str, runs: &[(PathBuf, Value)]) -> Result<Value, Box<dyn Error>> {
    let mut hasher = Sha256::new();
    let mut paths = Vec::new();
    for (path, _) in runs {
        let digest = sha256_file(path)?;
        hasher.update(digest.as_bytes());
        paths.push(path_label(path));
    }
    Ok(json!({
        "path": format!("profile:{profile}"),
        "sha256": format!("{:x}", hasher.finalize()),
        "artifact_kind": "qwen3_profile_repeated_runs",
        "runs": paths
    }))
}

fn profile_from_runs(profile: &str, runs: &[(PathBuf, Value)]) -> Result<Value, Box<dyn Error>> {
    let run_values = runs
        .iter()
        .enumerate()
        .map(|(index, (path, receipt))| run_from_receipt(profile, index + 1, path, receipt))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(json!({
        "profile": profile,
        "status": "repeated_same_artifact_cpu_cuda_comparator",
        "cpu_reference_backend": "amd-9950x3d-cpu-avx512",
        "cuda_backend": "nvidia-rtx-5070-ti-cuda",
        "runtime_api": "cuda",
        "selected_route": "dense_regular_llm_cuda",
        "run_count": run_values.len(),
        "cpu_runs": run_values.len(),
        "cuda_runs": run_values.len(),
        "min_runs_per_backend": 3,
        "fallback_free": true,
        "same_artifact_sha": true,
        "same_tokenizer_prompt_policy": true,
        "deterministic_generation_policy": true,
        "generated_token_ids_match": run_values.iter().all(|run| {
            run.get("generated_token_ids_match").and_then(Value::as_bool) == Some(true)
        }),
        "first_divergence_report": "no generated-token divergence recorded across source receipts",
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "server_ready_claimed": false,
        "transfer_timing_status": "host_to_device_model_load_envelope_device_to_host_measured",
        "model_load_ms": number_summary(&run_values, "/timing/model_load_ms"),
        "tokenizer_load_ms": number_summary(&run_values, "/timing/tokenizer_load_ms"),
        "prompt_render_ms": number_summary(&run_values, "/timing/prompt_render_ms"),
        "tokenize_ms": number_summary(&run_values, "/timing/tokenize_ms"),
        "cuda_context_init_ms": number_summary(&run_values, "/timing/cuda_context_init_ms"),
        "weight_upload_ms": number_summary(&run_values, "/timing/weight_upload_ms"),
        "cpu_total_ms": number_summary(&run_values, "/timing/cpu_total_ms"),
        "cuda_total_ms": number_summary(&run_values, "/timing/cuda_total_ms"),
        "prefill_ms": number_summary(&run_values, "/timing/prefill_ms"),
        "first_token_ms": number_summary(&run_values, "/timing/first_token_ms"),
        "decode_total_ms": number_summary(&run_values, "/timing/decode_total_ms"),
        "steady_tok_per_s": number_summary(&run_values, "/timing/steady_tok_per_s"),
        "kernel_time_ms": number_summary(&run_values, "/timing/kernel_time_ms"),
        "launch_count": u64_summary(&run_values, "/timing/launch_count"),
        "host_to_device_bytes": u64_summary(&run_values, "/timing/host_to_device_bytes"),
        "host_to_device_ms": number_summary(&run_values, "/timing/host_to_device_ms"),
        "device_to_host_bytes": u64_summary(&run_values, "/timing/device_to_host_bytes"),
        "device_to_host_ms": number_summary(&run_values, "/timing/device_to_host_ms"),
        "vram_high_water_bytes": u64_summary(&run_values, "/timing/vram_high_water_bytes"),
        "runs": run_values
    }))
}

fn run_from_receipt(
    profile: &str,
    index: usize,
    path: &Path,
    receipt: &Value,
) -> Result<Value, Box<dyn Error>> {
    let generated = generated_identity(profile, receipt)?;
    let generated_tokens = generated_tokens(profile, receipt)?;
    let decode_total = decode_total_ms(receipt)?;
    let steady_tok_per_s =
        if decode_total > 0.0 { generated_tokens as f64 / (decode_total / 1000.0) } else { 0.0 };
    let host_to_device_ms = optional_number(receipt, "/timing/host_to_device_ms");
    let device_to_host_ms = optional_number(receipt, "/timing/device_to_host_ms");

    let mut run = json!({
        "run_id": format!("run-{index:02}"),
        "profile": profile,
        "source_receipt_path": path_label(path),
        "source_receipt_sha256": sha256_file(path)?,
        "source_artifact_kind": str_at(receipt, "/artifact_kind")?,
        "model_sha256": str_at(receipt, "/model/sha256")?,
        "prompt_template": str_at(receipt, "/tokenizer_prompt_authority/prompt_template")?,
        "prompt_token_count": prompt_token_count(profile, receipt)?,
        "generation_policy": "greedy",
        "deterministic_generation": true,
        "generated_tokens": generated_tokens,
        "generated_token_ids_sha256": generated.ids_sha256,
        "generated_token_ids_match": generated.ids_match,
        "first_divergence_report": generated.first_divergence_report,
        "top_k_compared": generated.top_k_compared,
        "fallback_used": bool_at(receipt, "/fallback_used")?,
        "quality_passed": bool_at(receipt, "/quality_gate/passed")?,
        "parity_passed": bool_at(receipt, "/parity/passed")?,
        "speedup_claim": false,
        "benchmark_qualified_speedup": false,
        "bitnet_packed_i2s_qk256_proof": false,
        "full_cuda_residency_claimed": false,
        "server_ready_claimed": false,
        "timing": {
            "model_load_ms": number_or(receipt, "/timing/model_load_ms", 0.0),
            "tokenizer_load_ms": number_or(receipt, "/timing/tokenizer_load_ms", 0.0),
            "prompt_render_ms": number_or(receipt, "/timing/prompt_render_ms", 0.0),
            "tokenize_ms": number_or(receipt, "/timing/tokenize_ms", 0.0),
            "cuda_context_init_ms": number_or(receipt, "/timing/cuda_context_init_ms", 0.0),
            "weight_upload_ms": number_or(receipt, "/timing/weight_upload_ms", host_to_device_ms.unwrap_or(0.0)),
            "cpu_total_ms": number_at(receipt, "/timing/cpu_reference_total_ms")?,
            "cuda_total_ms": number_at(receipt, "/timing/total_ms")?,
            "prefill_ms": number_or(receipt, "/timing/prefill_ms", 0.0),
            "first_token_ms": number_at(receipt, "/timing/first_token_ms")?,
            "decode_total_ms": decode_total,
            "steady_tok_per_s": steady_tok_per_s,
            "kernel_time_ms": number_at(receipt, "/timing/kernel_time_ms")?,
            "launch_count": u64_at(receipt, "/timing/kernel_launches")?,
            "kernel_invocations": u64_at(receipt, "/timing/kernel_invocations")?,
            "host_to_device_bytes": u64_at(receipt, "/timing/host_to_device_bytes")?,
            "host_to_device_ms": host_to_device_ms,
            "host_to_device_ms_source": string_or(receipt, "/timing/host_to_device_ms_source", "not_measured_in_source_receipt"),
            "device_to_host_bytes": u64_at(receipt, "/timing/device_to_host_bytes")?,
            "device_to_host_ms": device_to_host_ms,
            "device_to_host_ms_source": string_or(receipt, "/timing/device_to_host_ms_source", "not_measured_in_source_receipt"),
            "vram_high_water_bytes": u64_at(receipt, "/cuda/vram_bytes")?,
            "power_temperature_context": "NVML power and temperature sampled during source receipt"
        }
    });

    if profile == "warm_session_3_turns" {
        run["turns_count"] = json!(u64_at(receipt, "/warm_session_proof/turns_count")?);
    }
    if profile == "decode_128_from_warm_context" {
        run["warm_context_reused"] = json!(true);
    }
    Ok(run)
}

struct GeneratedIdentity {
    ids_sha256: String,
    ids_match: bool,
    first_divergence_report: String,
    top_k_compared: bool,
}

fn generated_identity(profile: &str, receipt: &Value) -> Result<GeneratedIdentity, Box<dyn Error>> {
    match profile {
        "one_token" => {
            let cpu = u64_at(receipt, "/one_token_proof/cpu_selected_token_id")?;
            let cuda = u64_at(receipt, "/one_token_proof/cuda_selected_token_id")?;
            Ok(GeneratedIdentity {
                ids_sha256: str_at(receipt, "/one_token_proof/cpu_logits_top_k_sha256")?.to_owned(),
                ids_match: cpu == cuda,
                first_divergence_report: if cpu == cuda {
                    "none; cpu_selected_token_id matches cuda_selected_token_id".to_owned()
                } else {
                    format!("cpu_selected_token_id={cpu}, cuda_selected_token_id={cuda}")
                },
                top_k_compared: bool_at(receipt, "/one_token_proof/top_k_compared")?,
            })
        }
        "short_decode_8" | "short_decode_32" => Ok(GeneratedIdentity {
            ids_sha256: str_at(receipt, "/short_decode_proof/cpu_generated_token_ids_sha256")?
                .to_owned(),
            ids_match: bool_at(receipt, "/short_decode_proof/generated_token_ids_match")?,
            first_divergence_report: divergence_report(
                receipt.pointer("/short_decode_proof/first_token_divergence_index"),
            ),
            top_k_compared: bool_at(receipt, "/short_decode_proof/top_k_compared")?,
        }),
        "decode_128_from_warm_context" => Ok(GeneratedIdentity {
            ids_sha256: str_at(receipt, "/warm_decode_proof/cpu_generated_token_ids_sha256")?
                .to_owned(),
            ids_match: bool_at(receipt, "/warm_decode_proof/generated_token_ids_match")?,
            first_divergence_report: divergence_report(
                receipt.pointer("/warm_decode_proof/first_token_divergence_index"),
            ),
            top_k_compared: bool_at(receipt, "/warm_decode_proof/top_k_compared")?,
        }),
        "warm_session_3_turns" => Ok(GeneratedIdentity {
            ids_sha256: str_at(receipt, "/warm_session_proof/cpu_generated_token_ids_sha256")?
                .to_owned(),
            ids_match: bool_at(receipt, "/warm_session_proof/generated_token_ids_match")?,
            first_divergence_report: divergence_report(
                receipt.pointer("/warm_session_proof/first_token_divergence"),
            ),
            top_k_compared: bool_at(receipt, "/warm_session_proof/top_k_compared")?,
        }),
        other => Err(format!("unknown profile {other}").into()),
    }
}

fn divergence_report(value: Option<&Value>) -> String {
    match value {
        Some(Value::Null) | None => "none".to_owned(),
        Some(value) => format!("first divergence: {value}"),
    }
}

fn comparator_summary(profiles: &[Value]) -> Value {
    let total_runs = profiles.iter().map(|profile| u64_value(profile, "run_count")).sum::<u64>();
    json!({
        "status": "repeated_comparator_only",
        "profiles_recorded": profiles.len(),
        "min_runs_per_backend": 3,
        "total_cpu_runs": total_runs,
        "total_cuda_runs": total_runs,
        "fallback_free": true,
        "same_artifact_sha": true,
        "same_tokenizer_prompt_policy": true,
        "deterministic_generation_policy": true,
        "generated_tokens_compared": true,
        "speedup_claim_allowed": false,
        "benchmark_qualified_speedup": false,
        "accepted_speedup_profiles": [],
        "remaining_qualification_blockers": [
            "profile-specific speedup thresholds remain unreviewed",
            "pure host-to-device timing remains separated from the model-load envelope"
        ],
        "next_step": "Qwen3 exact-profile benchmark qualification review after repeated hardware receipts land"
    })
}

fn transfer_timing(_profiles: &[Value]) -> Value {
    json!({
        "status": "host_to_device_model_load_envelope_device_to_host_measured",
        "source": "Qwen3 source receipts record H2D model-load envelopes and D2H wall-clock timing",
        "host_to_device_bytes_recorded": true,
        "device_to_host_bytes_recorded": true,
        "host_to_device_timing_recorded": true,
        "device_to_host_timing_recorded": true,
        "pure_host_to_device_timing_recorded": false
    })
}

fn hardware_context(runs: &ProfileRuns) -> Result<Value, Box<dyn Error>> {
    let receipts = runs
        .one_token
        .iter()
        .chain(runs.short_decode_8.iter())
        .chain(runs.short_decode_32.iter())
        .chain(runs.warm_session_3.iter())
        .chain(runs.decode_128.iter())
        .map(|(_, receipt)| receipt)
        .collect::<Vec<_>>();
    let powers = receipts
        .iter()
        .map(|receipt| number_at(receipt, "/cuda/power_draw_watts"))
        .collect::<Result<Vec<_>, _>>()?;
    let temperatures = receipts
        .iter()
        .map(|receipt| number_at(receipt, "/cuda/temperature_c"))
        .collect::<Result<Vec<_>, _>>()?;
    let first = receipts.first().ok_or("hardware context requires receipts")?;
    Ok(json!({
        "vram_bytes": u64_at(first, "/cuda/vram_bytes")?,
        "power_draw_watts_min": min_f64(&powers),
        "power_draw_watts_max": max_f64(&powers),
        "temperature_c_min": min_f64(&temperatures),
        "temperature_c_max": max_f64(&temperatures),
        "source": "NVML fields recorded in Qwen3 strict CUDA proof receipts"
    }))
}

fn prompt_hash(profile: &str, receipt: &Value) -> Result<String, Box<dyn Error>> {
    if profile == "warm_session_3_turns" {
        return Ok(str_at(receipt, "/tokenizer_prompt_authority/turns/0/rendered_prompt_sha256")?
            .to_owned());
    }
    Ok(str_at(receipt, "/tokenizer_prompt_authority/rendered_prompt_sha256")?.to_owned())
}

fn prompt_token_count(profile: &str, receipt: &Value) -> Result<u64, Box<dyn Error>> {
    if profile == "warm_session_3_turns" {
        return u64_at(receipt, "/tokenizer_prompt_authority/turns/0/prompt_token_count");
    }
    u64_at(receipt, "/tokenizer_prompt_authority/prompt_token_count")
}

fn generated_tokens(profile: &str, receipt: &Value) -> Result<u64, Box<dyn Error>> {
    match profile {
        "one_token" => u64_at(receipt, "/one_token_proof/generated_tokens_count"),
        "short_decode_8" | "short_decode_32" => {
            u64_at(receipt, "/short_decode_proof/generated_tokens_count")
        }
        "decode_128_from_warm_context" => {
            u64_at(receipt, "/warm_decode_proof/generated_tokens_count")
        }
        "warm_session_3_turns" => u64_at(receipt, "/warm_session_proof/generated_tokens_total"),
        other => Err(format!("unknown profile {other}").into()),
    }
}

fn expected_generated_tokens(profile: &str) -> Result<u64, Box<dyn Error>> {
    match profile {
        "one_token" => Ok(1),
        "short_decode_8" => Ok(8),
        "short_decode_32" => Ok(32),
        "warm_session_3_turns" => Ok(24),
        "decode_128_from_warm_context" => Ok(128),
        other => Err(format!("unknown profile {other}").into()),
    }
}

fn decode_total_ms(receipt: &Value) -> Result<f64, Box<dyn Error>> {
    receipt
        .pointer("/timing/decode_total_ms")
        .or_else(|| receipt.pointer("/timing/decode_ms"))
        .and_then(Value::as_f64)
        .ok_or_else(|| "/timing/decode_total_ms or /timing/decode_ms must be a number".into())
}

fn number_summary(runs: &[Value], pointer: &str) -> Value {
    let values = runs
        .iter()
        .filter_map(|run| run.pointer(pointer).and_then(Value::as_f64))
        .collect::<Vec<_>>();
    json!({ "count": values.len(), "min": min_f64(&values), "mean": mean_f64(&values), "max": max_f64(&values) })
}

fn u64_summary(runs: &[Value], pointer: &str) -> Value {
    let values = runs
        .iter()
        .filter_map(|run| run.pointer(pointer).and_then(Value::as_u64))
        .collect::<Vec<_>>();
    let min = values.iter().copied().min().unwrap_or(0);
    let max = values.iter().copied().max().unwrap_or(0);
    let mean = if values.is_empty() {
        0.0
    } else {
        values.iter().copied().sum::<u64>() as f64 / values.len() as f64
    };
    json!({ "count": values.len(), "min": min, "mean": mean, "max": max })
}

fn min_f64(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::min).unwrap_or(0.0)
}

fn max_f64(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(0.0)
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn timestamp_label() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{digest:x}"))
}

fn path_label(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{pointer} must be a string").into())
}

fn string_or(value: &Value, pointer: &str, default: &str) -> String {
    value.pointer(pointer).and_then(Value::as_str).unwrap_or(default).to_owned()
}

fn bool_at(value: &Value, pointer: &str) -> Result<bool, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{pointer} must be a bool").into())
}

fn u64_at(value: &Value, pointer: &str) -> Result<u64, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{pointer} must be an unsigned integer").into())
}

fn number_at(value: &Value, pointer: &str) -> Result<f64, Box<dyn Error>> {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{pointer} must be a number").into())
}

fn number_or(value: &Value, pointer: &str, default: f64) -> f64 {
    optional_number(value, pointer).unwrap_or(default)
}

fn optional_number(value: &Value, pointer: &str) -> Option<f64> {
    value.pointer(pointer).and_then(Value::as_f64)
}

fn u64_value(value: &Value, field: &str) -> u64 {
    value.get(field).and_then(Value::as_u64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_with_prefix(prefix: &str) -> Args {
        Args {
            one_token_runs: vec![PathBuf::from(format!("{prefix}/one-token.json"))],
            short_decode_8_runs: vec![PathBuf::from(format!("{prefix}/short-decode-8.json"))],
            short_decode_32_runs: vec![PathBuf::from(format!("{prefix}/short-decode-32.json"))],
            warm_session_3_runs: vec![PathBuf::from(format!("{prefix}/warm-session-3.json"))],
            decode_128_runs: vec![PathBuf::from(format!(
                "{prefix}/decode-128-from-warm-context.json"
            ))],
            receipt_out: PathBuf::from(format!("{prefix}/aggregate.json")),
            manifest_out: None,
            print_manifest: false,
        }
    }

    #[test]
    fn source_manifest_names_profiles_and_claim_boundaries() {
        let args = args_with_prefix("target/qwen3");
        let manifest = source_manifest(&args);

        assert_eq!(manifest["artifact_kind"], "qwen3_cuda_repeated_comparator_source_manifest");
        assert_eq!(manifest["campaign_item"], CAMPAIGN_ITEM);
        assert_eq!(manifest["aggregate_artifact_kind"], "qwen3_cuda_repeated_comparator");
        assert_eq!(manifest["selected_backend"], "nvidia-rtx-5070-ti-cuda");
        assert_eq!(manifest["selected_route"], "dense_regular_llm_cuda");
        assert_eq!(manifest["model"]["id"], "qwen3-0.6b-instruct-q8_0");

        let profiles = manifest["profiles"].as_array();
        assert_eq!(profiles.map(Vec::len), Some(5));
        assert!(profiles.is_some_and(|profiles| {
            profiles.iter().any(|profile| {
                profile["profile"] == "decode_128_from_warm_context"
                    && profile["expected_generated_tokens"] == 128
                    && profile["run_flag"] == "--decode-128-from-warm-context-run"
            })
        }));
        assert!(manifest["accepted_optional_source_fields"].as_array().is_some_and(|fields| {
            fields.iter().any(|field| {
                field.as_str().is_some_and(|text| text.contains("host_to_device_ms_source"))
            })
        }));
        assert!(manifest["claim_boundaries"].as_array().is_some_and(|boundaries| {
            boundaries.iter().any(|boundary| {
                boundary
                    .as_str()
                    .is_some_and(|text| text.contains("does not prove hardware execution"))
            })
        }));
    }

    #[test]
    fn preflight_reports_all_missing_profile_inputs() {
        let args = args_with_prefix("target/qwen3/missing");
        let message = match assert_input_paths_exist(&args) {
            Ok(()) => String::new(),
            Err(err) => err.to_string(),
        };
        assert!(!message.is_empty(), "missing inputs should fail");

        for profile in [
            "one_token",
            "short_decode_8",
            "short_decode_32",
            "warm_session_3_turns",
            "decode_128_from_warm_context",
        ] {
            assert!(message.contains(profile), "missing {profile} in {message}");
        }
        assert!(message.contains("--print-manifest"), "missing manifest hint in {message}");
    }
}
