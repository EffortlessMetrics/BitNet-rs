use bitnet_bench_receipts::validate_strict_cpu_benchmark_receipt_json;
use serde_json::{Value, json};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const PROFILE_NAMES: [&str; 5] = ["micro", "layer", "prefill", "first_token", "decode"];

#[derive(Debug, Default)]
struct Args {
    strict_proof_receipts: Vec<PathBuf>,
    receipt_out: Option<PathBuf>,
    machine_id: Option<String>,
    hardware_lane: Option<String>,
    selected_backend: Option<String>,
    model_quant_format: Option<String>,
    platform_artifact: Option<PathBuf>,
    power_mode: Option<String>,
    temperature_c: Option<f64>,
    frequency_mhz: Option<f64>,
}

#[derive(Debug)]
struct PhaseMeasurement<'a> {
    profile_name: &'a str,
    requested_kernel: &'a str,
    selected_kernel: &'a str,
    rows: u64,
    cols: u64,
    iterations: u64,
    wall_time_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    token_count: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let receipt = build_receipt(&args)?;
    validate_strict_cpu_benchmark_receipt_json(&receipt)?;

    let json = serde_json::to_string_pretty(&receipt)?;
    if let Some(path) = args.receipt_out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json)?;
    } else {
        println!("{json}");
    }

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--strict-proof-receipt" => {
                args.strict_proof_receipts.push(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--receipt-out" => args.receipt_out = Some(PathBuf::from(next_value(&mut iter, &arg)?)),
            "--machine-id" => args.machine_id = Some(next_value(&mut iter, &arg)?),
            "--hardware-lane" => args.hardware_lane = Some(next_value(&mut iter, &arg)?),
            "--selected-backend" => args.selected_backend = Some(next_value(&mut iter, &arg)?),
            "--model-quant-format" => args.model_quant_format = Some(next_value(&mut iter, &arg)?),
            "--platform-artifact" => {
                args.platform_artifact = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--power-mode" => args.power_mode = Some(next_value(&mut iter, &arg)?),
            "--temperature-c" => args.temperature_c = Some(next_value(&mut iter, &arg)?.parse()?),
            "--frequency-mhz" => args.frequency_mhz = Some(next_value(&mut iter, &arg)?.parse()?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    if args.strict_proof_receipts.is_empty() {
        return Err("at least one --strict-proof-receipt is required".into());
    }

    Ok(args)
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    iter.next().ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: cpu_phase_benchmark_receipt --strict-proof-receipt PATH [--strict-proof-receipt PATH ...] [--receipt-out PATH]\n\
         Options: --machine-id, --hardware-lane, --selected-backend, --model-quant-format, --platform-artifact, --power-mode, --temperature-c, --frequency-mhz"
    );
}

fn build_receipt(args: &Args) -> Result<Value, Box<dyn Error>> {
    let proofs: Vec<Value> = args
        .strict_proof_receipts
        .iter()
        .map(|path| -> Result<Value, Box<dyn Error>> {
            Ok(serde_json::from_slice(&fs::read(path)?)?)
        })
        .collect::<Result<_, _>>()?;
    let platform_probe = args
        .platform_artifact
        .as_ref()
        .map(|path| -> Result<Value, Box<dyn Error>> {
            Ok(serde_json::from_slice(&fs::read(path)?)?)
        })
        .transpose()?;

    for proof in &proofs {
        require_bool_false(proof, "/fallback_used")?;
        require_bool_false(proof, "/execution/fallback_used")?;
    }

    let first = proofs.first().expect("strict proof receipts are required");
    let selected_backend = args
        .selected_backend
        .clone()
        .or_else(|| string_at(first, "/execution/selected_backend"))
        .unwrap_or_else(|| "cpu".to_string());
    let machine_id = args
        .machine_id
        .clone()
        .or_else(|| platform_probe.as_ref().and_then(|platform| string_at(platform, "/machine_id")))
        .unwrap_or_else(|| selected_backend.clone());
    let hardware_lane = args
        .hardware_lane
        .clone()
        .or_else(|| {
            platform_probe
                .as_ref()
                .and_then(|platform| string_at(platform, "/cpu/selected_backend"))
        })
        .unwrap_or_else(|| selected_backend.clone());
    let selected_kernel = string_at(first, "/kernel/kernel_id")
        .or_else(|| string_at(first, "/strict_provenance/selected_kernel"))
        .unwrap_or_else(|| "qk256-scalar-gemv".to_string());
    let requested_kernel = string_at(first, "/strict_provenance/requested_kernel")
        .or_else(|| string_at(first, "/kernel/kernel_id"))
        .unwrap_or_else(|| selected_kernel.clone());
    let prompt_tokens = u64_at(first, "/execution/prompt_tokens").unwrap_or(1).max(1);
    let generated_tokens = u64_at(first, "/execution/generated_tokens").unwrap_or(1).max(1);
    let model_quant_format = args
        .model_quant_format
        .clone()
        .or_else(|| string_at(first, "/model/quant_format"))
        .unwrap_or_else(|| "QK256/I2_S".to_string());
    let cpu_model = platform_probe
        .as_ref()
        .and_then(|platform| string_at(platform, "/cpu/model"))
        .or_else(|| string_at(first, "/cpu/model"))
        .unwrap_or_else(|| "unknown".to_string());
    let power_mode = args
        .power_mode
        .clone()
        .or_else(|| platform_probe.as_ref().and_then(|platform| string_at(platform, "/power/mode")))
        .unwrap_or_else(|| "unknown".to_string());
    let platform_context = args
        .platform_artifact
        .as_ref()
        .zip(platform_probe.as_ref())
        .map(|(path, platform)| build_platform_context(path, platform));

    let mut profiles = default_profiles(&requested_kernel, &selected_kernel);
    for proof in &proofs {
        apply_measured_phase(&mut profiles, proof, &requested_kernel, &selected_kernel);
    }
    let cpu258v_003_profiles =
        build_cpu258v_003_profiles(&profiles, prompt_tokens, generated_tokens);

    Ok(json!({
        "schema": 1,
        "artifact_kind": "cpu_benchmark",
        "machine_id": machine_id,
        "hardware_lane": hardware_lane,
        "timestamp_utc": timestamp_label(),
        "requested_backend": "cpu",
        "selected_backend": selected_backend,
        "runtime_api": "cpu",
        "claim": "cpu_benchmark_receipt",
        "speedup_claim": false,
        "fallback_used": false,
        "fallback_reason": null,
        "model": {
            "repo": string_at(first, "/model/repo").unwrap_or_else(|| "unknown".to_string()),
            "file": string_at(first, "/model/file").unwrap_or_else(|| "unknown".to_string()),
            "sha256": string_at(first, "/model/sha256").unwrap_or_else(|| "unknown".to_string()),
            "family": "bitnet",
            "quant_format": model_quant_format
        },
        "tokenizer": {
            "source": string_at(first, "/loader/tokenizer_source")
                .or_else(|| string_at(first, "/tokenizer/source"))
                .unwrap_or_else(|| "unknown".to_string()),
            "strict": true
        },
        "kernel": {
            "requested_kernel": requested_kernel,
            "selected_kernel": selected_kernel,
            "oracle_kernel": "qk256-scalar-gemv",
            "fallback_used": false,
            "fallback_reason": null,
            "dequantizes_before_compute": false
        },
        "cpu": {
            "model": cpu_model,
            "arch": string_at(first, "/cpu/arch").unwrap_or_else(|| env::consts::ARCH.to_string()),
            "features": array_at(first, "/cpu/features").unwrap_or_else(|| vec!["scalar".to_string()]),
            "threads": u64_at(first, "/cpu/threads")
                .or_else(|| u64_at(first, "/execution/thread_count"))
                .unwrap_or(1)
                .max(1),
            "avx512": array_at(first, "/cpu/features")
                .unwrap_or_default()
                .iter()
                .any(|feature| feature.eq_ignore_ascii_case("avx512f")),
            "power_mode": power_mode,
            "temperature_c": args.temperature_c,
            "frequency_mhz": args.frequency_mhz
        },
        "platform_context": platform_context,
        "cpu258v_003_profiles": cpu258v_003_profiles,
        "workload": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "batch_size": u64_at(first, "/execution/batch_size").unwrap_or(1).max(1)
        },
        "profiles": profiles,
        "profile_order": PROFILE_NAMES,
        "receipt_inputs": args.strict_proof_receipts
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>(),
        "artifact_path": args.receipt_out.as_ref().map(|path| path.display().to_string()),
        "claim_boundary": "Real phase profiles are measured only when backed by supplied strict CPU proof receipts; not_run profiles are explicit gaps, not performance evidence."
    }))
}

fn build_platform_context(path: &Path, platform: &Value) -> Value {
    json!({
        "artifact_path": path.display().to_string(),
        "machine_id": string_at(platform, "/machine_id").unwrap_or_else(|| "unknown".to_string()),
        "proof_stage": string_at(platform, "/proof_stage").unwrap_or_else(|| "unknown".to_string()),
        "status": string_at(platform, "/status").unwrap_or_else(|| "unknown".to_string()),
        "os": {
            "name": string_at(platform, "/os/name"),
            "version": string_at(platform, "/os/version"),
            "arch": string_at(platform, "/os/arch"),
            "native_or_virtualized": string_at(platform, "/os/native_or_virtualized")
        },
        "cpu": {
            "model": string_at(platform, "/cpu/model"),
            "cores": u64_at(platform, "/cpu/cores"),
            "threads": u64_at(platform, "/cpu/threads"),
            "p_core_count": u64_at(platform, "/cpu/p_core_count"),
            "lp_e_core_count": u64_at(platform, "/cpu/lp_e_core_count"),
            "avx2_detected": bool_at(platform, "/cpu/avx2_detected"),
            "avx512_detected": bool_at(platform, "/cpu/avx512_detected"),
            "fma_detected": bool_at(platform, "/cpu/fma_detected"),
            "scheduler_hint": string_at(platform, "/cpu/scheduler_hint")
        },
        "memory": {
            "kind": string_at(platform, "/memory/kind"),
            "total_bytes": u64_at(platform, "/memory/total_bytes"),
            "reported_speed_mt_s": u64_at(platform, "/memory/reported_speed_mt_s"),
            "reported_modules": u64_at(platform, "/memory/reported_modules"),
            "shared_memory": bool_at(platform, "/memory/shared_memory")
        },
        "power": {
            "mode": string_at(platform, "/power/mode"),
            "thermal_profile": string_at(platform, "/power/thermal_profile"),
            "sustained_run": bool_at(platform, "/power/sustained_run")
        }
    })
}

fn build_cpu258v_003_profiles(
    phase_profiles: &[Value],
    prompt_tokens: u64,
    generated_tokens: u64,
) -> Vec<Value> {
    let first_token = phase_profile(phase_profiles, "first_token");
    let prefill = phase_profile(phase_profiles, "prefill");
    let decode = phase_profile(phase_profiles, "decode");

    vec![
        cpu258v_profile_from_phase(
            "smoke_1",
            "strict one-token CPU proof",
            first_token,
            "first_token",
            "no supplied strict CPU proof receipt measured the one-token smoke path",
        ),
        cpu258v_profile_from_phase(
            "first_token",
            "first-token latency",
            first_token,
            "first_token",
            "no supplied strict CPU proof receipt measured first token",
        ),
        cpu258v_profile_from_phase(
            "decode_128",
            "steady decode over 128 generated tokens",
            decode,
            "decode",
            "supplied strict CPU proof generated fewer than 128 tokens or did not measure steady-state decode",
        ),
        cpu258v_profile_from_phase(
            "prefill_512",
            "prefill over 512 prompt tokens",
            prefill,
            "prefill",
            "supplied strict CPU proof did not measure a 512-token prefill",
        ),
    ]
    .into_iter()
    .map(|mut profile| {
        profile["prompt_tokens"] = json!(prompt_tokens);
        profile["generated_tokens"] = json!(generated_tokens);
        profile
    })
    .collect()
}

fn phase_profile<'a>(phase_profiles: &'a [Value], name: &str) -> Option<&'a Value> {
    phase_profiles.iter().find(|entry| entry.get("profile").and_then(Value::as_str) == Some(name))
}

fn cpu258v_profile_from_phase(
    profile: &str,
    purpose: &str,
    phase: Option<&Value>,
    source_profile: &str,
    not_run_reason: &str,
) -> Value {
    if let Some(phase) =
        phase.filter(|entry| entry.get("status").and_then(Value::as_str) == Some("measured"))
    {
        return json!({
            "profile": profile,
            "purpose": purpose,
            "status": "measured",
            "source_profile": source_profile,
            "wall_time_ms": phase.get("wall_time_ms").cloned().unwrap_or(Value::Null),
            "median_ms": phase.get("median_ms").cloned().unwrap_or(Value::Null),
            "p95_ms": phase.get("p95_ms").cloned().unwrap_or(Value::Null),
            "tokens_per_second": phase.get("tokens_per_second").cloned().unwrap_or(Value::Null),
            "requested_kernel": phase.get("requested_kernel").cloned().unwrap_or(Value::Null),
            "selected_kernel": phase.get("selected_kernel").cloned().unwrap_or(Value::Null),
            "fallback_used": false
        });
    }

    json!({
        "profile": profile,
        "purpose": purpose,
        "status": "not_run",
        "source_profile": source_profile,
        "reason": not_run_reason,
        "fallback_used": false
    })
}

fn default_profiles(requested_kernel: &str, selected_kernel: &str) -> Vec<Value> {
    PROFILE_NAMES
        .iter()
        .map(|profile| {
            json!({
                "profile": profile,
                "execution_phase": expected_phase(profile),
                "status": "not_run",
                "reason": not_run_reason(profile),
                "requested_kernel": requested_kernel,
                "selected_kernel": selected_kernel,
                "fallback_used": false,
                "fallback_reason": null,
                "shape": {
                    "rows": 1,
                    "cols": 1,
                    "iterations": 1
                }
            })
        })
        .collect()
}

fn apply_measured_phase(
    profiles: &mut [Value],
    proof: &Value,
    requested_kernel: &str,
    selected_kernel: &str,
) {
    let phase = string_at(proof, "/execution/phase")
        .or_else(|| string_at(proof, "/bitnet/execution_phase"))
        .unwrap_or_else(|| "decode".to_string());
    let generated_tokens = u64_at(proof, "/execution/generated_tokens").unwrap_or(1).max(1);
    let prompt_tokens = u64_at(proof, "/execution/prompt_tokens").unwrap_or(1).max(1);
    let cols = u64_at(proof, "/model/context_length")
        .or_else(|| u64_at(proof, "/model/vocab_size"))
        .unwrap_or(1)
        .max(1);

    match phase.as_str() {
        "prefill" => {
            set_prefill_profile(
                profiles,
                proof,
                requested_kernel,
                selected_kernel,
                prompt_tokens,
                cols,
            );
        }
        "first_token" => set_measured(
            profiles,
            PhaseMeasurement {
                profile_name: "first_token",
                requested_kernel,
                selected_kernel,
                rows: 1,
                cols,
                iterations: 1,
                wall_time_ms: number_at(proof, "/profile/decode/first_token_decode_ms")
                    .or_else(|| number_at(proof, "/timing/first_token_decode_ms"))
                    .or_else(|| number_at(proof, "/latency/decode_first_ms"))
                    .or_else(|| number_at(proof, "/latency/total_ms"))
                    .unwrap_or(0.0),
                median_ms: number_at(proof, "/profile/decode/first_token_decode_ms")
                    .or_else(|| number_at(proof, "/timing/first_token_decode_ms"))
                    .or_else(|| number_at(proof, "/latency/decode_first_ms"))
                    .or_else(|| number_at(proof, "/latency/total_ms"))
                    .unwrap_or(0.0),
                p95_ms: number_at(proof, "/profile/decode/first_token_decode_ms")
                    .or_else(|| number_at(proof, "/timing/first_token_decode_ms"))
                    .or_else(|| number_at(proof, "/latency/decode_first_ms"))
                    .or_else(|| number_at(proof, "/latency/total_ms"))
                    .unwrap_or(0.0),
                token_count: 1,
            },
        ),
        "decode" | "decode_steady_state" => {
            set_prefill_profile(
                profiles,
                proof,
                requested_kernel,
                selected_kernel,
                prompt_tokens,
                cols,
            );

            let first_ms = number_at(proof, "/profile/decode/first_token_decode_ms")
                .or_else(|| number_at(proof, "/timing/first_token_decode_ms"))
                .or_else(|| number_at(proof, "/latency/decode_first_ms"))
                .or_else(|| number_at(proof, "/latency/total_ms"))
                .unwrap_or(0.0);
            set_measured(
                profiles,
                PhaseMeasurement {
                    profile_name: "first_token",
                    requested_kernel,
                    selected_kernel,
                    rows: 1,
                    cols,
                    iterations: 1,
                    wall_time_ms: first_ms,
                    median_ms: first_ms,
                    p95_ms: first_ms,
                    token_count: 1,
                },
            );
            let steady_tokens = u64_at(proof, "/profile/decode/steady_state_tokens")
                .unwrap_or_else(|| generated_tokens.saturating_sub(1));
            if steady_tokens > 0 {
                let steady_ms = number_at(proof, "/profile/decode/steady_per_token_ms/total_ms")
                    .or_else(|| {
                        number_at(proof, "/timing/decode_total_ms").map(|total| {
                            if generated_tokens > 1 { (total - first_ms).max(0.0) } else { total }
                        })
                    })
                    .or_else(|| {
                        number_at(proof, "/latency/total_ms")
                            .map(|total| (total - first_ms).max(0.0))
                    })
                    .unwrap_or(0.0);
                let median_ms = number_at(proof, "/profile/decode/steady_per_token_ms/p50_ms")
                    .unwrap_or(steady_ms);
                let p95_ms = number_at(proof, "/profile/decode/steady_per_token_ms/p95_ms")
                    .unwrap_or(steady_ms);
                set_measured(
                    profiles,
                    PhaseMeasurement {
                        profile_name: "decode",
                        requested_kernel,
                        selected_kernel,
                        rows: 1,
                        cols,
                        iterations: steady_tokens,
                        wall_time_ms: steady_ms,
                        median_ms,
                        p95_ms,
                        token_count: steady_tokens,
                    },
                );
            }
        }
        _ => {}
    }
}

fn set_prefill_profile(
    profiles: &mut [Value],
    proof: &Value,
    requested_kernel: &str,
    selected_kernel: &str,
    prompt_tokens: u64,
    cols: u64,
) {
    let prefill_tokens = u64_at(proof, "/profile/prompt_prefill/tokens").unwrap_or(prompt_tokens);
    let prefill_ms = number_at(proof, "/profile/prompt_prefill/ms")
        .or_else(|| number_at(proof, "/timing/prefill_ms"));
    if let Some(wall_time_ms) = prefill_ms.filter(|value| *value > 0.0) {
        set_measured(
            profiles,
            PhaseMeasurement {
                profile_name: "prefill",
                requested_kernel,
                selected_kernel,
                rows: prefill_tokens,
                cols,
                iterations: prefill_tokens,
                wall_time_ms,
                median_ms: number_at(proof, "/profile/prompt_prefill/per_token_ms/p50_ms")
                    .unwrap_or(wall_time_ms),
                p95_ms: number_at(proof, "/profile/prompt_prefill/per_token_ms/p95_ms")
                    .unwrap_or(wall_time_ms),
                token_count: prefill_tokens,
            },
        );
    }
}

fn set_measured(profiles: &mut [Value], measurement: PhaseMeasurement<'_>) {
    if let Some(profile) = profiles.iter_mut().find(|entry| {
        entry.get("profile").and_then(Value::as_str) == Some(measurement.profile_name)
    }) {
        let tokens_per_second = if measurement.wall_time_ms > 0.0 {
            measurement.token_count as f64 / (measurement.wall_time_ms / 1000.0)
        } else {
            0.0
        };
        *profile = json!({
            "profile": measurement.profile_name,
            "execution_phase": expected_phase(measurement.profile_name),
            "status": "measured",
            "requested_kernel": measurement.requested_kernel,
            "selected_kernel": measurement.selected_kernel,
            "fallback_used": false,
            "fallback_reason": null,
            "shape": {
                "rows": measurement.rows.max(1),
                "cols": measurement.cols.max(1),
                "iterations": measurement.iterations.max(1)
            },
            "wall_time_ms": measurement.wall_time_ms.max(0.0),
            "median_ms": measurement.median_ms.max(0.0),
            "p95_ms": measurement.p95_ms.max(0.0),
            "bandwidth_gbps": 0.0,
            "tokens_per_second": tokens_per_second
        });
    }
}

fn expected_phase(profile: &str) -> &'static str {
    match profile {
        "micro" => "micro_kernel",
        "layer" => "layer_forward",
        "prefill" => "prefill",
        "first_token" => "first_token",
        "decode" => "decode_steady_state",
        _ => "unknown",
    }
}

fn not_run_reason(profile: &str) -> &'static str {
    match profile {
        "micro" => {
            "micro is covered by canonical QK256 synthetic benchmark receipts, not this real phase receipt surface"
        }
        "layer" => {
            "layer phase harness is not yet wired to a real transformer block benchmark receipt"
        }
        "prefill" => "no supplied strict CPU proof receipt measured prefill",
        "first_token" => "no supplied strict CPU proof receipt measured first token",
        "decode" => {
            "no supplied strict CPU proof receipt measured steady-state decode with more than one generated token"
        }
        _ => "profile is not wired",
    }
}

fn timestamp_label() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn require_bool_false(value: &Value, pointer: &str) -> Result<(), Box<dyn Error>> {
    match value.pointer(pointer).and_then(Value::as_bool) {
        Some(false) => Ok(()),
        Some(true) => Err(format!("{pointer} must be false").into()),
        None => Ok(()),
    }
}

fn string_at(value: &Value, pointer: &str) -> Option<String> {
    value.pointer(pointer)?.as_str().map(str::to_string)
}

fn u64_at(value: &Value, pointer: &str) -> Option<u64> {
    value.pointer(pointer)?.as_u64()
}

fn bool_at(value: &Value, pointer: &str) -> Option<bool> {
    value.pointer(pointer)?.as_bool()
}

fn number_at(value: &Value, pointer: &str) -> Option<f64> {
    value.pointer(pointer)?.as_f64()
}

fn array_at(value: &Value, pointer: &str) -> Option<Vec<String>> {
    Some(
        value
            .pointer(pointer)?
            .as_array()?
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_receipt_uses_phase_profile_timings_and_prefill() {
        let proof = json!({
            "execution": {
                "phase": "decode",
                "prompt_tokens": 6,
                "generated_tokens": 2
            },
            "model": {
                "context_length": 1024
            },
            "latency": {
                "decode_first_ms": 309902.0,
                "total_ms": 362358.0
            },
            "timing": {
                "first_token_decode_ms": 51212.925,
                "decode_total_ms": 103669.545,
                "prefill_ms": 258689.304
            },
            "profile": {
                "prompt_prefill": {
                    "tokens": 5,
                    "ms": 258689.304,
                    "per_token_ms": {
                        "p50_ms": 51318.026,
                        "p95_ms": 52824.14
                    }
                },
                "decode": {
                    "first_token_decode_ms": 51212.925,
                    "steady_state_tokens": 1,
                    "steady_per_token_ms": {
                        "total_ms": 52456.62,
                        "p50_ms": 52456.62,
                        "p95_ms": 52456.62
                    }
                }
            }
        });
        let mut profiles = default_profiles("i2_s-avx2-reference", "i2_s-avx2-reference");

        apply_measured_phase(&mut profiles, &proof, "i2_s-avx2-reference", "i2_s-avx2-reference");

        let profile = |name: &str| {
            profiles
                .iter()
                .find(|entry| entry.get("profile").and_then(Value::as_str) == Some(name))
                .expect("profile present")
        };
        assert_eq!(profile("prefill")["status"], "measured");
        assert_eq!(profile("prefill")["shape"]["iterations"], 5);
        assert_eq!(profile("prefill")["wall_time_ms"], 258689.304);
        assert_eq!(profile("prefill")["median_ms"], 51318.026);
        assert_eq!(profile("first_token")["wall_time_ms"], 51212.925);
        assert_eq!(profile("decode")["wall_time_ms"], 52456.62);
        assert_eq!(profile("decode")["shape"]["iterations"], 1);
    }

    #[test]
    fn platform_context_copies_258v_topology_memory_and_power() {
        let platform = json!({
            "machine_id": "intel-258v",
            "proof_stage": "detected",
            "status": "partial_platform_detected",
            "os": {
                "name": "Microsoft Windows 11 Home",
                "version": "10.0.26200",
                "arch": "x86_64",
                "native_or_virtualized": "native"
            },
            "cpu": {
                "model": "Intel(R) Core(TM) Ultra 7 258V",
                "cores": 8,
                "threads": 8,
                "p_core_count": 4,
                "lp_e_core_count": 4,
                "avx2_detected": true,
                "avx512_detected": false,
                "fma_detected": true,
                "scheduler_hint": "record topology before benchmark claims"
            },
            "memory": {
                "kind": "shared LPDDR-class system memory",
                "total_bytes": 33873780736_u64,
                "reported_speed_mt_s": 8533,
                "reported_modules": 8,
                "shared_memory": true
            },
            "power": {
                "mode": "Balanced",
                "thermal_profile": null,
                "sustained_run": false
            }
        });

        let context = build_platform_context(
            &PathBuf::from("ci/hardware/intel-258v/2026-05-06/platform-probe.json"),
            &platform,
        );

        assert_eq!(context["machine_id"], "intel-258v");
        assert_eq!(context["cpu"]["p_core_count"], 4);
        assert_eq!(context["cpu"]["lp_e_core_count"], 4);
        assert_eq!(context["cpu"]["avx2_detected"], true);
        assert_eq!(context["cpu"]["avx512_detected"], false);
        assert_eq!(context["memory"]["shared_memory"], true);
        assert_eq!(context["memory"]["reported_speed_mt_s"], 8533);
        assert_eq!(context["power"]["mode"], "Balanced");
    }

    #[test]
    fn cpu258v_003_profiles_keep_unavailable_profiles_explicit() {
        let mut profiles = default_profiles("i2_s-avx2-reference", "i2_s-avx2-reference");
        set_measured(
            &mut profiles,
            PhaseMeasurement {
                profile_name: "first_token",
                requested_kernel: "i2_s-avx2-reference",
                selected_kernel: "i2_s-avx2-reference",
                rows: 1,
                cols: 4096,
                iterations: 1,
                wall_time_ms: 28_156.269,
                median_ms: 28_156.269,
                p95_ms: 28_156.269,
                token_count: 1,
            },
        );

        let requested = build_cpu258v_003_profiles(&profiles, 1, 1);
        let profile = |name: &str| {
            requested
                .iter()
                .find(|entry| entry.get("profile").and_then(Value::as_str) == Some(name))
                .expect("profile present")
        };

        assert_eq!(profile("smoke_1")["status"], "measured");
        assert_eq!(profile("first_token")["status"], "measured");
        assert_eq!(profile("decode_128")["status"], "not_run");
        assert_eq!(profile("prefill_512")["status"], "not_run");
        assert_eq!(profile("smoke_1")["prompt_tokens"], 1);
        assert_eq!(profile("smoke_1")["generated_tokens"], 1);
    }
}
