use anyhow::{Context, Result, bail};
use bitnet_prompt_templates::TemplateType;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
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
    "reference_execution_proven",
    "rust_reference_parity_proven",
    "a770_semantic_quality_proven",
];

const A770_BITNET_QK256_ROUTE_ID: &str = "a770.bitnet.i2s.qk256";
const DEFAULT_REFERENCE_RUN_RECEIPT: &str = "target/a770-diagnostic/bitnet-reference-run.json";
const REFERENCE_TEXT_END_MARKER: &str = " [end of text]";
const REFERENCE_TEXT_PROBE_TOKEN_LIMIT: usize = 32;

#[derive(Debug)]
pub struct ReferencePlanArgs<'a> {
    pub model_contract: &'a Path,
    pub model: Option<&'a Path>,
    pub tokenizer: Option<&'a Path>,
    pub prompt_template: &'a str,
    pub system_prompt: Option<&'a str>,
    pub prompt: &'a str,
    pub max_new_tokens: usize,
    pub reference_exe: Option<&'a Path>,
    pub reference_run_receipt: Option<&'a Path>,
    pub reference_text: Option<&'a str>,
    pub cpp_root: Option<&'a Path>,
    pub output: Option<&'a Path>,
    pub format: &'a str,
}

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-plan") {
        return Ok(false);
    }
    if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(true);
    }

    let mut model_contract = PathBuf::from("docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml");
    let mut model: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut prompt_template = "llama3-chat".to_string();
    let mut system_prompt: Option<String> = None;
    let mut prompt = "What is 2+2?".to_string();
    let mut max_new_tokens = 16usize;
    let mut reference_exe: Option<PathBuf> = None;
    let mut reference_run_receipt = PathBuf::from(DEFAULT_REFERENCE_RUN_RECEIPT);
    let mut reference_text: Option<String> = None;
    let mut cpp_root: Option<PathBuf> = None;
    let mut output = PathBuf::from("target/a770-diagnostic/bitnet-reference-plan.json");
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
            "--model-contract" => model_contract = PathBuf::from(value()?),
            "--model" => model = Some(PathBuf::from(value()?)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value()?)),
            "--prompt-template" => prompt_template = value()?,
            "--system-prompt" => system_prompt = Some(value()?),
            "--prompt" => prompt = value()?,
            "--max-new-tokens" => {
                let raw = value()?;
                max_new_tokens =
                    raw.parse().with_context(|| format!("parsing --max-new-tokens {raw}"))?;
            }
            "--reference-exe" => reference_exe = Some(PathBuf::from(value()?)),
            "--reference-run-receipt" => reference_run_receipt = PathBuf::from(value()?),
            "--reference-text" => reference_text = Some(value()?),
            "--cpp-root" => cpp_root = Some(PathBuf::from(value()?)),
            "--output" => output = PathBuf::from(value()?),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-plan option {other}"),
        }
    }

    run(ReferencePlanArgs {
        model_contract: &model_contract,
        model: model.as_deref(),
        tokenizer: tokenizer.as_deref(),
        prompt_template: &prompt_template,
        system_prompt: system_prompt.as_deref(),
        prompt: &prompt,
        max_new_tokens,
        reference_exe: reference_exe.as_deref(),
        reference_run_receipt: Some(&reference_run_receipt),
        reference_text: reference_text.as_deref(),
        cpp_root: cpp_root.as_deref(),
        output: Some(&output),
        format: &format,
    })?;
    Ok(true)
}

fn print_help() {
    println!(
        "Emit a target-local BitNet C++ reference-readiness plan\n\nUsage: xtask.exe bitnet-reference-plan [OPTIONS]\n\nOptions:\n      --model-contract <PATH>      Model contract YAML file [default: docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml]\n      --model <PATH>               Override model path\n      --tokenizer <PATH>           Override tokenizer path\n      --prompt-template <NAME>     Prompt template [default: llama3-chat]\n      --system-prompt <TEXT>       Optional system prompt\n      --prompt <TEXT>              User prompt [default: What is 2+2?]\n      --max-new-tokens <N>         Max new tokens [default: 16]\n      --reference-exe <PATH>       Explicit C++ reference executable path\n      --reference-run-receipt <PATH> Optional reference-run receipt for diagnostic selected-logit probes [default: target/a770-diagnostic/bitnet-reference-run.json]\n      --reference-text <TEXT>      Explicit diagnostic reference text for selected-logit probes\n      --cpp-root <PATH>            C++ reference checkout/build root\n      --output <PATH>              Output plan JSON [default: target/a770-diagnostic/bitnet-reference-plan.json]\n      --format <human|json>        Output format [default: human]\n  -h, --help                       Print help"
    );
}

pub fn run(args: ReferencePlanArgs<'_>) -> Result<()> {
    let report = build_report(&args)?;
    if let Some(output) = args.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(output, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("writing {}", output.display()))?;
    }
    emit_report(&report, args.format)
}

fn build_report(args: &ReferencePlanArgs<'_>) -> Result<Value> {
    let contract = read_yaml(args.model_contract)?;
    let model_path = args
        .model
        .map(path_to_string)
        .or_else(|| str_at(&contract, "/local_path").map(ToOwned::to_owned))
        .context("model path missing; pass --model or set /local_path in the model contract")?;
    let tokenizer_path = args
        .tokenizer
        .map(path_to_string)
        .or_else(|| str_at(&contract, "/tokenizer/path").map(ToOwned::to_owned))
        .context(
            "tokenizer path missing; pass --tokenizer or set /tokenizer/path in the model contract",
        )?;
    let model_dir =
        Path::new(&model_path).parent().map(path_to_string).unwrap_or_else(|| ".".to_string());
    let template = args
        .prompt_template
        .parse::<TemplateType>()
        .with_context(|| format!("parsing prompt template {}", args.prompt_template))?;
    let rendered_prompt = template.apply(args.prompt, args.system_prompt);
    let add_bos = template.should_add_bos();
    let parse_special = template.parse_special();
    let tokenizer = bitnet_tokenizers::load_tokenizer(Path::new(&tokenizer_path))
        .with_context(|| format!("loading tokenizer {tokenizer_path}"))?;
    let token_ids = tokenizer
        .encode(&rendered_prompt, add_bos, parse_special)
        .with_context(|| "tokenizing rendered prompt with contract tokenizer")?;
    let reference_text_probe =
        reference_text_probe(args.reference_text, args.reference_run_receipt, tokenizer.as_ref())?;

    let candidates = reference_candidates(args.reference_exe, args.cpp_root);
    let selected = candidates.iter().find(|candidate| candidate.exists);
    let reference_ready = selected.is_some();
    let setup_prerequisites = reference_setup_prerequisites();
    let mut blocked_reasons = Vec::new();
    if !reference_ready {
        blocked_reasons.push("reference_executable_missing".to_string());
    }
    if !bool_at(&setup_prerequisites, "/ready").unwrap_or(false) {
        for reason in array_strings(&setup_prerequisites, "/missing") {
            blocked_reasons.push(reason);
        }
    }
    if !Path::new(&model_path).exists() {
        blocked_reasons.push("model_file_missing".to_string());
    }
    if !Path::new(&tokenizer_path).exists() {
        blocked_reasons.push("tokenizer_file_missing".to_string());
    }

    let reference_argv = selected.map(|candidate| {
        reference_argv(&candidate.path, &model_path, &rendered_prompt, args.max_new_tokens)
    });

    Ok(json!({
        "schema_version": 1,
        "diagnostic": "bitnet_reference_plan",
        "producer": "cargo xtask bitnet-reference-plan",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "model": {
            "contract": path_to_string(args.model_contract),
            "model_id": str_at(&contract, "/model_id").unwrap_or("unknown-model"),
            "model_path": model_path,
            "model_exists": Path::new(&model_path).exists(),
            "tokenizer_path": tokenizer_path,
            "tokenizer_exists": Path::new(&tokenizer_path).exists(),
            "weights_sha256": str_at(&contract, "/sha256").unwrap_or(""),
            "tokenizer_sha256": str_at(&contract, "/tokenizer/sha256").unwrap_or(""),
        },
        "prompt_identity": {
            "prompt_template": args.prompt_template,
            "system_prompt_present": args.system_prompt.is_some(),
            "rendered_prompt_sha256": sha256_text(&rendered_prompt),
            "prompt_token_ids_sha256": sha256_token_ids(&token_ids)?,
            "prompt_token_count": token_ids.len(),
            "add_bos": add_bos,
            "parse_special": parse_special,
            "max_new_tokens": args.max_new_tokens,
        },
        "reference": {
            "backend": "bitnet.cpp_or_llama.cpp_cli",
            "ready": reference_ready,
            "selected_executable": selected.map(|candidate| candidate.path.as_str()),
            "setup_prerequisites": setup_prerequisites,
            "candidate_executables": candidates.iter().map(|candidate| {
                json!({
                    "path": candidate.path,
                    "source": candidate.source,
                    "exists": candidate.exists,
                })
            }).collect::<Vec<_>>(),
            "command_argv": reference_argv,
            "command_policy": "uses_rendered_prompt_text_for_template_parity; disables reference auto-BOS because the rendered Llama3 prompt already contains BOS; suppresses prompt echo so stdout is generated text; enables verbose prompt output so the reference run receipt can compare actual reference prompt tokenization",
            "setup_command_pwsh": format!(
                "powershell -ExecutionPolicy Bypass -File ci\\fetch_bitnet_cpp.ps1 -Tag main -CachePath target\\external\\BitNet-reference -ModelDir \"{}\" -QuantType i2_s -Force -SkipPatches",
                model_dir.replace('"', "\\\"")
            ),
        },
        "rust_commands": {
            "proof_identity": {
                "model_contract": path_to_string(args.model_contract),
                "a770_kernel_route": A770_BITNET_QK256_ROUTE_ID,
                "cpu_command_declares_kernel_route": false,
                "a770_command_declares_kernel_route": true,
                "policy": "model contract binds both Rust receipts; kernel route binds only the strict A770 receipt"
            },
            "cpu_argv": rust_cli_argv(
                "cpu",
                &model_path,
                &tokenizer_path,
                args.prompt_template,
                args.system_prompt,
                args.prompt,
                args.max_new_tokens,
                "target/a770-diagnostic/reference-plan-cpu.json",
                Some(args.model_contract),
                None
            ),
            "a770_argv": rust_cli_argv(
                "intel-arc-a770-opencl",
                &model_path,
                &tokenizer_path,
                args.prompt_template,
                args.system_prompt,
                args.prompt,
                args.max_new_tokens,
                "target/a770-diagnostic/reference-plan-a770.json",
                Some(args.model_contract),
                Some(A770_BITNET_QK256_ROUTE_ID)
            ),
            "first_token_logit_policy": {
                "max_new_tokens": 1,
                "dump_logit_steps": 1,
                "logits_topk": 10,
                "selected_logit_probe_ids": reference_text_probe.logit_ids.clone(),
                "assert_greedy": true,
                "purpose": "bounded prompt-matched Rust CPU/A770 receipts for first-token, top-logit, and diagnostic selected-logit comparison"
            },
            "reference_text_tokenization": reference_text_probe.report,
            "cpu_first_token_logit_argv": rust_cli_first_token_logit_argv(
                "cpu",
                &model_path,
                &tokenizer_path,
                args.prompt_template,
                args.system_prompt,
                args.prompt,
                "target/a770-diagnostic/reference-plan-cpu-first-token.json",
                Some(args.model_contract),
                None,
                &reference_text_probe.logit_ids
            ),
            "a770_first_token_logit_argv": rust_cli_first_token_logit_argv(
                "intel-arc-a770-opencl",
                &model_path,
                &tokenizer_path,
                args.prompt_template,
                args.system_prompt,
                args.prompt,
                "target/a770-diagnostic/reference-plan-a770-first-token.json",
                Some(args.model_contract),
                Some(A770_BITNET_QK256_ROUTE_ID),
                &reference_text_probe.logit_ids
            ),
        },
        "decision": {
            "reference_required_before_math_change": true,
            "next_when_reference_ready": "run reference command, compare token ids/top-k logits with Rust CPU and strict A770 receipts",
            "current_blocked_reasons": blocked_reasons,
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn reference_argv(
    executable: &str,
    model_path: &str,
    rendered_prompt: &str,
    max_new_tokens: usize,
) -> Vec<String> {
    vec![
        executable.to_string(),
        "-m".to_string(),
        model_path.to_string(),
        "--override-kv".to_string(),
        "tokenizer.ggml.add_bos_token=bool:false".to_string(),
        "-p".to_string(),
        rendered_prompt.to_string(),
        "-n".to_string(),
        max_new_tokens.to_string(),
        "--temp".to_string(),
        "0".to_string(),
        "--seed".to_string(),
        "0".to_string(),
        "--no-display-prompt".to_string(),
        "--verbose-prompt".to_string(),
    ]
}

#[derive(Debug)]
struct Candidate {
    path: String,
    source: String,
    exists: bool,
}

fn reference_candidates(reference_exe: Option<&Path>, cpp_root: Option<&Path>) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    if let Some(path) = reference_exe {
        push_candidate(&mut candidates, path.to_path_buf(), "explicit --reference-exe");
    }
    for env in ["BITNET_REFERENCE_EXE", "BITNET_CPP_EXE", "LLAMA_CPP_EXE"] {
        if let Ok(value) = std::env::var(env) {
            push_candidate(&mut candidates, PathBuf::from(value), env);
        }
    }

    let mut roots = Vec::new();
    if let Some(root) = cpp_root {
        roots.push((root.to_path_buf(), "explicit --cpp-root".to_string()));
    }
    for env in ["BITNET_CPP_DIR", "BITNET_CPP_PATH", "LLAMA_CPP_DIR"] {
        if let Ok(value) = std::env::var(env) {
            roots.push((PathBuf::from(value), env.to_string()));
        }
    }
    roots.push((
        PathBuf::from("target/external/BitNet-reference"),
        "target/external/BitNet-reference".to_string(),
    ));
    roots.push((PathBuf::from("target/external/BitNet"), "target/external/BitNet".to_string()));
    if let Some(home) = dirs::home_dir() {
        roots.push((home.join(".cache/bitnet_cpp"), "$HOME/.cache/bitnet_cpp".to_string()));
    }

    for (root, source) in roots {
        for subdir in ["build/bin", "build", "bin", ""] {
            for name in executable_names() {
                push_candidate(&mut candidates, root.join(subdir).join(name), &source);
            }
        }
    }

    let mut seen = HashSet::new();
    candidates.retain(|candidate| seen.insert(candidate.path.clone()));
    candidates
}

fn push_candidate(candidates: &mut Vec<Candidate>, path: PathBuf, source: &str) {
    let exists = path.is_file();
    candidates.push(Candidate { path: path_to_string(&path), source: source.to_string(), exists });
}

fn executable_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["llama-cli.exe", "main.exe", "bitnet-cli.exe", "bitnet-lut.exe"]
    } else {
        &["llama-cli", "main", "bitnet-cli", "bitnet-lut"]
    }
}

fn reference_setup_prerequisites() -> Value {
    let git = command_probe("git");
    let cmake = command_probe("cmake");
    let python = command_probe("python");
    let vs_build_tools = visual_studio_build_tools_probe();
    let clang = reference_compiler_probe("clang", &vs_build_tools);
    let clangxx = reference_compiler_probe("clang++", &vs_build_tools);
    let windows = cfg!(windows);

    let mut missing = Vec::new();
    if !bool_at(&git, "/present").unwrap_or(false) {
        missing.push("git_missing".to_string());
    }
    if !bool_at(&cmake, "/present").unwrap_or(false) {
        missing.push("cmake_missing".to_string());
    }
    if !bool_at(&python, "/present").unwrap_or(false) {
        missing.push("python_missing".to_string());
    }
    if windows {
        if !bool_at(&clang, "/present").unwrap_or(false) {
            missing.push("clang_missing".to_string());
        }
        if !bool_at(&clangxx, "/present").unwrap_or(false) {
            missing.push("clangxx_missing".to_string());
        }
        if !bool_at(&vs_build_tools, "/present").unwrap_or(false) {
            missing.push("visual_studio_build_tools_missing".to_string());
        }
    }

    json!({
        "ready": missing.is_empty(),
        "windows": windows,
        "git": git,
        "cmake": cmake,
        "python": python,
        "clang": clang,
        "clangxx": clangxx,
        "visual_studio_build_tools": vs_build_tools,
        "missing": missing,
        "windows_required_components": [
            "C++ Clang Compiler for Windows",
            "MS-Build Support for LLVM-Toolset",
            "Desktop development with C++"
        ],
    })
}

fn command_probe(name: &str) -> Value {
    command_probe_with_candidates(name, &[])
}

fn reference_compiler_probe(name: &str, vs_build_tools: &Value) -> Value {
    let candidates = if cfg!(windows) {
        windows_clang_candidate_paths(name, vs_build_tools)
    } else {
        Vec::new()
    };
    command_probe_with_candidates(name, &candidates)
}

fn command_probe_with_candidates(name: &str, candidates: &[PathBuf]) -> Value {
    match which::which(name) {
        Ok(path) => json!({
            "present": true,
            "path": path_to_string(&path),
            "source": "path",
        }),
        Err(_) => {
            for path in candidates {
                if path.is_file() {
                    return json!({
                        "present": true,
                        "path": path_to_string(path),
                        "source": "known_windows_toolchain_path",
                    });
                }
            }
            json!({
                "present": false,
                "path": Value::Null,
                "source": Value::Null,
            })
        }
    }
}

fn windows_clang_candidate_paths(name: &str, vs_build_tools: &Value) -> Vec<PathBuf> {
    let executable = windows_tool_executable_name(name);
    let mut candidates = Vec::new();

    if let Some(program_files) = std::env::var_os("ProgramFiles") {
        candidates.push(PathBuf::from(program_files).join("LLVM").join("bin").join(&executable));
    }

    if let Some(vs_path) = str_at(vs_build_tools, "/path") {
        push_vs_llvm_candidates(&mut candidates, Path::new(vs_path), &executable);
    }

    if let Some(program_files_x86) = std::env::var_os("ProgramFiles(x86)") {
        let build_tools = PathBuf::from(program_files_x86)
            .join("Microsoft Visual Studio")
            .join("2022")
            .join("BuildTools");
        push_vs_llvm_candidates(&mut candidates, &build_tools, &executable);
    }

    dedupe_paths(candidates)
}

fn windows_tool_executable_name(name: &str) -> String {
    if name.to_ascii_lowercase().ends_with(".exe") {
        name.to_string()
    } else {
        format!("{name}.exe")
    }
}

fn push_vs_llvm_candidates(candidates: &mut Vec<PathBuf>, vs_path: &Path, executable: &str) {
    candidates.push(
        vs_path.join("VC").join("Tools").join("Llvm").join("x64").join("bin").join(executable),
    );
    candidates.push(vs_path.join("VC").join("Tools").join("Llvm").join("bin").join(executable));
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for path in paths {
        let key = path_to_string(&path).to_ascii_lowercase();
        if seen.insert(key) {
            deduped.push(path);
        }
    }
    deduped
}

fn visual_studio_build_tools_probe() -> Value {
    if !cfg!(windows) {
        return json!({
            "present": false,
            "path": Value::Null,
            "not_required_on_this_platform": true,
        });
    }
    let Some(program_files_x86) = std::env::var_os("ProgramFiles(x86)") else {
        return json!({
            "present": false,
            "path": Value::Null,
        });
    };
    let vswhere = PathBuf::from(program_files_x86)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    if !vswhere.is_file() {
        return json!({
            "present": false,
            "path": Value::Null,
            "vswhere": path_to_string(&vswhere),
        });
    }
    let output = std::process::Command::new(&vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            json!({
                "present": !path.is_empty(),
                "path": if path.is_empty() { Value::Null } else { Value::String(path) },
                "vswhere": path_to_string(&vswhere),
            })
        }
        _ => json!({
            "present": false,
            "path": Value::Null,
            "vswhere": path_to_string(&vswhere),
        }),
    }
}

fn rust_cli_argv(
    device: &str,
    model: &str,
    tokenizer: &str,
    prompt_template: &str,
    system_prompt: Option<&str>,
    prompt: &str,
    max_new_tokens: usize,
    json_out: &str,
    proof_model_contract: Option<&Path>,
    proof_kernel_route: Option<&str>,
) -> Vec<String> {
    let features =
        if device.contains("a770") || device.contains("opencl") { "opencl" } else { "cpu" };
    let mut argv = vec![
        "cargo".to_string(),
        "run".to_string(),
        "--locked".to_string(),
        "-p".to_string(),
        "bitnet-cli".to_string(),
        "--no-default-features".to_string(),
        "--features".to_string(),
        features.to_string(),
        "--".to_string(),
        "run".to_string(),
        "--device".to_string(),
        device.to_string(),
        "--model".to_string(),
        model.to_string(),
        "--tokenizer".to_string(),
        tokenizer.to_string(),
        "--model-format".to_string(),
        "gguf".to_string(),
        "--prompt-template".to_string(),
        prompt_template.to_string(),
    ];
    if let Some(system_prompt) = system_prompt {
        argv.push("--system-prompt".to_string());
        argv.push(system_prompt.to_string());
    }
    argv.extend([
        "--prompt".to_string(),
        prompt.to_string(),
        "--max-new-tokens".to_string(),
        max_new_tokens.to_string(),
        "--greedy".to_string(),
        "--deterministic".to_string(),
        "--strict-loader".to_string(),
        "--strict-tokenizer".to_string(),
        "--strict-backend".to_string(),
        "--no-warnings".to_string(),
        "--json-out".to_string(),
        json_out.to_string(),
    ]);
    if let Some(model_contract) = proof_model_contract {
        argv.push("--proof-model-contract".to_string());
        argv.push(path_to_string(model_contract));
    }
    if let Some(kernel_route) = proof_kernel_route {
        argv.push("--proof-kernel-route".to_string());
        argv.push(kernel_route.to_string());
    }
    argv
}

fn rust_cli_first_token_logit_argv(
    device: &str,
    model: &str,
    tokenizer: &str,
    prompt_template: &str,
    system_prompt: Option<&str>,
    prompt: &str,
    json_out: &str,
    proof_model_contract: Option<&Path>,
    proof_kernel_route: Option<&str>,
    selected_logit_ids: &[u32],
) -> Vec<String> {
    let mut argv = rust_cli_argv(
        device,
        model,
        tokenizer,
        prompt_template,
        system_prompt,
        prompt,
        1,
        json_out,
        proof_model_contract,
        proof_kernel_route,
    );
    argv.extend([
        "--dump-logit-steps".to_string(),
        "1".to_string(),
        "--logits-topk".to_string(),
        "10".to_string(),
        "--assert-greedy".to_string(),
    ]);
    for token_id in selected_logit_ids {
        argv.push("--logit-id".to_string());
        argv.push(token_id.to_string());
    }
    argv
}

#[derive(Debug)]
struct ReferenceTextProbe {
    report: Value,
    logit_ids: Vec<u32>,
}

fn reference_text_probe(
    explicit_text: Option<&str>,
    receipt_path: Option<&Path>,
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
) -> Result<ReferenceTextProbe> {
    let mut source = "not_configured";
    let mut source_path = Value::Null;
    let mut source_exists = Value::Null;
    let mut raw_text: Option<String> = None;

    if let Some(text) = explicit_text {
        source = "explicit_reference_text";
        raw_text = Some(text.to_string());
    } else if let Some(path) = receipt_path {
        source = "reference_run_receipt";
        source_path = Value::String(path_to_string(path));
        source_exists = Value::Bool(path.exists());
        if path.exists() {
            let receipt = read_json(path)?;
            raw_text = str_at(&receipt, "/generated_text")
                .or_else(|| str_at(&receipt, "/text"))
                .map(ToOwned::to_owned);
        }
    }

    let (normalized_text, normalization_notes) = raw_text
        .as_deref()
        .map(normalize_reference_text_for_probe)
        .unwrap_or_else(|| ("".to_string(), Vec::new()));
    let token_ids = if normalized_text.is_empty() {
        Vec::new()
    } else {
        tokenizer
            .encode(&normalized_text, false, false)
            .with_context(|| "tokenizing diagnostic reference text for selected logit probes")?
    };
    let logit_ids = unique_limited_token_ids(&token_ids, REFERENCE_TEXT_PROBE_TOKEN_LIMIT);

    Ok(ReferenceTextProbe {
        report: json!({
            "diagnostic_only": true,
            "claimable": false,
            "source": source,
            "source_path": source_path,
            "source_exists": source_exists,
            "raw_text_present": raw_text.as_deref().is_some_and(|text| !text.is_empty()),
            "normalized_text_present": !normalized_text.is_empty(),
            "normalization_notes": normalization_notes,
            "normalization_policy": "trim whitespace and strip trailing reference CLI end marker before diagnostic text tokenization",
            "tokenization_policy": {
                "add_bos": false,
                "parse_special": false,
                "token_id_limit": REFERENCE_TEXT_PROBE_TOKEN_LIMIT,
                "deduplicate_preserving_order": true
            },
            "normalized_text_sha256": if normalized_text.is_empty() {
                Value::Null
            } else {
                Value::String(sha256_text(&normalized_text))
            },
            "token_ids_sha256": if token_ids.is_empty() {
                Value::Null
            } else {
                Value::String(sha256_token_ids(&token_ids)?)
            },
            "token_count": token_ids.len(),
            "candidate_token_ids": token_ids,
            "selected_logit_probe_ids": logit_ids.clone(),
            "not_claims": [
                "reference_text_tokenization_proves_reference_generated_token_ids",
                "reference_text_tokenization_proves_reference_top_logits",
                "reference_text_tokenization_promotes_reference_parity",
                "reference_text_tokenization_promotes_a770_semantic_quality"
            ]
        }),
        logit_ids,
    })
}

fn normalize_reference_text_for_probe(text: &str) -> (String, Vec<String>) {
    let mut normalized = text.trim().to_string();
    let mut notes = Vec::new();
    if normalized.ends_with(REFERENCE_TEXT_END_MARKER) {
        normalized.truncate(normalized.len() - REFERENCE_TEXT_END_MARKER.len());
        normalized = normalized.trim_end().to_string();
        notes.push("stripped_trailing_reference_end_marker".to_string());
    }
    (normalized, notes)
}

fn unique_limited_token_ids(token_ids: &[u32], limit: usize) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for token_id in token_ids {
        if seen.insert(*token_id) {
            unique.push(*token_id);
        }
        if unique.len() == limit {
            break;
        }
    }
    unique
}

fn read_yaml(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_yaml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

fn bool_at(value: &Value, pointer: &str) -> Option<bool> {
    value.pointer(pointer).and_then(Value::as_bool)
}

fn array_strings(value: &Value, pointer: &str) -> Vec<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).map(ToOwned::to_owned).collect())
        .unwrap_or_default()
}

fn path_to_string(path: &Path) -> String {
    path.display().to_string()
}

fn sha256_text(value: &str) -> String {
    sha256_bytes(value.as_bytes())
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
            println!(
                "diagnostic: {}",
                str_at(value, "/diagnostic").unwrap_or("bitnet_reference_plan")
            );
            println!(
                "classification: {}",
                str_at(value, "/classification").unwrap_or("diagnostic_only")
            );
            println!(
                "reference_ready: {}",
                value.pointer("/reference/ready").and_then(Value::as_bool).unwrap_or(false)
            );
            if let Some(reasons) = value.pointer("/decision/current_blocked_reasons") {
                println!("blocked_reasons: {}", serde_json::to_string(reasons)?);
            }
            println!("not_claims: {}", serde_json::to_string(&value["not_claims"])?);
        }
        other => bail!("unsupported bitnet-reference-plan output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_cli_argv_keeps_a770_and_cpu_feature_routes_separate() {
        let contract = Path::new("docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml");
        let cpu = rust_cli_argv(
            "cpu",
            "m.gguf",
            "tok.json",
            "raw",
            None,
            "x",
            1,
            "cpu.json",
            Some(contract),
            None,
        );
        let a770 = rust_cli_argv(
            "intel-arc-a770-opencl",
            "m.gguf",
            "tok.json",
            "raw",
            None,
            "x",
            1,
            "a770.json",
            Some(contract),
            Some(A770_BITNET_QK256_ROUTE_ID),
        );
        assert!(cpu.windows(2).any(|args| args == ["--features", "cpu"]));
        assert!(a770.windows(2).any(|args| args == ["--features", "opencl"]));
        assert!(a770.windows(2).any(|args| args == ["--device", "intel-arc-a770-opencl"]));
        assert!(
            cpu.windows(2)
                .any(|args| args == ["--proof-model-contract", contract.to_str().unwrap()])
        );
        assert!(
            a770.windows(2)
                .any(|args| args == ["--proof-model-contract", contract.to_str().unwrap()])
        );
        assert!(!cpu.iter().any(|arg| arg == "--proof-kernel-route"));
        assert!(
            a770.windows(2)
                .any(|args| args == ["--proof-kernel-route", A770_BITNET_QK256_ROUTE_ID])
        );
    }

    #[test]
    fn reference_candidates_include_explicit_path() {
        let path = Path::new("target/reference/llama-cli.exe");
        let candidates = reference_candidates(Some(path), None);
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.path.ends_with("target/reference/llama-cli.exe"))
        );
    }

    #[test]
    fn setup_prerequisites_report_missing_list() {
        let report = reference_setup_prerequisites();
        assert!(report.pointer("/ready").and_then(Value::as_bool).is_some());
        assert!(report.pointer("/missing").and_then(Value::as_array).is_some());
    }

    #[test]
    fn command_probe_reports_known_candidate_path() {
        let dir = tempfile::tempdir().unwrap();
        let tool = dir.path().join(windows_tool_executable_name("definitely-not-on-path"));
        fs::write(&tool, b"").unwrap();

        let report = command_probe_with_candidates("definitely-not-on-path", &[tool.clone()]);

        assert_eq!(report.pointer("/present").and_then(Value::as_bool), Some(true));
        assert_eq!(
            report.pointer("/path").and_then(Value::as_str),
            Some(path_to_string(&tool).as_str())
        );
        assert_eq!(
            report.pointer("/source").and_then(Value::as_str),
            Some("known_windows_toolchain_path")
        );
    }

    #[test]
    fn windows_clang_candidates_include_vs_llvm_bins() {
        let vs_build_tools = json!({
            "present": true,
            "path": "C:/BuildTools",
        });
        let candidates = windows_clang_candidate_paths("clang++", &vs_build_tools);
        let candidate_strings =
            candidates.iter().map(|path| path_to_string(path)).collect::<Vec<_>>();

        assert!(
            candidate_strings.iter().any(|path| path
                .ends_with("C:/BuildTools\\VC\\Tools\\Llvm\\x64\\bin\\clang++.exe")
                || path.ends_with("C:/BuildTools/VC/Tools/Llvm/x64/bin/clang++.exe"))
        );
        assert_eq!(candidate_strings.iter().collect::<HashSet<_>>().len(), candidate_strings.len());
    }

    #[test]
    fn reference_argv_disables_auto_bos_and_prompt_echo() {
        let argv = reference_argv("llama-cli", "model.gguf", "<|begin_of_text|>prompt", 16);

        assert!(
            argv.windows(2).any(|args| {
                args == ["--override-kv", "tokenizer.ggml.add_bos_token=bool:false"]
            })
        );
        assert!(argv.iter().any(|arg| arg == "--no-display-prompt"));
        assert!(argv.iter().any(|arg| arg == "--verbose-prompt"));
    }

    #[test]
    fn first_token_logit_argv_is_bounded_and_dumps_top_logits() {
        let contract = Path::new("docs/model-contracts/bitnet-b1.58-2b-4t-i2s.yaml");
        let argv = rust_cli_first_token_logit_argv(
            "intel-arc-a770-opencl",
            "m.gguf",
            "tok.json",
            "llama3-chat",
            None,
            "What is 2+2?",
            "a770-first-token.json",
            Some(contract),
            Some(A770_BITNET_QK256_ROUTE_ID),
            &[17, 10, 17239],
        );

        assert!(argv.windows(2).any(|args| args == ["--max-new-tokens", "1"]));
        assert!(argv.windows(2).any(|args| args == ["--dump-logit-steps", "1"]));
        assert!(argv.windows(2).any(|args| args == ["--logits-topk", "10"]));
        assert!(argv.windows(2).any(|args| args == ["--logit-id", "17"]));
        assert!(argv.windows(2).any(|args| args == ["--logit-id", "10"]));
        assert!(argv.windows(2).any(|args| args == ["--logit-id", "17239"]));
        assert!(argv.iter().any(|arg| arg == "--assert-greedy"));
        assert!(
            argv.windows(2)
                .any(|args| args == ["--proof-kernel-route", A770_BITNET_QK256_ROUTE_ID])
        );
    }

    #[test]
    fn reference_text_probe_normalization_strips_cli_end_marker() {
        let (normalized, notes) = normalize_reference_text_for_probe("2+2 equals 4. [end of text]");

        assert_eq!(normalized, "2+2 equals 4.");
        assert_eq!(notes, vec!["stripped_trailing_reference_end_marker"]);
    }

    #[test]
    fn selected_logit_probe_ids_are_unique_and_limited() {
        let ids = unique_limited_token_ids(&[17, 10, 17, 17239, 220, 19, 13], 4);

        assert_eq!(ids, vec![17, 10, 17239, 220]);
    }
}
