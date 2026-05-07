use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde_json::Value;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    thread::sleep,
    time::Duration,
};

mod ci;
mod docker;
mod hooks;
mod models;
mod testing;
mod validation;

use self::{
    ci::{cmd_ci_local, cmd_quality_gate, cmd_sanity_check, cmd_verify_crossval, cmd_verify_tests},
    docker::{cmd_build_cpp_static, cmd_docker_build},
    hooks::cmd_install_hooks,
    models::{
        cmd_bitnet_accept, cmd_generate_policy, cmd_resolve_model_path, cmd_show_quant_status,
        cmd_vendor_ggml_quants,
    },
    testing::{
        cmd_detect_flake, cmd_docs_automation, cmd_docs_test, cmd_ffi_smoke,
        cmd_perf_phase1_quant_probe, cmd_quick_validate, cmd_run_fuzz, cmd_run_miri,
        cmd_smoke_inference, cmd_start, cmd_test_determinism, cmd_test_download,
        cmd_test_generation, cmd_test_iq2s_backend, cmd_test_memory_validation,
        cmd_test_optimizations, cmd_test_policy, cmd_test_quant_support, cmd_test_quick,
        cmd_test_real_tokenizer, cmd_test_simple, cmd_test_token_generation, cmd_xtask_smoke,
    },
    validation::{
        cmd_check_codeowners_teams, cmd_check_envlock, cmd_check_feature_gates,
        cmd_check_ignore_annotations, cmd_check_serial_annotations, cmd_check_units,
        cmd_check_units_imports, cmd_json_schema_gate, cmd_validate_fixtures,
        cmd_validate_iq2s_build, cmd_validate_strict,
    },
};

const DEFAULT_MODEL: &str = "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

#[derive(Parser)]
#[command(
    name = "bitnet-task",
    about = "Compatibility facade for migrated maintenance shell scripts; use xtask for internal developer workflows"
)]
struct Cli {
    #[command(subcommand)]
    command: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Equivalent of scripts/bitnet_accept.sh
    BitnetAccept {
        #[arg(default_value = DEFAULT_MODEL)]
        model: String,
        #[arg(default_value = "models/llama3-tokenizer/tokenizer.json")]
        tokenizer: String,
    },
    /// Equivalent of scripts/build_cpp_static.sh
    BuildCppStatic {
        /// Optional override for the BitNet.cpp checkout path.
        #[arg(long)]
        cpp_dir: Option<PathBuf>,
    },
    /// Equivalent of scripts/check-ignore-annotations.sh
    CheckIgnoreAnnotations,
    /// Equivalent of scripts/check-envlock.sh
    CheckEnvlock,
    /// Equivalent of scripts/check-units-imports.sh
    CheckUnitsImports,
    /// Equivalent of scripts/check-units.sh
    CheckUnits,
    /// Equivalent of scripts/check-serial-annotations.sh
    CheckSerialAnnotations,
    /// Equivalent of scripts/check-codeowners-teams.sh
    CheckCodeownersTeams,
    /// Equivalent of scripts/resolve_model_path.sh
    ResolveModelPath {
        /// Required model path or model name.
        model: String,
    },
    /// Equivalent of scripts/quality-gate.sh
    QualityGate,
    /// Equivalent of scripts/sanity-check.sh
    SanityCheck,
    /// Equivalent of scripts/quick-validate.sh
    QuickValidate {
        /// Rebuild bitnet CLI before running validation.
        #[arg(long)]
        rebuild: bool,
    },
    /// Equivalent of scripts/test-policy.sh
    TestPolicy,
    /// Equivalent of scripts/test-simple.sh
    TestSimple,
    /// Equivalent of scripts/test-quick.sh
    TestQuick {
        /// Optional model path for parity test.
        model: Option<String>,
    },
    /// Equivalent of scripts/detect_flake.sh
    DetectFlake,
    /// Equivalent of scripts/perf_phase1_quant_probe.sh
    PerfPhase1QuantProbe {
        /// Optional model override.
        #[arg(long)]
        model: Option<String>,
        /// Optional tokenizer override.
        #[arg(long)]
        tokenizer: Option<String>,
    },
    /// Equivalent of scripts/test_real_tokenizer.sh
    TestRealTokenizer,
    /// Equivalent of scripts/ffi_smoke.sh
    FfiSmoke,
    /// Equivalent of scripts/test_memory_validation.sh
    TestMemoryValidation,
    /// Equivalent of scripts/test_token_generation.sh
    TestTokenGeneration {
        #[arg(default_value = DEFAULT_MODEL)]
        model: String,
    },
    /// Equivalent of scripts/test-iq2s-backend.sh
    TestIq2sBackend,
    /// Equivalent of scripts/smoke_inference.sh
    SmokeInference {
        /// Model to use for smoke inference.
        #[arg(default_value = "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf")]
        model: String,
        /// Tokenizer to use for smoke inference.
        #[arg(default_value = "models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json")]
        tokenizer: String,
    },
    /// Equivalent of scripts/start.sh
    Start,
    /// Equivalent of scripts/docs_automation.sh
    DocsAutomation,
    /// Equivalent of scripts/docs-test.sh
    DocsTest,
    /// Equivalent of scripts/vendor_ggml_quants.sh
    VendorGgmlQuants {
        #[arg(default_value = "master")]
        commit: String,
    },
    /// Equivalent of scripts/generate_policy.sh
    GeneratePolicy {
        /// Path to GGUF model.
        model: String,
        /// Output policy file path.
        #[arg(default_value = "config/correction-policy.yml")]
        output: String,
    },
    /// Equivalent of scripts/docker-build.sh
    DockerBuild {
        /// Target image family to build (cpu | gpu | all).
        #[arg(default_value = "cpu")]
        target: String,
    },
    /// Equivalent of scripts/json-schema-gate.sh
    JsonSchemaGate {
        /// One or more JSON schema files to validate.
        #[arg(required = true)]
        files: Vec<String>,
    },
    /// Equivalent of scripts/show-quant-status.sh
    ShowQuantStatus,
    /// Equivalent of scripts/validate_fixtures.sh
    ValidateFixtures,
    /// Equivalent of scripts/test_download.sh
    TestDownload,
    /// Equivalent of scripts/test_generation.sh
    TestGeneration,
    /// Equivalent of scripts/test_quant_support.sh
    TestQuantSupport,
    /// Equivalent of scripts/install-hooks.sh
    InstallHooks,
    /// Equivalent of scripts/check-feature-gates.sh
    CheckFeatureGates,
    /// Equivalent of scripts/ci-local.sh
    CiLocal {
        /// Optional mode (workspace | bitnet-server-receipts).
        mode: Option<String>,
    },
    /// Equivalent of scripts/validate-iq2s-build.sh
    ValidateIq2sBuild,
    /// Equivalent of scripts/validate-strict.sh
    ValidateStrict,
    /// Equivalent of scripts/test-determinism.sh
    TestDeterminism,
    /// Equivalent of scripts/run-miri.sh
    RunMiri,
    /// Equivalent of scripts/run-fuzz.sh
    RunFuzz {
        /// Run a specific fuzz target.
        #[arg(short, long)]
        target: Option<String>,
        /// Duration in seconds for each target run.
        #[arg(short, long, default_value = "60")]
        duration: u64,
    },
    /// Equivalent of scripts/verify-crossval.sh
    VerifyCrossval,
    /// Equivalent of scripts/test-optimizations.sh
    TestOptimizations,
    /// Equivalent of scripts/verify-tests.sh
    VerifyTests,
    /// Equivalent of scripts/xtask_smoke.sh
    XtaskSmoke {
        /// Optional model path override (defaults to env MODEL if unset).
        #[arg(long)]
        model: Option<String>,
        /// Optional tokenizer path override.
        #[arg(long)]
        tokenizer: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = workspace_root()?;

    match cli.command {
        Task::BitnetAccept { model, tokenizer } => cmd_bitnet_accept(&root, &model, &tokenizer),
        Task::BuildCppStatic { cpp_dir } => cmd_build_cpp_static(&root, cpp_dir.as_deref()),
        Task::CheckIgnoreAnnotations => cmd_check_ignore_annotations(&root),
        Task::CheckEnvlock => cmd_check_envlock(&root),
        Task::CheckUnitsImports => cmd_check_units_imports(&root),
        Task::CheckUnits => cmd_check_units(&root),
        Task::ResolveModelPath { model } => cmd_resolve_model_path(&root, &model),
        Task::QualityGate => cmd_quality_gate(&root),
        Task::SanityCheck => cmd_sanity_check(&root),
        Task::QuickValidate { rebuild } => cmd_quick_validate(&root, rebuild),
        Task::TestPolicy => cmd_test_policy(&root),
        Task::TestSimple => cmd_test_simple(&root),
        Task::TestQuick { model } => cmd_test_quick(&root, model),
        Task::TestTokenGeneration { model } => cmd_test_token_generation(&root, model),
        Task::CheckSerialAnnotations => cmd_check_serial_annotations(&root),
        Task::CheckCodeownersTeams => cmd_check_codeowners_teams(&root),
        Task::DetectFlake => cmd_detect_flake(&root),
        Task::PerfPhase1QuantProbe { model, tokenizer } => {
            cmd_perf_phase1_quant_probe(&root, model, tokenizer)
        }
        Task::TestRealTokenizer => cmd_test_real_tokenizer(&root),
        Task::FfiSmoke => cmd_ffi_smoke(&root),
        Task::TestMemoryValidation => cmd_test_memory_validation(&root),
        Task::TestIq2sBackend => cmd_test_iq2s_backend(&root),
        Task::SmokeInference { model, tokenizer } => cmd_smoke_inference(&root, &model, &tokenizer),
        Task::Start => cmd_start(&root),
        Task::DocsAutomation => cmd_docs_automation(&root),
        Task::DocsTest => cmd_docs_test(&root),
        Task::VendorGgmlQuants { commit } => cmd_vendor_ggml_quants(&root, commit),
        Task::GeneratePolicy { model, output } => cmd_generate_policy(&root, &model, &output),
        Task::DockerBuild { target } => cmd_docker_build(&root, &target),
        Task::JsonSchemaGate { files } => cmd_json_schema_gate(&root, files),
        Task::ShowQuantStatus => cmd_show_quant_status(&root),
        Task::ValidateFixtures => cmd_validate_fixtures(&root),
        Task::TestDownload => cmd_test_download(&root),
        Task::TestGeneration => cmd_test_generation(&root),
        Task::TestQuantSupport => cmd_test_quant_support(&root),
        Task::InstallHooks => cmd_install_hooks(&root),
        Task::CheckFeatureGates => cmd_check_feature_gates(&root),
        Task::CiLocal { mode } => cmd_ci_local(&root, mode),
        Task::ValidateIq2sBuild => cmd_validate_iq2s_build(&root),
        Task::ValidateStrict => cmd_validate_strict(&root),
        Task::TestDeterminism => cmd_test_determinism(&root),
        Task::RunMiri => cmd_run_miri(&root),
        Task::RunFuzz { target, duration } => cmd_run_fuzz(&root, target, duration),
        Task::VerifyCrossval => cmd_verify_crossval(&root),
        Task::TestOptimizations => cmd_test_optimizations(&root),
        Task::VerifyTests => cmd_verify_tests(&root),
        Task::XtaskSmoke { model, tokenizer } => cmd_xtask_smoke(&root, model, tokenizer),
    }
}

fn workspace_root() -> Result<PathBuf> {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if root.join("Cargo.toml").exists() {
            return Ok(root);
        }
        if !root.pop() {
            break;
        }
    }
    bail!("unable to locate workspace root from {}", env!("CARGO_MANIFEST_DIR"));
}

fn command_to_string<S: AsRef<str>>(program: &str, args: &[S]) -> String {
    let mut pieces = vec![program.to_string()];
    pieces.extend(args.iter().map(|arg| arg.as_ref().to_string()));
    pieces.join(" ")
}

fn run_stream<S: AsRef<str>>(
    cwd: &Path,
    program: &str,
    args: &[S],
    envs: &[(&str, &str)],
) -> Result<()> {
    run_capture_with_env(cwd, program, args, envs, &[], false)?;
    Ok(())
}

fn run_capture<S: AsRef<str>>(
    cwd: &Path,
    program: &str,
    args: &[S],
    envs: &[(&str, &str)],
    allow_failure: bool,
) -> Result<Output> {
    run_capture_with_env(cwd, program, args, envs, &[], allow_failure)
}

fn run_capture_with_env<S: AsRef<str>>(
    cwd: &Path,
    program: &str,
    args: &[S],
    envs: &[(&str, &str)],
    remove_envs: &[&str],
    allow_failure: bool,
) -> Result<Output> {
    let mut command = Command::new(program);
    command.current_dir(cwd).args(args.iter().map(AsRef::as_ref));
    for (key, value) in envs {
        command.env(key, value);
    }
    for key in remove_envs {
        command.env_remove(key);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let output = command
        .output()
        .with_context(|| format!("failed to run `{}`", command_to_string(program, args)))?;
    if !allow_failure && !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "command `{}` failed with {}: {}",
            command_to_string(program, args),
            output.status,
            stderr.trim()
        );
    }
    Ok(output)
}

fn run_capture_with_timeout<S: AsRef<str>>(
    cwd: &Path,
    program: &str,
    args: &[S],
    envs: &[(&str, &str)],
    timeout: Duration,
) -> Result<(Output, bool)> {
    let mut command = Command::new(program);
    command.current_dir(cwd).args(args.iter().map(AsRef::as_ref));
    for (key, value) in envs {
        command.env(key, value);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to run `{}`", command_to_string(program, args)))?;
    let started = std::time::Instant::now();

    loop {
        if child
            .try_wait()
            .with_context(|| format!("failed to poll `{}`", command_to_string(program, args)))?
            .is_some()
        {
            let output = child.wait_with_output().with_context(|| {
                format!("failed to collect `{}` output", command_to_string(program, args))
            })?;
            return Ok((output, false));
        }

        if started.elapsed() >= timeout {
            let _ = child.kill();
            let output = child.wait_with_output().with_context(|| {
                format!("failed to collect `{}` output", command_to_string(program, args))
            })?;
            return Ok((output, true));
        }

        sleep(Duration::from_millis(50));
    }
}

fn run_xtask<S: AsRef<str>>(
    root: &Path,
    xtask_args: &[S],
    envs: &[(&str, &str)],
    allow_failure: bool,
) -> Result<Output> {
    let mut all_args = Vec::with_capacity(xtask_args.len() + 4);
    all_args.push("run".to_string());
    all_args.push("-p".to_string());
    all_args.push("xtask".to_string());
    all_args.push("--".to_string());
    all_args.extend(xtask_args.iter().map(|arg| arg.as_ref().to_string()));
    run_capture(root, "cargo", &all_args, envs, allow_failure)
}

fn run_xtask_binary<S: AsRef<str>>(
    root: &Path,
    xtask_bin: &Path,
    xtask_args: &[S],
    allow_failure: bool,
) -> Result<Output> {
    let program = xtask_bin
        .to_str()
        .with_context(|| format!("non-utf8 xtask path: {}", xtask_bin.display()))?;
    let args = xtask_args;
    run_capture(root, program, args, &[], allow_failure)
}

fn collect_preflight_env() -> Result<Vec<(String, String)>> {
    let pids_used = if command_available("ps") {
        let output = run_capture(Path::new("."), "ps", &["-e"], &[], true)?;
        if output.status.success() {
            String::from_utf8_lossy(&output.stdout).lines().count() as u64
        } else {
            0
        }
    } else {
        0
    };
    let pid_max = fs::read_to_string("/proc/sys/kernel/pid_max")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let file_stats = fs::read_to_string("/proc/sys/fs/file-nr").ok();
    let files_used = file_stats
        .as_deref()
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let files_max = fs::read_to_string("/proc/sys/fs/file-max")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let load_avg = fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|value| value.split_whitespace().next().map(ToString::to_string))
        .unwrap_or_else(|| "unknown".to_string());

    let pids_pct = if pid_max > 0 { pids_used.saturating_mul(100) / pid_max } else { 0 };
    let files_pct = if files_max > 0 { files_used.saturating_mul(100) / files_max } else { 0 };

    println!("=== BitNet-rs System Resource Check ===");
    println!("PIDs: {pids_used} / {pid_max} ({pids_pct}%)");
    println!("Open files: {files_used} / {files_max} ({files_pct}%)");
    println!("Load average: {load_avg}");

    let mut envs = vec![
        (
            "RUST_TEST_THREADS".to_string(),
            env::var("RUST_TEST_THREADS").unwrap_or_else(|_| "2".to_string()),
        ),
        (
            "RAYON_NUM_THREADS".to_string(),
            env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "2".to_string()),
        ),
        (
            "CROSSVAL_WORKERS".to_string(),
            env::var("CROSSVAL_WORKERS").unwrap_or_else(|_| "2".to_string()),
        ),
        (
            "OMP_NUM_THREADS".to_string(),
            env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "1".to_string()),
        ),
        (
            "OPENBLAS_NUM_THREADS".to_string(),
            env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "1".to_string()),
        ),
        (
            "MKL_NUM_THREADS".to_string(),
            env::var("MKL_NUM_THREADS").unwrap_or_else(|_| "1".to_string()),
        ),
        (
            "NUMEXPR_NUM_THREADS".to_string(),
            env::var("NUMEXPR_NUM_THREADS").unwrap_or_else(|_| "1".to_string()),
        ),
        (
            "BITNET_DETERMINISTIC".to_string(),
            env::var("BITNET_DETERMINISTIC").unwrap_or_else(|_| "1".to_string()),
        ),
        ("BITNET_SEED".to_string(), env::var("BITNET_SEED").unwrap_or_else(|_| "42".to_string())),
    ];

    if pids_pct > 85 {
        for env_name in [
            "RUST_TEST_THREADS",
            "RAYON_NUM_THREADS",
            "CROSSVAL_WORKERS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ] {
            for (name, value) in &mut envs {
                if name == env_name {
                    *value = "1".to_string();
                }
            }
        }
        println!("⚠️  System hot ({pids_pct}% PID usage) → auto-degraded to single-threaded mode");
    } else {
        println!(
            "✅ System resources OK → using capped concurrency (RUST_TEST_THREADS={}, RAYON={} )",
            envs[0].1, envs[1].1,
        );
    }

    println!("=== BitNet-rs Concurrency Configuration ===");
    for (name, value) in &envs {
        println!("{name}={value}");
    }
    println!("========================================");

    Ok(envs)
}

fn env_refs_from_pairs(envs: &[(String, String)]) -> Vec<(&str, &str)> {
    envs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
}

fn require_tests(root: &Path, expr: &str, envs: &[(&str, &str)]) -> Result<usize> {
    let output =
        run_capture(root, "cargo", &["nextest", "list", "-E", expr, "--workspace"], envs, false)?;
    if !output.status.success() {
        bail!("cargo nextest list failed for filter {expr}");
    }

    let mut count = 0usize;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.contains("::") {
            count += 1;
        }
    }

    if count == 0 {
        bail!("no tests discovered for filter: {expr}");
    }

    println!("discovered {count} tests for: {expr}");
    Ok(count)
}

#[allow(clippy::too_many_arguments)]
fn run_inference_with_seed(
    root: &Path,
    binary: &Path,
    model: &str,
    tokenizer: Option<&str>,
    prompt: &str,
    max_tokens: u32,
    seed: u64,
    output: &Path,
    deterministic: bool,
    greedy: bool,
) -> Result<()> {
    run_inference_with_opts(
        root,
        binary,
        model,
        tokenizer,
        prompt,
        max_tokens,
        seed,
        output,
        deterministic,
        greedy,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn run_inference_with_opts(
    root: &Path,
    binary: &Path,
    model: &str,
    tokenizer: Option<&str>,
    prompt: &str,
    max_tokens: u32,
    seed: u64,
    output: &Path,
    deterministic: bool,
    greedy: bool,
    opts: &[&str],
) -> Result<()> {
    let bin =
        binary.to_str().with_context(|| format!("non-utf8 bitnet path: {}", binary.display()))?;
    let out =
        output.to_str().with_context(|| format!("non-utf8 json path: {}", output.display()))?;

    let max_tokens = max_tokens.to_string();
    let seed = seed.to_string();
    let mut args = vec![
        "run",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-new-tokens",
        max_tokens.as_str(),
        "--seed",
        seed.as_str(),
        "--json-out",
        out,
    ];
    if deterministic {
        args.push("--deterministic");
    }
    if greedy {
        args.push("--greedy");
    }
    if let Some(tokenizer) = tokenizer {
        args.push("--tokenizer");
        args.push(tokenizer);
    }
    args.extend_from_slice(opts);

    let output = run_capture(root, bin, &args, &[], true)?;
    if !output.status.success() {
        bail!("inference command failed");
    }
    Ok(())
}

fn read_inference_text(path: &Path) -> Result<String> {
    let data =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let value: Value = serde_json::from_str(&data).context("invalid inference JSON")?;
    Ok(value.get("text").and_then(Value::as_str).unwrap_or("<missing text>").to_string())
}

fn check_benchmark_smoke(path: &Path) -> Result<()> {
    let data = fs::read_to_string(path).context("failed to read benchmark json")?;
    let value: Value = serde_json::from_str(&data).context("invalid benchmark JSON")?;
    let success = value.pointer("/success").and_then(Value::as_bool).unwrap_or(false);
    let total_ms = value.pointer("/timing/total_ms").and_then(Value::as_u64).unwrap_or(1);
    if !success || total_ms != 0 {
        bail!("benchmark short-circuit expectation failed");
    }
    Ok(())
}

fn command_available(command: &str) -> bool {
    if command.contains(std::path::MAIN_SEPARATOR) {
        return is_executable_file(Path::new(command));
    }
    command_available_in_path(command, env::var_os("PATH"))
}

fn command_available_in_path(command: &str, path: Option<std::ffi::OsString>) -> bool {
    let Some(path) = path else {
        return false;
    };
    let exts: &[&str] = if cfg!(windows) { &[".com", ".exe", ".bat", ".cmd", ""] } else { &[""] };
    for dir in env::split_paths(&path) {
        for ext in exts {
            let candidate = dir.join(format!("{command}{ext}"));
            if is_executable_file(&candidate) {
                return true;
            }
        }
    }
    false
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

struct ProcessKiller(Option<std::process::Child>);

impl Drop for ProcessKiller {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn collect_rust_files(root: PathBuf) -> Result<Vec<PathBuf>> {
    let mut stack = vec![root];
    let mut files = Vec::new();

    while let Some(dir) = stack.pop() {
        if dir.file_name().is_some_and(|name| name == "target") {
            continue;
        }
        let entries =
            fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
    }
    Ok(files)
}

fn print_matching_lines(
    output: &Output,
    patterns: &[&str],
    print_all_if_empty: bool,
) -> Result<()> {
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let mut printed = false;
    for line in text.lines() {
        if patterns.iter().any(|pattern| line.contains(pattern)) {
            println!("{line}");
            printed = true;
        }
    }

    if !printed && print_all_if_empty {
        for line in text.lines().take(5) {
            println!("{line}");
        }
    }
    Ok(())
}

fn contains_word_like_sequence(text: &str, min_len: usize) -> bool {
    let mut run = 0usize;
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() {
            run += 1;
            if run >= min_len {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

fn has_shift_constant(s: &str, suffix: &str) -> bool {
    let mut index = 0;
    while let Some(pos) = s[index..].find(suffix) {
        let shift_pos = index + pos;
        let mut token_start = 0usize;
        for i in (0..shift_pos).rev() {
            let c = s.as_bytes()[i];
            if !c.is_ascii_alphanumeric() && c != b'_' {
                token_start = i + 1;
                break;
            }
            token_start = 0;
        }
        let token = &s[token_start..shift_pos];
        if !token.is_empty() {
            let mut chars = token.chars();
            if chars.next().is_some_and(|c| c == '1')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return true;
            }
        }
        index = shift_pos + suffix.len();
    }
    false
}

fn relevant_flake_output(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    for (line_no, line) in lines.iter().enumerate() {
        if line.contains("test_cross_crate") {
            let start = line_no;
            let end = (line_no + 3).min(lines.len());
            return lines
                .iter()
                .skip(start)
                .take(end - start)
                .map(|line| line.to_string())
                .collect();
        }
    }

    lines.iter().take(5).map(|line| line.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::{command_available, command_available_in_path, is_executable_file};
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::{ffi::OsString, fs};

    #[cfg(unix)]
    fn make_executable(path: &std::path::Path) {
        let mut permissions = fs::metadata(path).expect("metadata").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).expect("set executable bit");
    }

    #[test]
    fn command_available_ignores_directories_in_path() {
        let temp = tempfile::tempdir().expect("create temp dir");
        let fake_cmd = temp.path().join("fakecmd");
        fs::create_dir_all(&fake_cmd).expect("create fake command directory");
        let found = command_available_in_path("fakecmd", Some(OsString::from(temp.path())));

        assert!(!found, "directory entries should not be treated as commands");
    }

    #[test]
    fn is_executable_file_rejects_directories() {
        let temp = tempfile::tempdir().expect("create temp dir");
        assert!(!is_executable_file(temp.path()));
    }

    #[test]
    fn command_available_finds_executable_files_in_path() {
        let temp = tempfile::tempdir().expect("create temp dir");
        let fake_cmd = temp.path().join("fakecmd");
        fs::write(&fake_cmd, b"#!/bin/sh\n").expect("write fake command");
        #[cfg(unix)]
        make_executable(&fake_cmd);

        let found = command_available_in_path("fakecmd", Some(OsString::from(temp.path())));
        assert!(found, "executable files on PATH should be treated as commands");
    }

    #[cfg(unix)]
    #[test]
    fn command_available_rejects_non_executable_files_in_path() {
        let temp = tempfile::tempdir().expect("create temp dir");
        let fake_cmd = temp.path().join("fakecmd");
        fs::write(&fake_cmd, b"not executable").expect("write fake command");

        let found = command_available_in_path("fakecmd", Some(OsString::from(temp.path())));
        assert!(!found, "non-executable files on PATH should not be treated as commands");
    }

    #[test]
    fn command_available_checks_explicit_paths_as_files() {
        let temp = tempfile::tempdir().expect("create temp dir");
        let fake_cmd_dir = temp.path().join("fakecmd");
        fs::create_dir_all(&fake_cmd_dir).expect("create fake command directory");

        assert!(
            !command_available(fake_cmd_dir.to_str().expect("utf-8 path")),
            "explicit directory paths should not be treated as commands"
        );
    }
}
