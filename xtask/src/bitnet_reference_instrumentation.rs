use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFAULT_REFERENCE_ROOT: &str = "target/external/BitNet-reference";
const DEFAULT_CPP_ROOT: &str = "target/external/BitNet-reference/3rdparty/llama.cpp";
const DEFAULT_PATCH: &str = "ci/reference-instrumentation/bitnet-rs-first-token-logits-main.patch";
const DEFAULT_OUTPUT: &str = "target/a770-diagnostic/bitnet-reference-instrumentation-plan.json";
const DEFAULT_RUN_OUTPUT: &str = "target/a770-diagnostic/bitnet-reference-instrumentation-run.json";
const DEFAULT_REFERENCE_PLAN: &str = "target/a770-diagnostic/bitnet-reference-plan.json";
const DEFAULT_SIDECAR: &str = "target/a770-diagnostic/reference-first-token-logits.json";

const REQUIRED_METADATA: &[&str] =
    &["Upstream-Issue", "Reason", "Status", "Created", "Review-By", "Author"];

const REQUIRED_ANCHORS: &[&str] = &[
    "BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS",
    "llama_get_logits_ith(ctx, -1)",
    "common_sampler_sample(smpl, ctx, -1)",
    "const llama_token probe_tokens[] = {17, 58428};",
    "probe_logits",
];

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
    "reference_raw_logits",
    "rust_reference_parity_proven",
    "a770_semantic_quality_proven",
];

const RUN_NOT_CLAIMS: &[&str] = &[
    "selected_attention_residency",
    "resident_kv_decode",
    "attention_scores_residency",
    "softmax_residency",
    "attention_value_mix_residency",
    "full_support_op_residency",
    "full_device_residency",
    "completion",
    "reference_generated_token_ids",
    "reference_logits_match_rust_cpu",
    "reference_logits_match_strict_a770",
    "rust_reference_parity_proven",
    "a770_semantic_quality_proven",
];

#[derive(Debug)]
struct ReferenceInstrumentationArgs {
    cpp_root: PathBuf,
    patch: PathBuf,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct ReferenceInstrumentationRunArgs {
    reference_root: PathBuf,
    cpp_root: PathBuf,
    patch: PathBuf,
    plan: PathBuf,
    sidecar: PathBuf,
    output: Option<PathBuf>,
    format: String,
}

#[derive(Debug)]
struct CommandCapture {
    status_code: Option<i32>,
    success: bool,
    stdout: String,
    stderr: String,
}

pub fn maybe_dispatch_from_env() -> Result<bool> {
    let args = std::env::args().collect::<Vec<_>>();
    maybe_dispatch(&args)
}

fn maybe_dispatch(args: &[String]) -> Result<bool> {
    match args.get(1).map(String::as_str) {
        Some("bitnet-reference-instrumentation-plan") => {
            if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_help();
                return Ok(true);
            }
            let opts = parse_args(args)?;
            let report = build_plan(&opts)?;
            if let Some(output) = &opts.output {
                if let Some(parent) = output.parent() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
                fs::write(output, serde_json::to_vec_pretty(&report)?)
                    .with_context(|| format!("writing {}", output.display()))?;
            }
            emit_report(&report, &opts.format)?;
            Ok(true)
        }
        Some("bitnet-reference-instrumentation-run") => {
            if args[2..].iter().any(|arg| arg == "-h" || arg == "--help") {
                print_run_help();
                return Ok(true);
            }
            let opts = parse_run_args(args)?;
            let report = run_instrumented_reference(&opts)?;
            if let Some(output) = &opts.output {
                if let Some(parent) = output.parent() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
                fs::write(output, serde_json::to_vec_pretty(&report)?)
                    .with_context(|| format!("writing {}", output.display()))?;
            }
            emit_report(&report, &opts.format)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn print_help() {
    println!(
        "Verify the target-local BitNet reference first-token logit instrumentation patch\n\nUsage: xtask.exe bitnet-reference-instrumentation-plan [OPTIONS]\n\nOptions:\n      --cpp-root <PATH>  llama.cpp checkout root [default: target/external/BitNet-reference/3rdparty/llama.cpp]\n      --patch <PATH>     Instrumentation patch [default: ci/reference-instrumentation/bitnet-rs-first-token-logits-main.patch]\n      --output <PATH>    Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-instrumentation-plan.json]\n      --format <FORMAT>  Output format: human or json [default: human]\n  -h, --help             Print help"
    );
}

fn print_run_help() {
    println!(
        "Temporarily apply the BitNet reference first-token logit instrumentation, run the matched reference plan, and restore source worktrees\n\nUsage: xtask.exe bitnet-reference-instrumentation-run [OPTIONS]\n\nOptions:\n      --reference-root <PATH>  BitNet.cpp checkout root [default: target/external/BitNet-reference]\n      --cpp-root <PATH>        llama.cpp checkout root [default: target/external/BitNet-reference/3rdparty/llama.cpp]\n      --patch <PATH>           Instrumentation patch [default: ci/reference-instrumentation/bitnet-rs-first-token-logits-main.patch]\n      --plan <PATH>            Reference plan JSON [default: target/a770-diagnostic/bitnet-reference-plan.json]\n      --sidecar <PATH>         First-token logit sidecar JSON [default: target/a770-diagnostic/reference-first-token-logits.json]\n      --output <PATH>          Output JSON receipt [default: target/a770-diagnostic/bitnet-reference-instrumentation-run.json]\n      --format <FORMAT>        Output format: human or json [default: human]\n  -h, --help                   Print help"
    );
}

fn parse_args(args: &[String]) -> Result<ReferenceInstrumentationArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-instrumentation-plan") {
        bail!("parse_args called for unexpected command");
    }
    let mut cpp_root = PathBuf::from(DEFAULT_CPP_ROOT);
    let mut patch = PathBuf::from(DEFAULT_PATCH);
    let mut output = Some(PathBuf::from(DEFAULT_OUTPUT));
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
            "--cpp-root" => cpp_root = PathBuf::from(value()?),
            "--patch" => patch = PathBuf::from(value()?),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-instrumentation-plan option {other}"),
        }
    }
    Ok(ReferenceInstrumentationArgs { cpp_root, patch, output, format })
}

fn parse_run_args(args: &[String]) -> Result<ReferenceInstrumentationRunArgs> {
    if args.get(1).map(String::as_str) != Some("bitnet-reference-instrumentation-run") {
        bail!("parse_run_args called for unexpected command");
    }
    let mut reference_root = PathBuf::from(DEFAULT_REFERENCE_ROOT);
    let mut cpp_root = PathBuf::from(DEFAULT_CPP_ROOT);
    let mut patch = PathBuf::from(DEFAULT_PATCH);
    let mut plan = PathBuf::from(DEFAULT_REFERENCE_PLAN);
    let mut sidecar = PathBuf::from(DEFAULT_SIDECAR);
    let mut output = Some(PathBuf::from(DEFAULT_RUN_OUTPUT));
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
            "--reference-root" => reference_root = PathBuf::from(value()?),
            "--cpp-root" => cpp_root = PathBuf::from(value()?),
            "--patch" => patch = PathBuf::from(value()?),
            "--plan" => plan = PathBuf::from(value()?),
            "--sidecar" => sidecar = PathBuf::from(value()?),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--format" => format = value()?,
            other => bail!("unknown bitnet-reference-instrumentation-run option {other}"),
        }
    }
    Ok(ReferenceInstrumentationRunArgs {
        reference_root,
        cpp_root,
        patch,
        plan,
        sidecar,
        output,
        format,
    })
}

fn run_instrumented_reference(args: &ReferenceInstrumentationRunArgs) -> Result<Value> {
    let reference_root = normalize_path(&args.reference_root)?;
    let cpp_root = normalize_path(&args.cpp_root)?;
    let patch = normalize_path(&args.patch)?;
    let plan_path = normalize_path(&args.plan)?;
    let sidecar = normalize_path(&args.sidecar)?;
    let build_dir = reference_root.join("build");
    let selected_exe = build_dir.join("bin").join(exe_name("llama-cli"));
    let generated_lut_header = reference_root.join("include/bitnet-lut-kernels.h");
    let generated_kernel_config = reference_root.join("include/kernel_config.ini");
    let plan_result = read_json(&plan_path);
    let plan_read_success = plan_result.is_ok();
    let plan = plan_result.unwrap_or(Value::Null);
    let reference_argv = reference_argv(&plan).unwrap_or_default();

    if let Some(parent) = sidecar.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if sidecar.exists() {
        fs::remove_file(&sidecar)
            .with_context(|| format!("removing stale {}", sidecar.display()))?;
    }

    let reference_status_before = git_status(&reference_root);
    let cpp_status_before = git_status(&cpp_root);
    let clean_before = capture_success_empty(&reference_status_before)
        && capture_success_empty(&cpp_status_before);

    let mut blocked_reasons = Vec::<String>::new();
    if !reference_root.is_dir() {
        blocked_reasons.push("reference_root_missing".to_string());
    }
    if !cpp_root.is_dir() {
        blocked_reasons.push("reference_llama_cpp_root_missing".to_string());
    }
    if !patch.is_file() {
        blocked_reasons.push("reference_instrumentation_patch_missing".to_string());
    }
    if !plan_path.is_file() {
        blocked_reasons.push("reference_plan_missing".to_string());
    }
    if plan_path.is_file() && !plan_read_success {
        blocked_reasons.push("reference_plan_json_invalid".to_string());
    }
    if plan_path.is_file() && reference_argv.is_empty() {
        blocked_reasons.push("reference_plan_command_argv_missing".to_string());
    }
    if !build_dir.is_dir() {
        blocked_reasons.push("reference_build_dir_missing".to_string());
    }
    if !clean_before {
        blocked_reasons.push("reference_external_worktree_not_clean_before_run".to_string());
    }

    let mut compatibility = Vec::<Value>::new();
    let mut codegen_capture = None;
    let mut patch_apply = None;
    let mut build_capture = None;
    let mut run_capture = None;
    let cleanup_capture;

    let generated_lut_header_exists_before = generated_lut_header.is_file();
    let generated_kernel_config_exists_before = generated_kernel_config.is_file();
    let mut generated_lut_header_exists_after_codegen = generated_lut_header_exists_before;
    let mut generated_kernel_config_exists_after_codegen = generated_kernel_config_exists_before;
    if blocked_reasons.is_empty() && !generated_lut_header_exists_before {
        codegen_capture = Some(run_reference_kernel_codegen(&reference_root)?);
        generated_lut_header_exists_after_codegen = generated_lut_header.is_file();
        generated_kernel_config_exists_after_codegen = generated_kernel_config.is_file();
        if !codegen_capture.as_ref().is_some_and(|capture| capture.success) {
            blocked_reasons.push("reference_kernel_codegen_failed".to_string());
        }
        if !generated_lut_header.is_file() {
            blocked_reasons.push("reference_generated_lut_header_missing".to_string());
        }
    }

    if blocked_reasons.is_empty() {
        compatibility = apply_windows_reference_compatibility_fixes(&reference_root)?;
        patch_apply = Some(run_git(&cpp_root, &["apply", &path_to_string(&patch)])?);
        if !patch_apply.as_ref().is_some_and(|capture| capture.success) {
            blocked_reasons.push("reference_instrumentation_patch_apply_failed".to_string());
        }
    }

    if blocked_reasons.is_empty() {
        build_capture = Some(build_reference_cli(&reference_root, &build_dir)?);
        if !build_capture.as_ref().is_some_and(|capture| capture.success) {
            blocked_reasons.push("reference_instrumented_build_failed".to_string());
        }
    }

    if blocked_reasons.is_empty() {
        if !selected_exe.is_file() {
            blocked_reasons.push("reference_instrumented_executable_missing".to_string());
        } else if reference_argv.is_empty() {
            blocked_reasons.push("reference_plan_command_argv_missing".to_string());
        } else {
            let mut argv = reference_argv.clone();
            argv[0] = path_to_string(&selected_exe);
            run_capture = Some(run_reference_with_sidecar(&argv, &sidecar)?);
            if !run_capture.as_ref().is_some_and(|capture| capture.success) {
                blocked_reasons.push("reference_instrumented_run_failed".to_string());
            }
        }
    }

    let sidecar_value = if sidecar.is_file() { Some(read_json(&sidecar)?) } else { None };
    if run_capture.as_ref().is_some_and(|capture| capture.success) && sidecar_value.is_none() {
        blocked_reasons.push("reference_first_token_logit_sidecar_missing".to_string());
    }

    cleanup_capture = if reference_root.is_dir() && cpp_root.is_dir() {
        Some(cleanup_reference_sources(
            &reference_root,
            &cpp_root,
            &generated_lut_header,
            generated_lut_header_exists_before,
            &generated_kernel_config,
            generated_kernel_config_exists_before,
        )?)
    } else {
        None
    };
    let reference_status_after = git_status(&reference_root);
    let cpp_status_after = git_status(&cpp_root);
    let clean_after =
        capture_success_empty(&reference_status_after) && capture_success_empty(&cpp_status_after);
    if !clean_after {
        blocked_reasons.push("reference_external_worktree_not_clean_after_run".to_string());
    }

    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();
    let reference_raw_logits_available = sidecar_value.is_some() && blocked_reasons.is_empty();

    Ok(json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_instrumentation_run",
        "diagnostic": "bitnet_reference_instrumentation_run",
        "producer": "cargo xtask bitnet-reference-instrumentation-run",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "paths": {
            "reference_root": path_to_string(&reference_root),
            "cpp_root": path_to_string(&cpp_root),
            "patch": path_to_string(&patch),
            "plan": path_to_string(&plan_path),
            "build_dir": path_to_string(&build_dir),
            "selected_executable": path_to_string(&selected_exe),
            "sidecar": path_to_string(&sidecar),
        },
        "preflight": {
            "reference_root_exists": reference_root.is_dir(),
            "cpp_root_exists": cpp_root.is_dir(),
            "patch_exists": patch.is_file(),
            "plan_exists": plan_path.is_file(),
            "build_dir_exists": build_dir.is_dir(),
            "external_worktrees_clean_before_run": clean_before,
            "generated_lut_header": {
                "path": path_to_string(&generated_lut_header),
                "exists_before": generated_lut_header_exists_before,
                "exists_after_codegen": generated_lut_header_exists_after_codegen,
            },
            "generated_kernel_config": {
                "path": path_to_string(&generated_kernel_config),
                "exists_before": generated_kernel_config_exists_before,
                "exists_after_codegen": generated_kernel_config_exists_after_codegen,
            },
            "reference_status_before": capture_json(reference_status_before.as_ref()),
            "cpp_status_before": capture_json(cpp_status_before.as_ref()),
        },
        "kernel_codegen": capture_json(codegen_capture.as_ref()),
        "compatibility_fixes": compatibility,
        "patch_apply": capture_json(patch_apply.as_ref()),
        "build": capture_json(build_capture.as_ref()),
        "reference_run": capture_json(run_capture.as_ref()),
        "sidecar": {
            "exists": sidecar.is_file(),
            "sha256": sidecar.is_file().then(|| sha256_bytes(&fs::read(&sidecar).unwrap_or_default())),
            "receipt": sidecar_value,
            "policy": "reference-side first-token raw logits are diagnostic evidence only until compared with Rust CPU and strict A770 receipts",
        },
        "cleanup": {
            "source_restore": capture_json(cleanup_capture.as_ref()),
            "external_worktrees_clean_after_run": clean_after,
            "reference_status_after": capture_json(reference_status_after.as_ref()),
            "cpp_status_after": capture_json(cpp_status_after.as_ref()),
        },
        "decision": {
            "reference_raw_logits_available": reference_raw_logits_available,
            "current_blocked_reasons": blocked_reasons,
            "next_when_available": "compare reference first-token logits for token 17 and token 58428 against Rust CPU and strict A770 first-token receipts",
        },
        "not_claims": RUN_NOT_CLAIMS,
    }))
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn reference_argv(plan: &Value) -> Result<Vec<String>> {
    plan.pointer("/reference/command_argv")
        .and_then(Value::as_array)
        .context("plan missing /reference/command_argv")?
        .iter()
        .map(|item| {
            item.as_str().map(ToOwned::to_owned).context("command_argv item is not a string")
        })
        .collect()
}

fn apply_windows_reference_compatibility_fixes(reference_root: &Path) -> Result<Vec<Value>> {
    let mut applied = Vec::new();
    applied.push(replace_file_text(
        &reference_root.join("src/ggml-bitnet-mad.cpp"),
        "        int8_t * y_col = y + col * by;",
        "        const int8_t * y_col = y + col * by;",
        "windows_const_compatibility",
    )?);
    applied.push(insert_after_if_missing(
        &reference_root.join("3rdparty/llama.cpp/common/common.cpp"),
        "#include <ctime>",
        "#include <chrono>",
        "windows_common_chrono_include",
    )?);
    applied.push(insert_after_if_missing(
        &reference_root.join("3rdparty/llama.cpp/common/log.cpp"),
        "#include <condition_variable>",
        "#include <chrono>",
        "windows_log_chrono_include",
    )?);
    Ok(applied)
}

fn run_reference_kernel_codegen(reference_root: &Path) -> Result<CommandCapture> {
    run_command(Command::new("python").current_dir(reference_root).args([
        "utils/codegen_tl2.py",
        "--model",
        "bitnet_b1_58-3B",
        "--BM",
        "160,320,320",
        "--BK",
        "96,96,96",
        "--bm",
        "32,32,32",
    ]))
}

fn replace_file_text(path: &Path, before: &str, after: &str, fix_id: &str) -> Result<Value> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    if content.contains(before) {
        fs::write(path, content.replace(before, after))
            .with_context(|| format!("writing {}", path.display()))?;
        return Ok(json!({
            "fix_id": fix_id,
            "path": path_to_string(path),
            "applied": true,
            "already_present": false,
        }));
    }
    if content.contains(after) {
        return Ok(json!({
            "fix_id": fix_id,
            "path": path_to_string(path),
            "applied": false,
            "already_present": true,
        }));
    }
    bail!("expected compatibility fix target not found in {}", path.display())
}

fn insert_after_if_missing(path: &Path, anchor: &str, insert: &str, fix_id: &str) -> Result<Value> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    if content.contains(insert) {
        return Ok(json!({
            "fix_id": fix_id,
            "path": path_to_string(path),
            "applied": false,
            "already_present": true,
        }));
    }
    if !content.contains(anchor) {
        bail!("expected compatibility include anchor not found in {}", path.display());
    }
    fs::write(path, content.replace(anchor, &format!("{anchor}\n{insert}")))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(json!({
        "fix_id": fix_id,
        "path": path_to_string(path),
        "applied": true,
        "already_present": false,
    }))
}

fn build_reference_cli(reference_root: &Path, build_dir: &Path) -> Result<CommandCapture> {
    if cfg!(windows) {
        let vsdevcmd = find_vsdevcmd();
        let Some(vsdevcmd) = vsdevcmd else {
            return Ok(CommandCapture {
                status_code: None,
                success: false,
                stdout: String::new(),
                stderr: "Visual Studio developer command prompt not found".to_string(),
            });
        };
        let script = reference_root.join("build_bitnet_rs_instrumented_reference.cmd");
        let lines = [
            "@echo off".to_string(),
            format!("call {} -arch=x64 -host_arch=x64 || exit /b 1", cmd_quote(&vsdevcmd)),
            format!(
                "cmake --build {} --config Release --target llama-cli || exit /b 1",
                cmd_quote(build_dir)
            ),
        ];
        fs::write(&script, lines.join("\r\n"))
            .with_context(|| format!("writing {}", script.display()))?;
        let capture = run_command(Command::new("cmd.exe").args(["/d", "/c"]).arg(&script))?;
        let _ = fs::remove_file(&script);
        Ok(capture)
    } else {
        run_command(Command::new("cmake").args([
            "--build",
            &path_to_string(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-cli",
        ]))
    }
}

fn find_vsdevcmd() -> Option<PathBuf> {
    let program_files_x86 = std::env::var_os("ProgramFiles(x86)")?;
    let vswhere =
        PathBuf::from(program_files_x86).join("Microsoft Visual Studio/Installer/vswhere.exe");
    if !vswhere.is_file() {
        return None;
    }
    let output = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        return None;
    }
    let vsdevcmd = PathBuf::from(path).join("Common7/Tools/VsDevCmd.bat");
    vsdevcmd.is_file().then_some(vsdevcmd)
}

fn run_reference_with_sidecar(argv: &[String], sidecar: &Path) -> Result<CommandCapture> {
    let executable = argv.first().context("empty reference command")?;
    let mut command = Command::new(executable);
    command
        .args(&argv[1..])
        .env("BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS", sidecar)
        .stdin(Stdio::null());
    run_command(&mut command)
}

fn run_command(command: &mut Command) -> Result<CommandCapture> {
    let output = command.output().with_context(|| format!("running command {:?}", command))?;
    Ok(CommandCapture {
        status_code: output.status.code(),
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn cleanup_reference_sources(
    reference_root: &Path,
    cpp_root: &Path,
    generated_lut_header: &Path,
    generated_lut_header_existed_before: bool,
    generated_kernel_config: &Path,
    generated_kernel_config_existed_before: bool,
) -> Result<CommandCapture> {
    let mut capture = run_git(reference_root, &["restore", "--", "src/ggml-bitnet-mad.cpp"])?;
    let cpp_capture = run_git(
        cpp_root,
        &["restore", "--", "common/common.cpp", "common/log.cpp", "examples/main/main.cpp"],
    )?;
    if !generated_lut_header_existed_before && generated_lut_header.exists() {
        fs::remove_file(generated_lut_header)
            .with_context(|| format!("removing {}", generated_lut_header.display()))?;
    }
    if !generated_kernel_config_existed_before && generated_kernel_config.exists() {
        fs::remove_file(generated_kernel_config)
            .with_context(|| format!("removing {}", generated_kernel_config.display()))?;
    }
    capture.success = capture.success && cpp_capture.success;
    capture.status_code = if capture.status_code == Some(0) && cpp_capture.status_code == Some(0) {
        Some(0)
    } else {
        cpp_capture.status_code.or(capture.status_code)
    };
    if !cpp_capture.stdout.is_empty() {
        capture.stdout.push_str(&cpp_capture.stdout);
    }
    if !cpp_capture.stderr.is_empty() {
        capture.stderr.push_str(&cpp_capture.stderr);
    }
    Ok(capture)
}

fn git_status(path: &Path) -> Option<CommandCapture> {
    path.is_dir().then(|| run_git(path, &["status", "--porcelain"]).ok()).flatten()
}

fn capture_success_empty(capture: &Option<CommandCapture>) -> bool {
    capture.as_ref().is_some_and(|capture| capture.success && capture.stdout.trim().is_empty())
}

fn capture_json(capture: Option<&CommandCapture>) -> Value {
    match capture {
        Some(capture) => json!({
            "success": capture.success,
            "exit_code": capture.status_code,
            "stdout": capture.stdout.trim(),
            "stderr": capture.stderr.trim(),
        }),
        None => Value::Null,
    }
}

fn exe_name(stem: &str) -> String {
    if cfg!(windows) { format!("{stem}.exe") } else { stem.to_string() }
}

fn cmd_quote(path: &Path) -> String {
    format!("\"{}\"", path.display().to_string().replace('"', "\"\""))
}

fn build_plan(args: &ReferenceInstrumentationArgs) -> Result<Value> {
    let cpp_root = normalize_path(&args.cpp_root)?;
    let patch = normalize_path(&args.patch)?;
    let patch_text = fs::read_to_string(&patch).unwrap_or_default();
    let metadata = patch_metadata(&patch_text);
    let missing_metadata = missing_metadata(&metadata);
    let missing_anchors = missing_patch_anchors(&patch_text);
    let source_target = cpp_root.join("examples/main/main.cpp");
    let git_status = if cpp_root.is_dir() {
        Some(run_git(&cpp_root, &["status", "--porcelain"])?)
    } else {
        None
    };
    let apply_check = if cpp_root.is_dir() && patch.is_file() {
        Some(run_git(&cpp_root, &["apply", "--check", &path_to_string(&patch)])?)
    } else {
        None
    };
    let git_clean = git_status.as_ref().is_some_and(|capture| {
        capture.success && capture.stdout.trim().is_empty() && capture.stderr.trim().is_empty()
    });
    let patch_applies = apply_check.as_ref().is_some_and(|capture| capture.success);
    let ready = cpp_root.is_dir()
        && patch.is_file()
        && source_target.is_file()
        && git_clean
        && patch_applies
        && missing_metadata.is_empty()
        && missing_anchors.is_empty();
    let mut blocked_reasons = Vec::new();
    if !cpp_root.is_dir() {
        blocked_reasons.push("reference_llama_cpp_root_missing".to_string());
    }
    if !source_target.is_file() {
        blocked_reasons.push("reference_llama_main_source_missing".to_string());
    }
    if !patch.is_file() {
        blocked_reasons.push("reference_instrumentation_patch_missing".to_string());
    }
    if !git_clean {
        blocked_reasons.push("reference_llama_cpp_worktree_not_clean".to_string());
    }
    if !patch_applies {
        blocked_reasons.push("reference_instrumentation_patch_apply_check_failed".to_string());
    }
    if !missing_metadata.is_empty() {
        blocked_reasons.push("reference_instrumentation_patch_metadata_incomplete".to_string());
    }
    if !missing_anchors.is_empty() {
        blocked_reasons.push("reference_instrumentation_patch_anchor_missing".to_string());
    }
    blocked_reasons.sort_unstable();
    blocked_reasons.dedup();

    Ok(json!({
        "schema_version": 1,
        "receipt_type": "bitnet_reference_instrumentation_plan",
        "diagnostic": "bitnet_reference_instrumentation_plan",
        "producer": "cargo xtask bitnet-reference-instrumentation-plan",
        "created_at": chrono::Utc::now().to_rfc3339(),
        "diagnostic_only": true,
        "promotion_allowed": false,
        "claim_allowed": false,
        "classification": "diagnostic_only",
        "cpp_root": {
            "path": path_to_string(&cpp_root),
            "exists": cpp_root.is_dir(),
            "main_source": path_to_string(&source_target),
            "main_source_exists": source_target.is_file(),
            "git_status_clean": git_clean,
            "git_status_stdout": git_status.as_ref().map(|capture| capture.stdout.trim().to_string()),
            "git_status_stderr": git_status.as_ref().map(|capture| capture.stderr.trim().to_string()),
        },
        "patch": {
            "path": path_to_string(&patch),
            "exists": patch.is_file(),
            "sha256": patch.is_file().then(|| sha256_bytes(&fs::read(&patch).unwrap_or_default())),
            "metadata": metadata,
            "missing_metadata": missing_metadata,
            "required_anchors": REQUIRED_ANCHORS,
            "missing_anchors": missing_anchors,
            "apply_check_success": patch_applies,
            "apply_check_exit_code": apply_check.as_ref().and_then(|capture| capture.status_code),
            "apply_check_stdout": apply_check.as_ref().map(|capture| capture.stdout.trim().to_string()),
            "apply_check_stderr": apply_check.as_ref().map(|capture| capture.stderr.trim().to_string()),
            "default_applied": false,
            "policy": "target-local diagnostic instrumentation patch; not stored under patches/ and not applied by default fetch scripts",
        },
        "instrumentation": {
            "environment_variable": "BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS",
            "receipt_type_when_applied": "bitnet_reference_first_token_logits",
            "probe_token_ids": [17, 58428],
            "top_k": 16,
            "captures": [
                "prompt_token_count",
                "n_vocab",
                "probe token logits and probabilities",
                "top logits for first generated token"
            ],
            "not_claim": "instrumentation output is reference-side diagnostic evidence only until compared against Rust CPU and strict A770 receipts",
        },
        "operator_commands": {
            "apply_patch": format!("git -C {} apply {}", path_to_string(&cpp_root), path_to_string(&patch)),
            "rebuild_reference": "cmake --build target/external/BitNet-reference/build --config Release --target llama-cli",
            "run_with_receipt_env": "set BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS=target/a770-diagnostic/reference-first-token-logits.json before running bitnet-reference-run",
        },
        "decision": {
            "instrumentation_ready_to_apply": ready,
            "reference_raw_logits_available": false,
            "current_blocked_reasons": blocked_reasons,
            "next_action": "apply instrumentation patch in the external reference worktree, rebuild llama-cli, run matched reference prompt with BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS, then compare token 17 and 58428 against Rust CPU and strict A770 logits",
        },
        "not_claims": CRITICAL_NOT_CLAIMS,
    }))
}

fn normalize_path(path: &Path) -> Result<PathBuf> {
    let path =
        if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
    Ok(path)
}

fn patch_metadata(text: &str) -> Value {
    let mut map = Map::new();
    for line in text.lines().take_while(|line| line.trim_start().starts_with('#')) {
        let line = line.trim_start().trim_start_matches('#').trim();
        if let Some((key, value)) = line.split_once(':') {
            map.insert(key.trim().to_string(), Value::String(value.trim().to_string()));
        }
    }
    Value::Object(map)
}

fn missing_metadata(metadata: &Value) -> Vec<String> {
    REQUIRED_METADATA
        .iter()
        .filter(|key| metadata.pointer(&format!("/{key}")).and_then(Value::as_str).is_none())
        .map(|key| (*key).to_string())
        .collect()
}

fn missing_patch_anchors(text: &str) -> Vec<String> {
    REQUIRED_ANCHORS
        .iter()
        .filter(|anchor| !text.contains(**anchor))
        .map(|anchor| (*anchor).to_string())
        .collect()
}

fn run_git(cwd: &Path, args: &[&str]) -> Result<CommandCapture> {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("running git {} in {}", args.join(" "), cwd.display()))?;
    Ok(CommandCapture {
        status_code: output.status.code(),
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn path_to_string(path: &Path) -> String {
    path.display().to_string()
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
            println!("diagnostic: bitnet_reference_instrumentation_plan");
            println!(
                "instrumentation_ready_to_apply: {}",
                value
                    .pointer("/decision/instrumentation_ready_to_apply")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            );
            if let Some(reasons) = value.pointer("/decision/current_blocked_reasons") {
                println!("blocked_reasons: {}", serde_json::to_string(reasons)?);
            }
            println!("not_claims: {}", serde_json::to_string(&value["not_claims"])?);
        }
        other => bail!("unsupported bitnet-reference-instrumentation-plan output format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_metadata_reads_header_fields() {
        let metadata = patch_metadata(
            "# Upstream-Issue: not-applicable\n# Reason: test\n\n--- a/file\n+++ b/file\n",
        );

        assert_eq!(metadata["Upstream-Issue"], json!("not-applicable"));
        assert_eq!(metadata["Reason"], json!("test"));
    }

    #[test]
    fn missing_metadata_reports_required_fields() {
        let metadata = json!({
            "Upstream-Issue": "not-applicable",
            "Reason": "test"
        });

        let missing = missing_metadata(&metadata);

        assert!(missing.contains(&"Status".to_string()));
        assert!(missing.contains(&"Created".to_string()));
        assert!(!missing.contains(&"Reason".to_string()));
    }

    #[test]
    fn anchor_check_requires_probe_tokens_and_env_name() {
        let missing = missing_patch_anchors(
            "BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS const llama_token probe_tokens[] = {17, 58428};",
        );

        assert!(!missing.contains(&"BITNET_RS_REFERENCE_FIRST_TOKEN_LOGITS".to_string()));
        assert!(!missing.contains(&"const llama_token probe_tokens[] = {17, 58428};".to_string()));
        assert!(missing.contains(&"llama_get_logits_ith(ctx, -1)".to_string()));
    }

    #[test]
    fn compatibility_replace_handles_mixed_existing_and_unfixed_lines() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("source.cpp");
        fs::write(
            &path,
            "const int8_t * y_col = y + col * by;\n        int8_t * y_col = y + col * by;\n",
        )
        .expect("write fixture");

        let report = replace_file_text(
            &path,
            "        int8_t * y_col = y + col * by;",
            "        const int8_t * y_col = y + col * by;",
            "test_fix",
        )
        .expect("replace");
        let content = fs::read_to_string(&path).expect("read fixture");

        assert_eq!(report["applied"], json!(true));
        assert!(!content.contains("        int8_t * y_col = y + col * by;"));
        assert!(content.contains("        const int8_t * y_col = y + col * by;"));
    }
}
