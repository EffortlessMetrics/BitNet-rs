use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};
use tempfile::TempDir;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("repo root")
}

fn bitnet_task_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_bitnet-task"))
}

fn run_bitnet_task(args: &[&str]) -> Output {
    Command::new(bitnet_task_bin()).args(args).output().expect("run bitnet-task")
}

fn assert_success(output: &Output) -> String {
    if !output.status.success() {
        panic!(
            "command failed with {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn assert_failure(output: &Output) -> String {
    if output.status.success() {
        panic!(
            "command unexpectedly succeeded\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn fake_tool_dir() -> (TempDir, PathBuf, PathBuf) {
    let temp = tempfile::tempdir().expect("tempdir");
    let bin_dir = temp.path().join("bin");
    fs::create_dir_all(&bin_dir).expect("create bin dir");

    let cargo_log = temp.path().join("cargo.log");
    let rustup_log = temp.path().join("rustup.log");

    write_stub(&bin_dir.join("cargo"), &cargo_log);
    write_stub(&bin_dir.join("rustup"), &rustup_log);

    (temp, cargo_log, rustup_log)
}

fn write_stub(path: &Path, log_path: &Path) {
    let script = format!(
        "#!/usr/bin/env bash\nset -euo pipefail\n{{\n  for arg in \"$@\"; do\n    printf 'arg=%s\\n' \"$arg\"\n  done\n  printf -- '--\\n'\n}} >> \"{}\"\n",
        log_path.display()
    );
    fs::write(path, script).expect("write stub");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path).expect("metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).expect("chmod");
    }
}

fn path_with_fake_tools(bin_dir: &Path) -> String {
    let current = std::env::var("PATH").unwrap_or_default();
    format!("{}:{current}", bin_dir.display())
}

fn read_invocations(log_path: &Path) -> Vec<Vec<String>> {
    let mut invocations = Vec::new();
    let mut current = Vec::new();
    let content = fs::read_to_string(log_path).unwrap_or_default();
    for line in content.lines() {
        if line == "--" {
            if !current.is_empty() {
                invocations.push(std::mem::take(&mut current));
            }
            continue;
        }
        if let Some(arg) = line.strip_prefix("arg=") {
            current.push(arg.to_string());
        }
    }
    if !current.is_empty() {
        invocations.push(current);
    }
    invocations
}

fn run_wrapper(script_name: &str, args: &[&str]) -> Vec<Vec<String>> {
    let repo_root = repo_root();
    let script = repo_root.join("scripts").join(script_name);
    let (temp, cargo_log, _rustup_log) = fake_tool_dir();
    let status = Command::new("bash")
        .arg(&script)
        .args(args)
        .current_dir(temp.path())
        .env("PATH", path_with_fake_tools(&temp.path().join("bin")))
        .status()
        .expect("run wrapper");
    assert!(status.success(), "wrapper failed: {}", script.display());
    read_invocations(&cargo_log)
}

#[test]
fn top_level_help_states_facade_boundary() {
    let output = run_bitnet_task(&["--help"]);
    let stdout = assert_success(&output);
    assert!(stdout.contains("Compatibility facade"));
    assert!(stdout.contains("xtask"));
    assert!(stdout.contains("generate-policy"));
}

#[test]
fn ci_local_defaults_to_workspace_mode() {
    let (temp, cargo_log, rustup_log) = fake_tool_dir();
    let output = Command::new(bitnet_task_bin())
        .arg("ci-local")
        .current_dir(temp.path())
        .env("PATH", path_with_fake_tools(&temp.path().join("bin")))
        .output()
        .expect("run ci-local");
    assert_success(&output);

    let cargo_invocations = read_invocations(&cargo_log);
    assert!(!cargo_invocations.is_empty(), "expected cargo invocations");
    assert_eq!(cargo_invocations[0], vec!["clean"]);
    assert!(cargo_invocations.iter().any(|args| {
        args.starts_with(&["build".to_string(), "--locked".to_string(), "--workspace".to_string()])
    }));
    assert!(cargo_invocations.iter().any(|args| {
        args.starts_with(&["test".to_string(), "--locked".to_string(), "--workspace".to_string()])
    }));

    let rustup_invocations = read_invocations(&rustup_log);
    assert_eq!(
        rustup_invocations,
        vec![vec![
            "toolchain".to_string(),
            "install".to_string(),
            "1.89.0".to_string(),
            "-q".to_string(),
        ]]
    );
}

#[test]
fn ci_local_receipts_mode_dispatches_explicit_sequence() {
    let (temp, cargo_log, _rustup_log) = fake_tool_dir();
    let output = Command::new(bitnet_task_bin())
        .args(["ci-local", "bitnet-server-receipts"])
        .current_dir(temp.path())
        .env("PATH", path_with_fake_tools(&temp.path().join("bin")))
        .output()
        .expect("run ci-local bitnet-server-receipts");
    assert_success(&output);

    let cargo_invocations = read_invocations(&cargo_log);
    assert!(!cargo_invocations.is_empty(), "expected cargo invocations");
    assert!(
        !cargo_invocations.iter().any(|args| args == &vec!["clean".to_string()]),
        "explicit receipts mode should not fall back to workspace defaults"
    );
    assert!(cargo_invocations.iter().any(|args| {
        args == &vec![
            "+stable".to_string(),
            "check".to_string(),
            "-p".to_string(),
            "bitnet-server".to_string(),
            "--locked".to_string(),
            "--no-default-features".to_string(),
            "--features".to_string(),
            "cpu".to_string(),
        ]
    }));
    assert!(cargo_invocations.iter().any(|args| {
        args == &vec![
            "+stable".to_string(),
            "check".to_string(),
            "-p".to_string(),
            "bitnet-server".to_string(),
            "--locked".to_string(),
            "--no-default-features".to_string(),
            "--features".to_string(),
            "cpu,receipts".to_string(),
        ]
    }));
    assert!(cargo_invocations.iter().any(|args| {
        args == &vec![
            "+stable".to_string(),
            "test".to_string(),
            "-p".to_string(),
            "bitnet-server".to_string(),
            "--no-default-features".to_string(),
            "--features".to_string(),
            "cpu,receipts,tuning".to_string(),
            "--".to_string(),
            "emits_eviction_receipt_with_correct_payload".to_string(),
        ]
    }));
    assert!(cargo_invocations.iter().any(|args| {
        args == &vec![
            "+stable".to_string(),
            "test".to_string(),
            "-p".to_string(),
            "bitnet-server".to_string(),
            "--no-default-features".to_string(),
            "--features".to_string(),
            "cpu,receipts".to_string(),
            "--".to_string(),
            "does_not_emit_receipt_when_disabled".to_string(),
        ]
    }));
}

#[test]
fn generate_policy_writes_expected_fingerprint() {
    let temp = tempfile::tempdir().expect("tempdir");
    let model_path = temp.path().join("fixture.gguf");
    let output_path = temp.path().join("policies").join("correction-policy.yml");
    let model_bytes = b"bitnet-task-generate-policy-fixture";
    fs::write(&model_path, model_bytes).expect("write model");

    let output = Command::new(bitnet_task_bin())
        .args([
            "generate-policy",
            model_path.to_str().expect("model utf8"),
            output_path.to_str().expect("output utf8"),
        ])
        .output()
        .expect("run generate-policy");
    let stdout = assert_success(&output);

    let expected = format!("sha256-{:x}", Sha256::digest(model_bytes));
    let policy = fs::read_to_string(&output_path).expect("read policy");
    assert!(policy.contains(&expected));
    assert!(policy.contains("I2S_DEQUANT_OVERRIDE"));
    assert!(stdout.contains(output_path.to_str().expect("output utf8")));
}

#[test]
fn generate_policy_missing_model_reports_useful_stderr() {
    let temp = tempfile::tempdir().expect("tempdir");
    let missing_model = temp.path().join("missing.gguf");
    let output_path = temp.path().join("policies").join("correction-policy.yml");

    let output = Command::new(bitnet_task_bin())
        .args([
            "generate-policy",
            missing_model.to_str().expect("model utf8"),
            output_path.to_str().expect("output utf8"),
        ])
        .output()
        .expect("run generate-policy");
    let stderr = assert_failure(&output);

    assert!(stderr.contains("Model file not found:"));
    assert!(stderr.contains(missing_model.to_str().expect("model utf8")));
    assert!(!output_path.exists(), "policy should not be written on failure");
}

#[test]
fn check_coverage_accepts_tarpaulin_report() {
    let temp = tempfile::tempdir().expect("tempdir");
    let coverage_path = temp.path().join("coverage.json");
    fs::write(
        &coverage_path,
        r#"{
            "files": {
                "src/lib.rs": { "coverage": [1, 0, null, 3] },
                "src/main.rs": { "coverage": [null, 2] }
            }
        }"#,
    )
    .expect("write coverage report");

    let output = Command::new(bitnet_task_bin())
        .args(["check-coverage", coverage_path.to_str().expect("coverage path utf8"), "75"])
        .output()
        .expect("run check-coverage");
    let stdout = assert_success(&output);

    assert!(stdout.contains("Coverage: 75.00%"));
    assert!(stdout.contains("Threshold: 75.00%"));
    assert!(stdout.contains("Coverage check passed"));
}

#[test]
fn check_coverage_rejects_below_threshold() {
    let temp = tempfile::tempdir().expect("tempdir");
    let coverage_path = temp.path().join("coverage.json");
    fs::write(&coverage_path, r#"{ "coverage": 42.5 }"#).expect("write coverage report");

    let output = Command::new(bitnet_task_bin())
        .args(["check-coverage", coverage_path.to_str().expect("coverage path utf8"), "90"])
        .output()
        .expect("run check-coverage");
    assert_failure(&output);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("Coverage: 42.50%"));
    assert!(stdout.contains("❌ Coverage below threshold (42.50% < 90.00%)"));
}

#[test]
fn check_coverage_wrapper_dispatches_to_rust_facade() {
    let repo_root = repo_root();
    let manifest_path = repo_root.join("Cargo.toml");
    let invocations = run_wrapper("check_coverage.sh", &["coverage.json", "85"]);
    assert_eq!(invocations.len(), 1);
    assert_eq!(
        invocations[0],
        vec![
            "run",
            "--quiet",
            "--locked",
            "--manifest-path",
            manifest_path.to_string_lossy().as_ref(),
            "-p",
            "bitnet-task",
            "--",
            "check-coverage",
            "coverage.json",
            "85",
        ]
    );
}

#[test]
fn perf_wrapper_rewrites_legacy_positionals_into_flags() {
    let repo_root = repo_root();
    let manifest_path = repo_root.join("Cargo.toml");
    let invocations = run_wrapper(
        "perf_phase1_quant_probe.sh",
        &["fixtures/model.gguf", "fixtures/tokenizer.json", "--sentinel"],
    );
    assert_eq!(invocations.len(), 1);
    assert_eq!(
        invocations[0],
        vec![
            "run",
            "--quiet",
            "--locked",
            "--manifest-path",
            manifest_path.to_string_lossy().as_ref(),
            "-p",
            "bitnet-task",
            "--",
            "perf-phase1-quant-probe",
            "--model",
            "fixtures/model.gguf",
            "--tokenizer",
            "fixtures/tokenizer.json",
            "--sentinel",
        ]
    );
}

#[test]
fn vendor_wrapper_injects_master_default() {
    let repo_root = repo_root();
    let manifest_path = repo_root.join("Cargo.toml");
    let invocations = run_wrapper("vendor_ggml_quants.sh", &[]);
    assert_eq!(invocations.len(), 1);
    assert_eq!(
        invocations[0],
        vec![
            "run",
            "--quiet",
            "--locked",
            "--manifest-path",
            manifest_path.to_string_lossy().as_ref(),
            "-p",
            "bitnet-task",
            "--",
            "vendor-ggml-quants",
            "master",
        ]
    );
}
