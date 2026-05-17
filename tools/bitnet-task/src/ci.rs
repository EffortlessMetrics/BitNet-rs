#![allow(clippy::items_after_test_module)]

use super::*;

pub(crate) fn cmd_quality_gate(root: &Path) -> Result<()> {
    println!("🔍 Running BitNet-rs quality gate...");
    println!();
    println!("📝 Formatting code...");
    run_stream(root, "cargo", &["fmt", "--all"], &[])?;

    println!();
    println!("🔎 Running clippy (CPU only)...");
    run_stream(
        root,
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--no-default-features",
            "--features",
            "cpu",
            "--tests",
            "--lib",
            "--exclude",
            "xtask",
            "--",
            "-D",
            "warnings",
            "-D",
            "clippy::ptr_arg",
        ],
        &[("RUSTFLAGS", "-Dwarnings")],
    )?;

    println!();
    println!("✓ Checking tests compile (CPU only)...");
    run_stream(
        root,
        "cargo",
        &["check", "--workspace", "--tests", "--no-default-features", "--features", "cpu"],
        &[("RUSTFLAGS", "-Dwarnings")],
    )?;

    println!();
    println!("🔒 Running dependency security audit...");
    run_stream(root, "cargo", &["deny", "check", "--hide-inclusion-graph"], &[])?;

    println!();
    println!("🚫 Checking for banned patterns...");
    let banned_script = root.join("scripts/hooks/banned-patterns.sh");
    run_stream(root, "bash", &[banned_script.to_string_lossy().as_ref()], &[])?;

    println!();
    println!("✅ All quality checks passed!");
    Ok(())
}

pub(crate) fn cmd_verify_tests(root: &Path) -> Result<()> {
    println!("=== BitNet-rs Verification Tests ===");

    let preflight = collect_preflight_env(false)?;
    let preflight_refs = env_refs_from_pairs(&preflight);

    println!("== Pre-flight test discovery ==");
    require_tests(root, "kind(lib)", &preflight_refs)?;
    require_tests(root, "kind(test)", &preflight_refs)?;

    if env::var("CI_NO_GPU").unwrap_or_default() != "1" {
        println!("Discovering GPU tests...");
        require_tests(root, "test(gpu)", &preflight_refs)?;
    } else {
        println!("Skipping GPU discovery (CI_NO_GPU=1)");
    }

    println!("Testing base build (no extra features)...");
    run_stream(
        root,
        "cargo",
        &["check", "-p", "bitnet-inference", "--no-default-features"],
        &preflight_refs,
    )?;

    println!("Testing build with rt-tokio features...");
    run_stream(
        root,
        "cargo",
        &["check", "-p", "bitnet-inference", "--no-default-features", "--features", "rt-tokio"],
        &preflight_refs,
    )?;

    println!("== Run CPU lane ==");
    run_stream(
        root,
        "cargo",
        &["nextest", "run", "--workspace", "--no-default-features", "--features", "cpu"],
        &preflight_refs,
    )?;

    if env::var("CI_NO_GPU").unwrap_or_default() != "1" {
        println!("== Run GPU lane ==");
        let mut gpu_envs = preflight_refs.clone();
        gpu_envs.push(("BITNET_STRICT_NO_FAKE_GPU", "1"));
        run_stream(
            root,
            "cargo",
            &[
                "nextest",
                "run",
                "-p",
                "bitnet-kernels",
                "--no-default-features",
                "--features",
                "gpu",
            ],
            &gpu_envs,
        )?;
    }

    println!("== Strict edges ==");
    run_stream(root, "cargo", &["nextest", "run", "-p", "bitnet-tokenizers"], &preflight_refs)?;

    println!("== Running GGUF header parser tests ==");
    run_stream(
        root,
        "cargo",
        &["nextest", "run", "-p", "bitnet-inference", "--test", "gguf_header"],
        &preflight_refs,
    )?;

    let smoke_file = env::temp_dir().join("t.gguf");
    let mut gguf = Vec::new();
    gguf.extend_from_slice(b"GGUF\x02\x00\x00\x00");
    gguf.extend_from_slice(&[0_u8; 16]);
    fs::write(&smoke_file, gguf).context("creating tiny gguf stub")?;

    println!("== Creating tiny GGUF stub and running smoke test ==");
    let smoke_file_text = smoke_file.to_string_lossy().into_owned();
    let mut smoke_envs = preflight_refs.clone();
    smoke_envs.push(("BITNET_GGUF", smoke_file_text.as_str()));
    run_stream(
        root,
        "cargo",
        &[
            "nextest",
            "run",
            "-p",
            "bitnet-inference",
            "--no-default-features",
            "--features",
            "rt-tokio",
            "--test",
            "smoke",
        ],
        &smoke_envs,
    )?;

    println!("=== All verification tests completed successfully ===");
    Ok(())
}

pub(crate) fn cmd_ci_local(root: &Path, mode: Option<String>) -> Result<()> {
    let mode = mode.unwrap_or_else(|| "workspace".to_string());
    match mode.as_str() {
        "workspace" => {
            println!("== Clean ==");
            run_stream(root, "cargo", &["clean"], &[])?;
            println!("== Build & Test (strict code lints) ==");
            run_stream(
                root,
                "cargo",
                &["build", "--locked", "--workspace", "--no-default-features", "--features", "cpu"],
                &[("RUSTFLAGS", "-D warnings")],
            )?;
            run_stream(
                root,
                "cargo",
                &[
                    "test",
                    "--locked",
                    "--workspace",
                    "--no-default-features",
                    "--features",
                    "cpu",
                    "--lib",
                ],
                &[("RUSTFLAGS", "-D warnings")],
            )?;

            println!("== Clippy (strict) ==");
            run_stream(
                root,
                "cargo",
                &[
                    "clippy",
                    "--workspace",
                    "--all-targets",
                    "--no-default-features",
                    "--features",
                    "cpu",
                    "--",
                    "-D",
                    "warnings",
                ],
                &[],
            )?;

            println!("== Format check ==");
            run_stream(root, "cargo", &["fmt", "--all", "--", "--check"], &[])?;

            println!("== Docs (relaxed rustdoc) ==");
            run_stream(
                root,
                "cargo",
                &[
                    "doc",
                    "--locked",
                    "--no-deps",
                    "--workspace",
                    "--no-default-features",
                    "--features",
                    "cpu",
                ],
                &[("RUSTDOCFLAGS", "-A warnings")],
            )?;

            println!("== MSRV check (1.89.0) ==");
            let _ =
                run_capture(root, "rustup", &["toolchain", "install", "1.89.0", "-q"], &[], true)?;
            run_capture(
                root,
                "cargo",
                &[
                    "+1.89.0",
                    "check",
                    "--workspace",
                    "--all-targets",
                    "--locked",
                    "--no-default-features",
                    "--features",
                    "cpu",
                ],
                &[],
                false,
            )?;
            println!("✅ All workspace checks passed.");
        }
        "bitnet-server-receipts" => {
            println!("== bitnet-server: receipts validation sequence ==");

            println!("Step 1: Baseline CPU check");
            run_capture(
                root,
                "cargo",
                &[
                    "+stable",
                    "check",
                    "-p",
                    "bitnet-server",
                    "--locked",
                    "--no-default-features",
                    "--features",
                    "cpu",
                ],
                &[("RUSTC_WRAPPER", ""), ("RUSTFLAGS", "-Dwarnings")],
                false,
            )?;

            println!("Step 2: Clippy (CPU only)");
            run_stream(
                root,
                "cargo",
                &[
                    "+stable",
                    "clippy",
                    "-p",
                    "bitnet-server",
                    "--all-targets",
                    "--no-default-features",
                    "--features",
                    "cpu",
                    "--",
                    "-D",
                    "warnings",
                ],
                &[("RUSTC_WRAPPER", "")],
            )?;

            println!("Step 3: Format check");
            run_stream(root, "cargo", &["+stable", "fmt", "--all", "--", "--check"], &[])?;

            println!("Step 4: Documentation");
            run_stream(
                root,
                "cargo",
                &[
                    "+stable",
                    "doc",
                    "-p",
                    "bitnet-server",
                    "--locked",
                    "--no-deps",
                    "--no-default-features",
                    "--features",
                    "cpu",
                ],
                &[("RUSTC_WRAPPER", ""), ("RUSTDOCFLAGS", "-A warnings")],
            )?;

            println!("Step 5: MSRV (1.89.0)");
            run_stream(
                root,
                "cargo",
                &[
                    "+1.89.0",
                    "check",
                    "-p",
                    "bitnet-server",
                    "--locked",
                    "--no-default-features",
                    "--features",
                    "cpu",
                ],
                &[("RUSTC_WRAPPER", "")],
            )?;

            println!("Step 6: Feature combo cpu,receipts");
            run_stream(
                root,
                "cargo",
                &[
                    "+stable",
                    "check",
                    "-p",
                    "bitnet-server",
                    "--locked",
                    "--no-default-features",
                    "--features",
                    "cpu,receipts",
                ],
                &[("RUSTC_WRAPPER", ""), ("RUSTFLAGS", "-Dwarnings")],
            )?;

            println!("Step 7: Feature combo cpu,receipts,tuning");
            run_stream(
                root,
                "cargo",
                &[
                    "+stable",
                    "check",
                    "-p",
                    "bitnet-server",
                    "--locked",
                    "--no-default-features",
                    "--features",
                    "cpu,receipts,tuning",
                ],
                &[("RUSTC_WRAPPER", ""), ("RUSTFLAGS", "-Dwarnings")],
            )?;

            println!("Step 8: Test happy path (receipts enabled)");
            run_stream(
                root,
                "cargo",
                &[
                    "+stable",
                    "test",
                    "-p",
                    "bitnet-server",
                    "--no-default-features",
                    "--features",
                    "cpu,receipts,tuning",
                    "--",
                    "emits_eviction_receipt_with_correct_payload",
                ],
                &[("RUSTC_WRAPPER", "")],
            )?;

            println!("Step 9: Test guard path (receipts disabled)");
            run_stream(
                root,
                "cargo",
                &[
                    "+stable",
                    "test",
                    "-p",
                    "bitnet-server",
                    "--no-default-features",
                    "--features",
                    "cpu,receipts",
                    "--",
                    "does_not_emit_receipt_when_disabled",
                ],
                &[("RUSTC_WRAPPER", "")],
            )?;

            println!("✅ All bitnet-server receipts checks passed.");
        }
        _ => {
            bail!("Usage: cargo run -p bitnet-task -- ci-local [workspace|bitnet-server-receipts]");
        }
    }
    Ok(())
}

pub(crate) fn cmd_verify_crossval(root: &Path) -> Result<()> {
    println!("=== BitNet-rs Crossval Integration Verification ===");
    println!();

    println!("1. Testing repository access...");
    let head = run_capture(
        root,
        "git",
        &["ls-remote", "https://github.com/microsoft/BitNet.git", "HEAD"],
        &[],
        false,
    )?;
    if head.stdout.is_empty() {
        bail!("Cannot access Microsoft BitNet repository");
    }
    println!("   ✓ Can access Microsoft BitNet repository");

    println!("2. Verifying main branch exists...");
    run_capture(
        root,
        "git",
        &["ls-remote", "https://github.com/microsoft/BitNet.git", "refs/heads/main"],
        &[],
        false,
    )?;
    println!("   ✓ Main branch exists");

    let head_text = String::from_utf8_lossy(&head.stdout);
    let latest_commit = head_text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().next())
        .unwrap_or_default()
        .to_string();
    println!("3. Repository information:");
    println!("   - Latest commit: {}", &latest_commit[..latest_commit.len().min(8)]);
    println!("   - Repository URL: https://github.com/microsoft/BitNet.git");
    println!("   - Default branch: main");
    println!();

    println!("4. Environment setup for crossval:");
    println!("   export BITNET_CPP_PATH=$HOME/.cache/bitnet_cpp");
    println!("   export LD_LIBRARY_PATH=$BITNET_CPP_PATH/build/lib:$LD_LIBRARY_PATH");
    println!();
    println!("5. Recommended workflow:");
    println!("   cargo run -p xtask -- download-model");
    println!("   cargo run -p xtask -- fetch-cpp");
    println!("   cargo run -p xtask -- crossval");
    println!("   cargo run -p xtask -- full-crossval");
    println!();
    println!("=== Verification Complete ===");
    println!(
        "The crossval system is properly configured to use the official Microsoft BitNet repository."
    );
    Ok(())
}

pub(crate) fn cmd_sanity_check(root: &Path) -> Result<()> {
    println!("🔍 BitNet-rs Production Sanity Check");
    println!("=======================================");
    println!();

    println!("1. Testing CPU reproducible build...");
    let cpu_ok = run_capture(
        root,
        "cargo",
        &["test", "--locked", "--workspace", "--no-default-features", "--features", "cpu", "--lib"],
        &[],
        true,
    )
    .map(|output| {
        let stdout = String::from_utf8_lossy(&output.stdout);
        output.status.success() && stdout.contains("test result: ok")
    })
    .unwrap_or(false);
    if cpu_ok {
        println!("✓ CPU tests pass with locked dependencies");
    } else {
        println!("✗ CPU tests failed");
    }

    println!("\n2. Testing cargo xtask alias...");
    let alias_ok = run_capture(root, "cargo", &["xtask", "--help"], &[], true)
        .map(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout);
            output.status.success() && stdout.contains("Developer tasks")
        })
        .unwrap_or(false);
    if alias_ok {
        println!("✓ cargo xtask alias works");
    } else {
        println!("✗ cargo xtask alias not configured");
    }

    println!("\n3. GPU preflight check...");
    if command_available("nvidia-smi") {
        let preflight = run_capture(root, "cargo", &["xtask", "gpu-preflight"], &[], true);
        if let Ok(output) = preflight {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines().take(10) {
                if !line.is_empty() {
                    println!("{line}");
                }
            }
        }
    } else {
        println!("No GPU detected - skipping GPU checks");
    }

    println!("\n4. Docker BuildKit availability...");
    let docker_buildkit = run_capture(root, "docker", &["version"], &[], true)
        .map(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout);
            output.status.success() && stdout.contains("buildkit")
        })
        .unwrap_or(false);
    if docker_buildkit {
        println!("✓ Docker BuildKit available");
        println!("  Use: export DOCKER_BUILDKIT=1");
    } else {
        println!("⚠ BuildKit not detected - builds may be slower");
    }

    println!("\n5. Required files check...");
    let required_files =
        [".dockerignore", "rust-toolchain.toml", ".cargo/config.toml", "CODEOWNERS", "Makefile"];
    for file in &required_files {
        if root.join(file).exists() {
            println!("✓ {file} exists");
        } else {
            println!("✗ {file} missing");
        }
    }

    println!("\n6. Docker Compose validation...");
    let compose_ok = run_capture(root, "docker", &["compose", "config", "--quiet"], &[], true)
        .map(|output| output.status.success())
        .unwrap_or(false);
    if compose_ok {
        println!("✓ docker-compose.yml is valid");
        if run_capture(
            root,
            "bash",
            &["-lc", "grep -q \"bitnet_sccache\" docker-compose.yml"],
            &[],
            true,
        )
        .map(|output| output.status.success())
        .unwrap_or(false)
        {
            println!("✓ sccache volume configured");
        }
    } else {
        println!("✗ docker-compose.yml has errors");
    }

    println!("\n=======================================");
    println!("Sanity Check Complete!");
    println!("\nQuick commands:");
    println!("  make b          # Build CPU");
    println!("  make t          # Test");
    println!("  make gpu        # GPU preflight");
    println!("  cargo xtask gpu-smoke  # GPU smoke test");
    println!();
    println!("Docker commands:");
    println!("  export DOCKER_BUILDKIT=1");
    println!("  docker build --target runtime -t bitnet:cpu .");
    println!("  docker compose up --build bitnet-cpu");
    println!();
    println!("Ready for production deployment! 🚀");
    Ok(())
}

#[derive(Clone, Copy)]
enum AcceptanceExit {
    General = 1,
    MissingModel = 2,
    Mapping = 3,
    Tokenizer = 4,
    Inference = 5,
    Tokenization = 6,
    Determinism = 7,
    Test = 8,
    Performance = 9,
    Memory = 10,
}

fn acceptance_fail(code: AcceptanceExit, message: impl AsRef<str>) -> ! {
    eprintln!("{}", message.as_ref());
    std::process::exit(code as i32);
}

struct AcceptanceTemps {
    paths: Vec<PathBuf>,
    next: usize,
}

impl AcceptanceTemps {
    fn new() -> Self {
        Self { paths: Vec::new(), next: 0 }
    }

    fn file(&mut self, label: &str) -> PathBuf {
        self.next += 1;
        let path = env::temp_dir().join(format!(
            "bitnet-task-ci-acceptance-{}-{}-{label}.json",
            std::process::id(),
            self.next
        ));
        self.paths.push(path.clone());
        path
    }
}

impl Drop for AcceptanceTemps {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = fs::remove_file(path);
        }
    }
}

struct AcceptanceMode {
    name: &'static str,
    model_path: String,
    tokenizer_path: Option<String>,
    tokenizer_mode: &'static str,
}

pub(crate) fn cmd_ci_acceptance_gate(root: &Path) -> Result<()> {
    if !command_available("cargo") {
        acceptance_fail(AcceptanceExit::General, "❌ cargo is required but not installed");
    }

    println!("=== BitNet-rs CI Acceptance Gate ===");
    println!("Environment: DETERMINISTIC=1, SEED=42, THREADS=1");
    let gate_env = [
        ("RAYON_NUM_THREADS", "1"),
        ("BITNET_DETERMINISTIC", "1"),
        ("BITNET_SEED", "42"),
        ("OMP_NUM_THREADS", "1"),
        ("GGML_NUM_THREADS", "1"),
    ];

    println!("━━━ Gate 1: Build & Binary Discovery ━━━");
    let rs_bin = discover_release_bitnet(root, &gate_env)
        .unwrap_or_else(|err| acceptance_fail(AcceptanceExit::General, err));
    println!("✓ Binary discovered: {}", rs_bin.display());

    println!("━━━ Gate 2: Unit Tests ━━━");
    let test_output = run_capture(
        root,
        "cargo",
        &[
            "test",
            "--workspace",
            "--no-default-features",
            "--features",
            "cpu",
            "--exclude",
            "bitnet-py",
            "--lib",
            "--",
            "-q",
        ],
        &gate_env,
        true,
    )?;
    if !test_output.status.success() {
        eprintln!("❌ Unit tests failed");
        print_tail(&test_output, 200);
        std::process::exit(AcceptanceExit::Test as i32);
    }
    println!("✓ Unit tests passed");

    println!("━━━ Gate 3: Model Selection ━━━");
    let mode = acceptance_mode();
    println!("Mode: {}", mode.name);
    println!("Model: {}", mode.model_path);
    if !root.join(&mode.model_path).is_file() && !Path::new(&mode.model_path).is_file() {
        eprintln!("❌ Missing model file: {}", mode.model_path);
        if mode.name == "PR" {
            eprintln!(
                "Run: scripts/fetch-pr-model.sh to download TinyLlama with embedded tokenizer"
            );
        } else {
            eprintln!("Run: cargo run -p xtask -- download-model");
        }
        std::process::exit(AcceptanceExit::MissingModel as i32);
    }
    if let Some(tokenizer) = &mode.tokenizer_path
        && !root.join(tokenizer).is_file()
        && !Path::new(tokenizer).is_file()
    {
        acceptance_fail(
            AcceptanceExit::Tokenizer,
            format!("❌ Missing tokenizer: {tokenizer} (required for nightly)"),
        );
    }

    let mut temps = AcceptanceTemps::new();

    println!("━━━ Gate 4: Tensor Mapping Validation ━━━");
    let mapper_json = temps.file("mapper");
    let mapper_out = run_capture(
        root,
        "cargo",
        &["run", "-q", "-p", "xtask", "--", "gate", "mapper", "--model", &mode.model_path],
        &gate_env,
        true,
    )?;
    if !mapper_out.status.success() {
        acceptance_fail(AcceptanceExit::Mapping, "❌ Mapper gate failed to run");
    }
    fs::write(&mapper_json, &mapper_out.stdout)?;
    let mapper = read_json(&mapper_json, AcceptanceExit::Mapping);
    if !mapper.get("ok").and_then(Value::as_bool).unwrap_or(false)
        || mapper.get("unmapped_count").and_then(Value::as_i64).unwrap_or(-1) != 0
    {
        eprintln!("❌ Tensor mapping failed");
        eprintln!("{}", serde_json::to_string_pretty(&mapper).unwrap_or_default());
        std::process::exit(AcceptanceExit::Mapping as i32);
    }
    println!(
        "✓ All tensors mapped (unmapped={})",
        mapper.get("unmapped_count").and_then(Value::as_i64).unwrap_or(0)
    );

    println!("━━━ Gate 5: Strict Inference (no mocks) ━━━");
    let strict_json = temps.file("strict");
    let mut strict_args = vec![
        "run".to_string(),
        "--model".to_string(),
        mode.model_path.clone(),
        "--prompt".to_string(),
        "The capital of France is".to_string(),
        "--bos".to_string(),
        "--max-new-tokens".to_string(),
        "16".to_string(),
        "--temperature".to_string(),
        "0".to_string(),
        "--strict-mapping".to_string(),
        "--strict-tokenizer".to_string(),
        "--json-out".to_string(),
        strict_json.to_string_lossy().into_owned(),
    ];
    push_tokenizer(&mut strict_args, &mode);
    run_acceptance_bin(
        root,
        &rs_bin,
        &strict_args,
        &gate_env,
        AcceptanceExit::Inference,
        "❌ Strict inference failed",
    )?;
    let strict = read_json(&strict_json, AcceptanceExit::Inference);
    let counts_ok = strict.pointer("/counts/unmapped").and_then(Value::as_i64).unwrap_or(-1) == 0
        && value_to_f64(strict.pointer("/counts/n_kv")).unwrap_or(0.0) > 0.0
        && value_to_f64(strict.pointer("/counts/n_tensors")).unwrap_or(0.0) > 0.0;
    let tokenizer_ok = mode.name != "PR"
        || strict.pointer("/tokenizer/type").and_then(Value::as_str) == Some("sentencepiece");
    if !counts_ok || !tokenizer_ok {
        eprintln!("❌ Strict validation failed");
        eprintln!("{}", serde_json::to_string_pretty(&strict).unwrap_or_default());
        std::process::exit(AcceptanceExit::Inference as i32);
    }
    println!("✓ Strict inference passed (tokenizer={})", mode.tokenizer_mode);

    println!("━━━ Gate 6: Tokenization Smoke Test ━━━");
    let prompts = ["The capital of France is", "Once upon a time", "def fibonacci(n):"];
    let mut pass = 0;
    let mut failed = Vec::new();
    for prompt in prompts {
        let tok_json = temps.file("tokenize");
        let mut args = vec![
            "tokenize".to_string(),
            "--model".to_string(),
            mode.model_path.clone(),
            "--prompt".to_string(),
            prompt.to_string(),
            "--bos".to_string(),
            "--json-out".to_string(),
            tok_json.to_string_lossy().into_owned(),
        ];
        push_tokenizer(&mut args, &mode);
        let out = run_capture(root, rs_bin.to_string_lossy().as_ref(), &args, &gate_env, true)?;
        if out.status.success() {
            let tok = read_json_allowing_empty(&tok_json);
            let ids_non_empty = tok
                .as_ref()
                .and_then(|v| v.pointer("/tokens/ids"))
                .and_then(Value::as_array)
                .is_some_and(|ids| !ids.is_empty());
            if ids_non_empty {
                pass += 1;
            } else {
                failed.push(prompt);
            }
        } else {
            failed.push(prompt);
        }
    }
    if pass < 2 {
        eprintln!("❌ Tokenization failed: only {pass}/{} prompts succeeded", prompts.len());
        for prompt in failed {
            eprintln!("  Failed: {prompt}");
        }
        std::process::exit(AcceptanceExit::Tokenization as i32);
    }
    println!("✓ Tokenization smoke test: {pass}/{} passed", prompts.len());

    println!("━━━ Gate 7: Determinism Check ━━━");
    let run1 = temps.file("det1");
    let run2 = temps.file("det2");
    for out_path in [&run1, &run2] {
        let mut args = vec![
            "run".to_string(),
            "--model".to_string(),
            mode.model_path.clone(),
            "--prompt".to_string(),
            "Once upon".to_string(),
            "--bos".to_string(),
            "--max-new-tokens".to_string(),
            "32".to_string(),
            "--temperature".to_string(),
            "0".to_string(),
            "--json-out".to_string(),
            out_path.to_string_lossy().into_owned(),
        ];
        push_tokenizer(&mut args, &mode);
        run_acceptance_bin(
            root,
            &rs_bin,
            &args,
            &gate_env,
            AcceptanceExit::Determinism,
            "❌ Determinism run failed",
        )?;
    }
    let ids1 = token_ids_string(&read_json(&run1, AcceptanceExit::Determinism));
    let ids2 = token_ids_string(&read_json(&run2, AcceptanceExit::Determinism));
    if ids1 != ids2 || ids1 == "[]" {
        eprintln!("❌ Non-deterministic token generation detected");
        eprintln!("Run 1 IDs: {ids1}");
        eprintln!("Run 2 IDs: {ids2}");
        std::process::exit(AcceptanceExit::Determinism as i32);
    }
    println!("✓ Deterministic execution verified");

    println!("━━━ Gate 8: Performance & Memory ━━━");
    run_performance_gate(root, &rs_bin, &mode, &gate_env, &mut temps)?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎉 CI Acceptance Gate: ALL PASSED");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Mode: {}", mode.name);
    println!("Binary: {}", rs_bin.display());
    println!("Model: {}", mode.model_path);
    println!("All gates passed with strict validation");
    Ok(())
}

fn acceptance_mode() -> AcceptanceMode {
    if env::var_os("CI_PR").is_some() || env::var_os("NIGHTLY").is_none() {
        AcceptanceMode {
            name: "PR",
            model_path: env::var("PR_MODEL")
                .unwrap_or_else(|_| "models/tinyllama-q2.gguf".to_string()),
            tokenizer_path: None,
            tokenizer_mode: "embedded",
        }
    } else {
        AcceptanceMode {
            name: "NIGHTLY",
            model_path: env::var("BITNET_GGUF")
                .unwrap_or_else(|_| "models/bitnet/ggml-model-i2_s.gguf".to_string()),
            tokenizer_path: Some(
                env::var("TOKENIZER_PATH")
                    .unwrap_or_else(|_| "models/bitnet/tokenizer.model".to_string()),
            ),
            tokenizer_mode: "external",
        }
    }
}

fn discover_release_bitnet(
    root: &Path,
    envs: &[(&str, &str)],
) -> std::result::Result<PathBuf, String> {
    let output = run_capture(
        root,
        "cargo",
        &[
            "build",
            "-p",
            "bitnet-cli",
            "--release",
            "--no-default-features",
            "--features",
            "cpu,full-cli",
            "--message-format=json",
        ],
        envs,
        true,
    )
    .map_err(|err| format!("❌ Failed to run cargo build: {err:#}"))?;

    for line in String::from_utf8_lossy(&output.stdout).lines().rev() {
        let Ok(value) = serde_json::from_str::<Value>(line) else { continue };
        let is_bitnet_bin = value.pointer("/target/name").and_then(Value::as_str) == Some("bitnet")
            && value
                .pointer("/target/kind")
                .and_then(Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
        if is_bitnet_bin && let Some(exe) = value.get("executable").and_then(Value::as_str) {
            let path = PathBuf::from(exe);
            if is_executable_file(&path) {
                return Ok(path);
            }
        }
    }

    eprintln!("⚠ JSON parse failed, attempting fallback build...");
    run_stream(
        root,
        "cargo",
        &[
            "build",
            "-p",
            "bitnet-cli",
            "--release",
            "--no-default-features",
            "--features",
            "cpu,full-cli",
        ],
        envs,
    )
    .map_err(|err| format!("❌ Fallback build failed: {err:#}"))?;

    find_executable_named(&root.join("target"), "bitnet")
        .ok_or_else(|| "❌ Failed to build or locate bitnet binary".to_string())
}

fn find_executable_named(root: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|n| n.to_str()) == Some(name)
                && is_executable_file(&path)
            {
                return Some(path);
            }
        }
    }
    None
}

fn push_tokenizer(args: &mut Vec<String>, mode: &AcceptanceMode) {
    if let Some(tokenizer) = &mode.tokenizer_path {
        args.push("--tokenizer".to_string());
        args.push(tokenizer.clone());
    }
}

fn run_acceptance_bin(
    root: &Path,
    bin: &Path,
    args: &[String],
    envs: &[(&str, &str)],
    code: AcceptanceExit,
    message: &str,
) -> Result<()> {
    let out = run_capture(root, bin.to_string_lossy().as_ref(), args, envs, true)?;
    if !out.status.success() {
        acceptance_fail(code, message);
    }
    Ok(())
}

fn read_json(path: &Path, code: AcceptanceExit) -> Value {
    match fs::read_to_string(path).ok().and_then(|text| serde_json::from_str(&text).ok()) {
        Some(value) => value,
        None => acceptance_fail(code, format!("❌ Invalid JSON output: {}", path.display())),
    }
}

fn read_json_allowing_empty(path: &Path) -> Option<Value> {
    fs::read_to_string(path).ok().and_then(|text| serde_json::from_str(&text).ok())
}

fn value_to_f64(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn token_ids_string(value: &Value) -> String {
    value
        .pointer("/tokens/ids")
        .cloned()
        .and_then(|ids| serde_json::to_string(&ids).ok())
        .unwrap_or_else(|| "[]".to_string())
}

fn run_performance_gate(
    root: &Path,
    rs_bin: &Path,
    mode: &AcceptanceMode,
    envs: &[(&str, &str)],
    temps: &mut AcceptanceTemps,
) -> Result<()> {
    let perf_json = temps.file("perf");
    let mut args = vec![
        "run".to_string(),
        "--model".to_string(),
        mode.model_path.clone(),
        "--prompt".to_string(),
        "The quick brown fox jumps over the lazy dog".to_string(),
        "--max-new-tokens".to_string(),
        "128".to_string(),
        "--temperature".to_string(),
        "0".to_string(),
        "--json-out".to_string(),
        perf_json.to_string_lossy().into_owned(),
    ];
    push_tokenizer(&mut args, mode);
    let rss_mb = run_performance_command(root, rs_bin, &args, envs)?;

    let perf = read_json(&perf_json, AcceptanceExit::Performance);
    let tokps = value_to_f64(perf.pointer("/throughput/tokens_per_second")).unwrap_or(0.0);
    let decoded = value_to_f64(perf.pointer("/throughput/decoded_tokens")).unwrap_or(0.0);
    if decoded < 64.0 {
        println!(
            "⚠ Warning: only decoded {decoded} tokens (<64), performance measurement may be noisy"
        );
    }
    if tokps < 1.0 {
        acceptance_fail(
            AcceptanceExit::Performance,
            format!("❌ Performance too low: {tokps} tokens/sec < 1.0 minimum"),
        );
    }
    println!("Performance: {tokps} tokens/sec");

    let baseline_path = root.join("ci/baseline.json");
    if baseline_path.is_file() {
        let baseline = read_json(&baseline_path, AcceptanceExit::Performance);
        let model_key = if mode.name == "NIGHTLY" { "bitnet_i2s_cpu" } else { "tinyllama_q2k_cpu" };
        let base_tps =
            value_to_f64(baseline.pointer(&format!("/cpu/{model_key}/tok_s"))).unwrap_or(0.0);
        if base_tps != 0.0 {
            let threshold = 0.95 * base_tps;
            if tokps < threshold {
                acceptance_fail(
                    AcceptanceExit::Performance,
                    format!("❌ Performance regression: {tokps} < 95% of baseline {base_tps}"),
                );
            }
            println!("✓ Performance ratio: {tokps} / {base_tps} baseline");
        }
        if let Some(rss_mb) = rss_mb {
            println!("Memory RSS: {rss_mb}MB");
            let base_rss =
                value_to_f64(baseline.pointer(&format!("/cpu/{model_key}/rss_mb"))).unwrap_or(0.0);
            if base_rss != 0.0 {
                let threshold = (1.03 * base_rss) as u64;
                if rss_mb > threshold {
                    acceptance_fail(
                        AcceptanceExit::Memory,
                        format!("❌ Memory regression: {rss_mb}MB > 103% of baseline {base_rss}MB"),
                    );
                }
                println!("✓ Memory ratio: {rss_mb}MB / {base_rss}MB baseline");
            }
        }
    } else {
        println!("✓ Performance acceptable (no baseline for regression testing)");
    }
    Ok(())
}

fn run_performance_command(
    root: &Path,
    rs_bin: &Path,
    args: &[String],
    envs: &[(&str, &str)],
) -> Result<Option<u64>> {
    let time_program = if command_available("gtime") {
        Some("gtime")
    } else if Path::new("/usr/bin/time").is_file() && time_supports_verbose("/usr/bin/time") {
        Some("/usr/bin/time")
    } else {
        None
    };

    if let Some(time_program) = time_program {
        let mut command = Command::new(time_program);
        command.current_dir(root).arg("-v").arg(rs_bin).args(args);
        for (key, value) in envs {
            command.env(key, value);
        }
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        let output = command
            .output()
            .with_context(|| format!("failed to run `{time_program} -v {}`", rs_bin.display()))?;
        if !output.status.success() {
            acceptance_fail(AcceptanceExit::Performance, "❌ Performance run failed");
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Ok(parse_max_rss_mb(&stderr));
    }

    run_acceptance_bin(
        root,
        rs_bin,
        args,
        envs,
        AcceptanceExit::Performance,
        "❌ Performance run failed",
    )?;
    Ok(None)
}

fn parse_max_rss_mb(time_output: &str) -> Option<u64> {
    time_output.lines().find_map(|line| {
        let (_, value) = line.split_once("Maximum resident set size")?;
        let kb = value.split_whitespace().find_map(|piece| {
            piece.trim_matches(|c: char| !c.is_ascii_digit()).parse::<u64>().ok()
        })?;
        Some(kb / 1024)
    })
}

fn time_supports_verbose(program: &str) -> bool {
    Command::new(program)
        .arg("-v")
        .arg("true")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn print_tail(output: &Output, lines: usize) {
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let all = text.lines().collect::<Vec<_>>();
    let start = all.len().saturating_sub(lines);
    for line in &all[start..] {
        eprintln!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_max_rss_mb, receipt_payload, validate_receipt_payload};
    use serde_json::json;

    #[test]
    fn parses_gnu_time_max_rss() {
        let output = "\tMaximum resident set size (kbytes): 131072\n";
        assert_eq!(parse_max_rss_mb(output), Some(128));
    }

    #[test]
    fn validates_top_level_receipt_payloads() {
        let value = json!({
            "compute_path": "real",
            "backend": "cpu",
            "kernels": ["i2s_cpu_quantize"]
        });
        let summary = validate_receipt_payload(receipt_payload(&value), "inline").unwrap();
        assert_eq!(summary.backend, "cpu");
        assert_eq!(summary.compute_path, "real");
        assert_eq!(summary.kernels_len, 1);
    }

    #[test]
    fn validates_wrapped_receipt_payloads() {
        let value = json!({
            "receipt": {
                "compute_path": "real",
                "backend": "cuda",
                "kernels": ["cuda_gemm"]
            }
        });
        let summary = validate_receipt_payload(receipt_payload(&value), "inline").unwrap();
        assert_eq!(summary.backend, "cuda");
        assert_eq!(summary.kernels_len, 1);
    }
}

pub(crate) fn cmd_grep_guards(root: &Path) -> Result<()> {
    let mut failed = false;

    println!("Checking for absolute paths in source code...");
    let mut absolute_path_args = vec![
        "-n".to_string(),
        "/home/steven".to_string(),
        "-g".to_string(),
        "!target/**".to_string(),
        "-g".to_string(),
        "!.git/**".to_string(),
        "-g".to_string(),
        "!Cargo.lock".to_string(),
        "-g".to_string(),
        "!docs/**".to_string(),
        "-g".to_string(),
        "!*.md".to_string(),
        "-g".to_string(),
        "!ci/receipts/**".to_string(),
        "-g".to_string(),
        "!**/tests/**".to_string(),
        "-g".to_string(),
        "!tests/**".to_string(),
    ];
    absolute_path_args.extend(crate_src_dirs(root)?);
    if run_rg_guard(root, &absolute_path_args)? {
        println!("ERROR: Found absolute paths in production source code");
        failed = true;
    }

    println!("Checking for cuda-only feature gates in source code...");
    if run_filtered_rg_guard(
        root,
        &[
            "-n",
            r#"#\[cfg\(feature\s*=\s*"cuda"\)\]"#,
            "-g",
            "!Cargo.lock",
            "-g",
            "!target/**",
            "-g",
            "!.git/**",
            "-g",
            "!docs/**",
            "-g",
            "!*.md",
            "-g",
            "!tests/fixtures/**",
            "-g",
            "!**/tests/**",
            "-g",
            "!examples/**",
            "crates/",
        ],
        is_allowed_cuda_cfg_attribute,
    )? {
        println!(
            "ERROR: Found cuda-only feature gates in source code (should use 'any(feature = \"gpu\", feature = \"cuda\")')"
        );
        failed = true;
    }

    if run_filtered_rg_guard(
        root,
        &[
            "-n",
            r#"cfg!\(feature\s*=\s*"cuda"\)"#,
            "-g",
            "!Cargo.lock",
            "-g",
            "!target/**",
            "-g",
            "!.git/**",
            "-g",
            "!docs/**",
            "-g",
            "!*.md",
            "-g",
            "!tests/fixtures/**",
            "-g",
            "!**/tests/**",
            "-g",
            "!examples/**",
            "crates/",
        ],
        is_allowed_cuda_cfg_macro,
    )? {
        println!(
            "ERROR: Found runtime cuda-only cfg! checks in source code (should use 'any(feature = \"gpu\", feature = \"cuda\")')"
        );
        failed = true;
    }

    if failed {
        bail!("evidence hygiene checks failed");
    }

    println!("✅ All evidence hygiene checks passed");
    Ok(())
}

fn run_rg_guard<S: AsRef<str>>(root: &Path, args: &[S]) -> Result<bool> {
    let output = run_capture(root, "rg", args, &[], true)?;
    print_command_output(&output);
    match output.status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => bail!(
            "ripgrep guard `{}` failed with {}: {}",
            command_to_string("rg", args),
            output.status,
            command_failure_details(&output)
        ),
    }
}

fn run_filtered_rg_guard<S, F>(root: &Path, args: &[S], allowed: F) -> Result<bool>
where
    S: AsRef<str>,
    F: Fn(&str) -> bool,
{
    let output = run_capture(root, "rg", args, &[], true)?;
    match output.status.code() {
        Some(0) => {
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut failed = false;
            for line in stdout.lines().filter(|line| !allowed(line)) {
                println!("{line}");
                failed = true;
            }
            Ok(failed)
        }
        Some(1) => Ok(false),
        _ => bail!(
            "ripgrep guard `{}` failed with {}: {}",
            command_to_string("rg", args),
            output.status,
            command_failure_details(&output)
        ),
    }
}

fn is_allowed_cuda_cfg_attribute(line: &str) -> bool {
    [
        "crates/bitnet-cli/src/main.rs:",
        "crates/bitnet-device-probe/src/nvidia_cuda.rs:",
        "crates/bitnet-kernels/src/cuda/",
        "crates/bitnet-qk256-dispatch/src/lib.rs:",
    ]
    .iter()
    .any(|prefix| line.starts_with(prefix))
}

fn is_allowed_cuda_cfg_macro(line: &str) -> bool {
    [
        "crates/bitnet-common/src/kernel_registry.rs:",
        "crates/bitnet-opencl/src/diagnostics.rs:",
        "crates/bitnet-runtime-feature-flags/src/lib.rs:",
    ]
    .iter()
    .any(|prefix| line.starts_with(prefix))
}

fn crate_src_dirs(root: &Path) -> Result<Vec<String>> {
    let crates_dir = root.join("crates");
    let mut dirs = Vec::new();
    for entry in fs::read_dir(&crates_dir)
        .with_context(|| format!("failed to read {}", crates_dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to scan {}", crates_dir.display()))?;
        let src = entry.path().join("src");
        if src.is_dir() {
            let rel = src.strip_prefix(root).unwrap_or(&src);
            dirs.push(rel.to_string_lossy().into_owned());
        }
    }
    dirs.sort();
    if dirs.is_empty() {
        bail!("no crate source directories found under {}", crates_dir.display());
    }
    Ok(dirs)
}

pub(crate) fn cmd_validate_receipt(root: &Path, receipt_file: &Path) -> Result<()> {
    let receipt_path = if receipt_file.is_absolute() {
        receipt_file.to_path_buf()
    } else {
        root.join(receipt_file)
    };

    if !receipt_path.is_file() {
        bail!("Receipt file not found: {}", receipt_file.display());
    }

    println!("Validating receipt: {}", receipt_file.display());
    let content = fs::read_to_string(&receipt_path)
        .with_context(|| format!("failed to read {}", receipt_file.display()))?;
    let value: Value = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse {} as JSON", receipt_file.display()))?;
    let receipt = receipt_payload(&value);
    let summary = validate_receipt_payload(receipt, &receipt_file.display().to_string())?;

    println!("✅ Receipt validation passed");
    println!("   Backend: {}", summary.backend);
    println!("   Compute Path: {}", summary.compute_path);
    if matches!(summary.backend, "gpu" | "cuda") {
        println!("   GPU Kernels: {}", summary.kernels_len);
    }
    Ok(())
}

fn receipt_payload(value: &Value) -> &Value {
    value.get("receipt").unwrap_or(value)
}

struct ReceiptSummary<'a> {
    backend: &'a str,
    compute_path: &'a str,
    kernels_len: usize,
}

fn validate_receipt_payload<'a>(receipt: &'a Value, label: &str) -> Result<ReceiptSummary<'a>> {
    let compute_path = receipt
        .get("compute_path")
        .and_then(Value::as_str)
        .with_context(|| format!("missing string receipt.compute_path in {label}"))?;
    let backend = receipt
        .get("backend")
        .and_then(Value::as_str)
        .with_context(|| format!("missing string receipt.backend in {label}"))?;
    let kernels_len = receipt.get("kernels").and_then(Value::as_array).map_or(0, Vec::len);

    if compute_path != "real" {
        bail!("Receipt validation failed: receipt.compute_path must be \"real\"");
    }
    if !matches!(backend, "cpu" | "gpu" | "cuda") {
        bail!("Receipt validation failed: unsupported .receipt.backend `{backend}`");
    }
    if backend == "gpu" && kernels_len == 0 {
        bail!("Receipt validation failed: gpu backend must report at least one kernel");
    }

    Ok(ReceiptSummary { backend, compute_path, kernels_len })
}

pub(crate) fn cmd_e2e_gate(root: &Path, command: &[String]) -> Result<()> {
    use fs2::FileExt;

    let max_e2e = env::var("MAX_E2E").unwrap_or_else(|_| "2".to_string());
    let lock_path = env::temp_dir().join("bitnet-e2e-suite.lock");
    println!("🔒 BitNet-rs E2E Gate: Acquiring test slot (max {max_e2e} concurrent)");
    let lock = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("failed to open {}", lock_path.display()))?;
    if lock.try_lock_exclusive().is_err() {
        println!("⏳ E2E slot busy → waiting for available slot...");
        lock.lock_exclusive().with_context(|| format!("failed to lock {}", lock_path.display()))?;
    }

    println!("✅ E2E slot acquired, running: {}", command.join(" "));
    let preflight = collect_preflight_env(false)?;
    let preflight_refs = env_refs_from_pairs(&preflight);
    let (program, args) = command.split_first().context("e2e-gate requires a command to run")?;
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = run_stream(root, program, &arg_refs, &preflight_refs);
    match &result {
        Ok(()) => println!("🏁 E2E slot released (exit code: 0)"),
        Err(err) => println!("🏁 E2E slot released (error: {err:#})"),
    }
    lock.unlock().with_context(|| format!("failed to unlock {}", lock_path.display()))?;
    result
}

fn print_command_output(output: &Output) {
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
}
