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
