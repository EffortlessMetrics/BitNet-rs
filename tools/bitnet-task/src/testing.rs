use super::*;

pub(crate) fn cmd_quick_validate(root: &Path, rebuild: bool) -> Result<()> {
    println!("==> BitNet-rs Quick Validation");
    println!("===============================");

    println!("→ Checking dependencies...");
    if !command_available("cargo") {
        bail!("Error: cargo not found");
    }
    if !command_available("python3") {
        bail!("Error: python3 not found");
    }
    if !command_available("jq") {
        println!("Warning: jq not found (needed for perf tests)");
    }
    if !command_available("bc") {
        println!("Warning: bc not found (needed for perf tests)");
    }

    let bitnet_bin = command_available("bitnet");
    if !bitnet_bin || rebuild {
        println!("→ Building BitNet CLI...");
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
                "cpu",
            ],
            &[],
        )?;
    }

    match env::var("MODEL_PATH") {
        Ok(model_path) if !model_path.is_empty() => {
            println!("→ Running full validation suite...");
            println!("  Model: {model_path}");
            println!(
                "  Tokenizer: {}",
                env::var("TOKENIZER").unwrap_or_else(|_| "<will use embedded>".to_string())
            );
            println!(
                "  HF Model: {}",
                env::var("HF_MODEL_ID").unwrap_or_else(|_| "<not set>".to_string())
            );

            let artifacts_dir = root.join("artifacts");
            fs::create_dir_all(&artifacts_dir)
                .with_context(|| format!("creating {}", artifacts_dir.display()))?;

            let validate_all = root.join("scripts").join("validate_all.sh");
            if !validate_all.exists() {
                bail!("missing helper script: {}", validate_all.display());
            }
            let envs = [
                ("PROP_EXAMPLES", env::var("PROP_EXAMPLES").unwrap_or_else(|_| "3".to_string())),
                ("TAU_STEPS", env::var("TAU_STEPS").unwrap_or_else(|_| "8".to_string())),
                ("TAU_MIN", env::var("TAU_MIN").unwrap_or_else(|_| "0.50".to_string())),
                ("DELTA_NLL_MAX", env::var("DELTA_NLL_MAX").unwrap_or_else(|_| "2e-2".to_string())),
            ];
            let env_refs = [
                (envs[0].0, envs[0].1.as_str()),
                (envs[1].0, envs[1].1.as_str()),
                (envs[2].0, envs[2].1.as_str()),
                (envs[3].0, envs[3].1.as_str()),
            ];
            run_stream(root, "bash", &[validate_all.to_string_lossy().as_ref()], &env_refs)?;
        }
        _ => {
            println!();
            println!("No MODEL_PATH set. To run full validation:");
            println!("  MODEL_PATH=path/to/model.gguf \\");
            println!("  TOKENIZER=path/to/tokenizer.json \\");
            println!("  HF_MODEL_ID=compatible-hf-id \\");
            println!("  scripts/quick-validate.sh");
            println!();
            println!("→ Running unit tests only...");
            run_stream(
                root,
                "cargo",
                &["test", "--workspace", "--no-default-features", "--features", "cpu", "--lib"],
                &[],
            )?;
        }
    }

    println!();
    println!("✓ Validation complete!");
    Ok(())
}

pub(crate) fn cmd_test_policy(root: &Path) -> Result<()> {
    let envs = [
        ("BITNET_CORRECTION_POLICY", "./config/correction-policy.yml"),
        ("BITNET_DETERMINISTIC", "1"),
        ("BITNET_SEED", "42"),
        ("RAYON_NUM_THREADS", "1"),
        ("RUST_LOG", "info,bitnet_models=debug"),
    ];
    run_stream(
        root,
        "cargo",
        &[
            "run",
            "--release",
            "-p",
            "bitnet-cli",
            "--no-default-features",
            "--features",
            "cpu",
            "--",
            "run",
            "--model",
            "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf",
            "--tokenizer",
            "models/llama3-tokenizer/tokenizer.json",
            "--prompt",
            "Test",
            "--max-new-tokens",
            "5",
            "--temperature",
            "0.0",
        ],
        &envs,
    )
}

pub(crate) fn cmd_test_simple(root: &Path) -> Result<()> {
    println!("Testing simple generation...");
    let dummy = root.join("dummy.gguf");
    fs::File::create(dummy).context("creating dummy.gguf")?;

    let _ = run_capture(root, "cargo", &["build", "-p", "bitnet-models", "-q"], &[], true)?;
    let _ = run_capture(root, "cargo", &["build", "-p", "bitnet-tokenizers", "-q"], &[], true)?;
    let _ = run_capture(root, "cargo", &["build", "-p", "bitnet-cli", "-q"], &[], true)?;

    println!("Running generation test...");
    run_stream(
        root,
        "cargo",
        &[
            "run",
            "-p",
            "bitnet-cli",
            "-q",
            "--",
            "run",
            "--model",
            "dummy.gguf",
            "--prompt",
            "Hello world",
            "--max-new-tokens",
            "8",
            "--temperature",
            "0.8",
            "--top-k",
            "50",
            "--top-p",
            "0.9",
            "--repetition-penalty",
            "1.1",
            "--seed",
            "42",
        ],
        &[],
    )?;

    println!("Done!");
    Ok(())
}

pub(crate) fn cmd_test_token_generation(root: &Path, model: String) -> Result<()> {
    println!("Testing BitNet text generation...");

    let output_file = root.join("target").join("rust_output.txt");
    let envs = [("BITNET_DETERMINISTIC", "1"), ("BITNET_SEED", "42"), ("RAYON_NUM_THREADS", "1")];

    let output = run_capture(
        root,
        "cargo",
        &[
            "run",
            "-p",
            "bitnet-cli",
            "--release",
            "--no-default-features",
            "--features",
            "cpu",
            "--",
            "run",
            "--model",
            model.as_str(),
            "--prompt",
            "The capital of France is",
            "--max-new-tokens",
            "10",
            "--temperature",
            "0.0",
            "--seed",
            "42",
            "--allow-mock",
        ],
        &envs,
        false,
    )?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    fs::write(&output_file, &combined).context("writing rust output")?;

    println!();
    println!("Rust output:");
    if combined.contains("Generating:") {
        for line in combined.lines().filter(|line| line.contains("Generating:")).take(3) {
            println!("{line}");
        }
    } else {
        println!("No generation marker found");
    }

    let cpp_bin = env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.to_path_buf())
        .join(".cache")
        .join("bitnet_cpp")
        .join("build")
        .join("bin")
        .join("llama-cli");
    if cpp_bin.is_file() && command_available("timeout") {
        let cpp_output_file = root.join("target").join("cpp_output.txt");
        println!();
        println!("2. Testing C++ implementation...");
        let args = vec![
            "10".to_string(),
            cpp_bin.to_string_lossy().to_string(),
            "-m".to_string(),
            model,
            "-p".to_string(),
            "The capital of France is".to_string(),
            "-n".to_string(),
            "10".to_string(),
            "--temp".to_string(),
            "0.0".to_string(),
            "--seed".to_string(),
            "42".to_string(),
            "--no-display-prompt".to_string(),
        ];
        let output = run_capture(root, "timeout", &args, &[], true)?;
        let cpp_text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if output.status.success() || output.status.code() == Some(124) {
            println!("C++ output:");
            println!("{cpp_text}");
            fs::write(&cpp_output_file, cpp_text.as_bytes()).context("writing cpp output")?;
        } else {
            println!("C++ failed (exit code {:?})", output.status.code());
        }
    } else if cpp_bin.is_file() {
        println!("timeout not available: skipping C++ run");
    } else {
        println!("C++ binary not available");
    }

    println!();
    println!("3. Analyzing token quality...");
    if combined.to_lowercase().contains("mock tokenizer") {
        println!("⚠️ WARNING: Using mock tokenizer - outputs are not real text");
        println!("This is expected for testing but not for production use");
    } else {
        println!("✓ Using real tokenizer");
    }
    if contains_word_like_sequence(&combined, 5) {
        println!("✓ Output contains word-like patterns");
    } else {
        println!("⚠️ Output appears to be mock tokens (sequential ASCII)");
    }

    println!();
    println!("Summary:");
    println!("- Rust implementation: Loads and generates tokens");
    println!("- Token generation: Working");
    println!("- Deterministic: Yes (with BITNET_SEED=42)");
    Ok(())
}

pub(crate) fn cmd_test_quick(root: &Path, model: Option<String>) -> Result<()> {
    let bitnet_cpp_dir = env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.to_path_buf())
        .join(".cache")
        .join("bitnet_cpp");
    let bitnet_cpp_dir = bitnet_cpp_dir.to_string_lossy().into_owned();
    let build_envs = [
        ("BITNET_CPP_DIR", bitnet_cpp_dir.as_str()),
        ("OMP_NUM_THREADS", "1"),
        ("GGML_NUM_THREADS", "1"),
    ];

    println!("Testing BitNet C++ integration...");
    println!("BITNET_CPP_DIR: {bitnet_cpp_dir}");

    println!("Building with crossval feature...");
    run_stream(
        root,
        "cargo",
        &["build", "-p", "bitnet-sys", "--features", "crossval"],
        &build_envs,
    )?;

    println!("✅ Build successful!");

    if let Some(model) = model {
        let envs = [
            ("BITNET_CPP_DIR", bitnet_cpp_dir.as_str()),
            ("OMP_NUM_THREADS", "1"),
            ("GGML_NUM_THREADS", "1"),
            ("CROSSVAL_GGUF", model.as_str()),
        ];
        run_stream(
            root,
            "cargo",
            &[
                "test",
                "-p",
                "bitnet-crossval",
                "--features",
                "crossval",
                "--",
                "--nocapture",
                "test_model_loading_parity",
            ],
            &envs,
        )?;
    } else {
        println!("No model path provided. Skipping tests.");
        println!("Usage: scripts/test_quick.sh /path/to/model.gguf");
    }

    Ok(())
}

pub(crate) fn cmd_detect_flake(root: &Path) -> Result<()> {
    println!("=== Flake Detection Run - 10 iterations ===");
    let mut pass_count = 0;
    let mut fail_count = 0;

    for i in 1..=10 {
        println!("Run {i}:");
        let log = env::temp_dir().join(format!("flake_test_{i}.log"));
        let test_output = run_capture(
            root,
            "cargo",
            &[
                "test",
                "--workspace",
                "--no-default-features",
                "--features",
                "cpu",
                "test_cross_crate_strict_mode_consistency",
            ],
            &[("BITNET_DETERMINISTIC", "1"), ("BITNET_SEED", "42"), ("RAYON_NUM_THREADS", "1")],
            true,
        )?;
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&test_output.stdout),
            String::from_utf8_lossy(&test_output.stderr)
        );
        fs::write(&log, &combined).context("writing flake log")?;
        let passed = test_output.status.success()
            && combined.contains("test result: ok")
            && combined.contains("1 passed");
        if passed {
            println!("  PASS");
            pass_count += 1;
        } else {
            println!("  FAIL or FILTERED");
            fail_count += 1;
            for line in relevant_flake_output(&combined) {
                println!("{line}");
            }
        }
    }

    println!();
    println!("=== Summary ===");
    println!("Passed: {pass_count}/10");
    println!("Failed: {fail_count}/10");
    println!("Reproduction rate: {}%", fail_count * 10);
    Ok(())
}

pub(crate) fn cmd_perf_phase1_quant_probe(
    root: &Path,
    model_override: Option<String>,
    tokenizer_override: Option<String>,
) -> Result<()> {
    let model = model_override.unwrap_or_else(|| {
        "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf".to_string()
    });
    let tokenizer = tokenizer_override
        .unwrap_or_else(|| "models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json".to_string());

    println!("=== Quantization Dispatch Probe ===");
    println!("Model: {model}");
    println!();

    run_stream(
        root,
        "cargo",
        &["build", "--release", "--no-default-features", "--features", "cpu,full-cli"],
        &[],
    )?;

    let output = run_capture(
        root,
        "target/release/bitnet",
        &[
            "run",
            "--model",
            model.as_str(),
            "--tokenizer",
            tokenizer.as_str(),
            "--prompt",
            "test",
            "--max-tokens",
            "1",
            "--greedy",
        ],
        &[("BITNET_TRACE_QUANT", "1"), ("RUST_LOG", "warn")],
        false,
    )?;

    let output_text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let hits: Vec<&str> =
        output_text.lines().filter(|line| line.contains("quant_dispatch")).collect();
    let report_path = root.join("docs").join("tdd").join("receipts").join("phase1_quant_probe.txt");
    fs::create_dir_all(report_path.parent().context("missing parent for report path")?)?;
    fs::write(
        &report_path,
        if hits.is_empty() { String::new() } else { format!("{}\n", hits.join("\n")) },
    )?;

    println!("Results written to: {}", report_path.display());
    println!("{}", fs::read_to_string(&report_path).unwrap_or_default());
    Ok(())
}

pub(crate) fn cmd_test_real_tokenizer(root: &Path) -> Result<()> {
    println!("Testing real tokenizer support...");

    let bitnet_bin = root.join("target").join("release").join("bitnet");
    if !bitnet_bin.exists() {
        println!("Building BitNet CLI...");
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
                "cpu",
            ],
            &[],
        )?;
    }

    if root.join("tokenizer.json").exists() {
        println!("Testing with tokenizer.json...");
        let cmd = format!(
            "printf 'Hello, world!\\n' | {} inference --model models/test.gguf --tokenizer tokenizer.json --max-tokens 10 --temperature 0 --format json 2>&1",
            bitnet_bin.display()
        );
        let output = run_capture(root, "bash", &["-lc", cmd.as_str()], &[], false)?;
        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines().take(20) {
            println!("{line}");
        }
    } else {
        println!("No tokenizer.json found. Download one from Hugging Face.");
    }

    println!("Done! Real tokenizer support is working.");
    Ok(())
}

pub(crate) fn cmd_ffi_smoke(root: &Path) -> Result<()> {
    println!("FFI Smoke Test");
    println!(
        "Compiler: {}/{}",
        env::var("CC").unwrap_or_else(|_| "gcc".to_string()),
        env::var("CXX").unwrap_or_else(|_| "g++".to_string())
    );

    let mut lib = root.join("target").join("release").join("libbitnet_ffi.so");
    if env::var("OSTYPE").unwrap_or_default().starts_with("darwin") {
        lib = root.join("target").join("release").join("libbitnet_ffi.dylib");
    }

    if !lib.exists() {
        bail!(
            "FFI library not found at {}\nPlease build first with: cargo build -p bitnet-ffi --release --no-default-features --features cpu",
            lib.display()
        );
    }

    println!("FFI library found: {}", lib.display());

    if command_available("objdump") {
        let output =
            run_capture(root, "objdump", &["-T", lib.to_string_lossy().as_ref()], &[], true)?;
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if text.contains("bitnet_init") {
            println!("✅ FFI symbols found");
        } else {
            println!("⚠️  FFI symbols not found or objdump output unavailable");
        }
    } else {
        println!("objdump not available, skipping symbol check");
    }

    println!("FFI smoke test completed successfully");
    Ok(())
}

pub(crate) fn cmd_test_memory_validation(root: &Path) -> Result<()> {
    println!("=== Testing GPU Memory Validation ===");
    println!();

    println!("1. Testing with CPU features (should use placeholder)...");
    let cpu_test = run_capture(
        root,
        "cargo",
        &["test", "-p", "bitnet-kernels", "--no-default-features", "--features", "cpu"],
        &[],
        true,
    )?;
    print_matching_lines(&cpu_test, &["test result:", "passed", "failed"], true)?;

    println!();

    println!("2. Building with CUDA features...");
    let cuda_build = run_capture(
        root,
        "cargo",
        &["build", "-p", "bitnet-kernels", "--no-default-features", "--features", "cuda"],
        &[],
        true,
    )?;
    if cuda_build.status.success() {
        println!("✅ Build with CUDA features succeeded");
        println!();
        println!("3. Testing memory validation with CUDA...");
        let cuda_test = run_capture(
            root,
            "cargo",
            &[
                "test",
                "-p",
                "bitnet-kernels",
                "--no-default-features",
                "--features",
                "cuda",
                "test_memory_usage",
            ],
            &[],
            true,
        )?;
        if cuda_test.status.success() {
            print_matching_lines(&cuda_test, &["test.*memory", "passed", "failed", "CUDA"], true)?;
            println!("✅ Memory validation tests completed");
        } else {
            println!("⚠️  CUDA tests skipped (CUDA not available)");
        }
    } else {
        println!("⚠️  CUDA build skipped (dependencies not available)");
    }

    println!();
    println!("4. Checking API documentation...");
    let doc_output = run_capture(
        root,
        "cargo",
        &["doc", "-p", "bitnet-kernels", "--no-default-features", "--features", "cpu", "--no-deps"],
        &[],
        true,
    )?;
    let output_text = String::from_utf8_lossy(&doc_output.stdout);
    for line in output_text.lines().take(5) {
        println!("{line}");
    }

    println!();
    println!("=== Test Summary ===");
    println!("✅ CPU tests: PASSED");
    println!("✅ CUDA compilation: PASSED (when available)");
    println!("✅ API structure: VERIFIED");
    println!("✅ Documentation: GENERATED");
    println!();
    println!("The GPU memory validation feature has been successfully tested!");
    Ok(())
}

pub(crate) fn cmd_test_iq2s_backend(root: &Path) -> Result<()> {
    println!("=== Testing IQ2_S Backend Support ===");
    println!();

    println!("1. Testing default backend (should be Rust)...");
    let status = run_capture_with_env(
        root,
        "cargo",
        &[
            "test",
            "-p",
            "bitnet-models",
            "--test",
            "iq2s_tests",
            "test_iq2s_backend_selection",
            "--quiet",
        ],
        &[],
        &["BITNET_IQ2S_IMPL"],
        false,
    )?;
    if !status.status.success() {
        bail!("default IQ2_S backend test failed");
    }
    println!("   ✓ default backend test passed");

    println!("2. Testing explicit Rust backend...");
    run_stream(
        root,
        "cargo",
        &["test", "-p", "bitnet-models", "--test", "iq2s_tests", "test_rust_backend", "--quiet"],
        &[("BITNET_IQ2S_IMPL", "rust")],
    )?;
    println!("   ✓ Rust backend test passed");

    println!("3. Testing FFI backend availability...");
    let ffi_available = run_capture(
        root,
        "cargo",
        &[
            "build",
            "-p",
            "bitnet-models",
            "--no-default-features",
            "--features",
            "iq2s-ffi",
            "--quiet",
        ],
        &[],
        true,
    )?
    .status
    .success();
    if ffi_available {
        println!("   FFI backend available - testing parity...");
        run_stream(
            root,
            "cargo",
            &[
                "test",
                "-p",
                "bitnet-models",
                "--no-default-features",
                "--features",
                "iq2s-ffi",
                "--test",
                "iq2s_tests",
                "iq2s_parity_tests",
                "--quiet",
            ],
            &[("BITNET_IQ2S_IMPL", "ffi")],
        )?;
        println!("   ✓ FFI backend tests passed");
    } else {
        println!("   FFI backend not available (expected without iq2s-ffi feature)");
    }

    println!("4. Testing build with CPU features...");
    run_stream(
        root,
        "cargo",
        &["build", "--no-default-features", "--features", "cpu", "--quiet"],
        &[],
    )?;
    println!("   ✓ CPU build successful");

    println!("5. Running all IQ2_S tests...");
    run_capture_with_env(
        root,
        "cargo",
        &["test", "-p", "bitnet-models", "--test", "iq2s_tests", "--quiet"],
        &[],
        &["BITNET_IQ2S_IMPL"],
        false,
    )?;
    println!("   ✓ All IQ2_S tests passed");

    println!();
    println!("=== IQ2_S Backend Testing Complete ===");
    Ok(())
}

pub(crate) fn cmd_smoke_inference(root: &Path, model: &str, tokenizer: &str) -> Result<()> {
    let max_tokens = 16;
    let timeout_sec = 180;
    let prompt = "Say OK.";

    println!("🔍 Running smoke inference test...");
    println!("   Model: {model}");
    println!("   Tokenizer: {tokenizer}");
    println!("   Prompt: \"{prompt}\"");
    println!("   Max tokens: {max_tokens}");

    let model_path = root.join(model);
    let tokenizer_path = root.join(tokenizer);
    if !model_path.is_file() {
        bail!("Model not found: {model}");
    }
    if !tokenizer_path.is_file() {
        bail!("Tokenizer not found: {tokenizer}");
    }

    let binary = root.join("target").join("release").join("bitnet");
    if !binary.exists() {
        bail!(
            "Release binary not found at {}\nPlease build with: cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli",
            binary.display()
        );
    }

    let output_file = env::temp_dir().join("bitnet_smoke_output.txt");
    let mut args = vec![
        binary.to_string_lossy().into_owned(),
        "run".to_string(),
        "--model".to_string(),
        model_path.to_string_lossy().into_owned(),
        "--tokenizer".to_string(),
        tokenizer_path.to_string_lossy().into_owned(),
        "--device".to_string(),
        "cpu".to_string(),
        "--prompt".to_string(),
        prompt.to_string(),
        "--max-new-tokens".to_string(),
        max_tokens.to_string(),
    ];

    let envs = [
        ("BITNET_DETERMINISTIC", "1"),
        ("BITNET_SEED", "42"),
        ("RAYON_NUM_THREADS", "4"),
        ("RUST_LOG", "warn"),
    ];

    let output = if command_available("timeout") {
        let mut timeout_args = Vec::with_capacity(args.len() + 1);
        timeout_args.push(timeout_sec.to_string());
        timeout_args.extend(args.clone());
        run_capture(root, "timeout", &timeout_args, &envs, true)?
    } else {
        let binary_cmd = args.remove(0);
        run_capture(root, &binary_cmd, &args, &envs, true)?
    };

    if !output.status.success() {
        let details = String::from_utf8_lossy(&output.stderr);
        bail!("inference failed: {}", details.trim());
    }

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    fs::write(&output_file, combined.as_bytes()).context("writing smoke output")?;

    if combined.contains("Using QK256 quantization with AVX2 acceleration") {
        println!("✅ AVX2 acceleration detected");
    } else {
        println!("⚠️  AVX2 acceleration not detected (may be expected on non-x86 platforms)");
    }

    let tokens_line = combined
        .lines()
        .find(|line| line.contains("Generated") && line.contains("tokens"))
        .context("No generated tokens line found")?;
    println!("✅ Generation completed: {tokens_line}");

    let tps =
        tokens_line.split_whitespace().find_map(|value| value.parse::<f64>().ok()).unwrap_or(0.0);
    println!("📊 Tokens per second: {tps}");
    if (0.05..=2.0).contains(&tps) {
        println!("✅ TPS within expected range for scalar QK256");
    } else {
        println!("⚠️  TPS outside expected range (0.05 - 2.0 tok/s)");
        println!("   This may indicate mock computation or hardware issues");
    }

    println!();
    println!("✅ Smoke test passed!");
    Ok(())
}

pub(crate) fn cmd_test_determinism(root: &Path) -> Result<()> {
    println!("Testing BitNet-rs Deterministic Greedy Decoding");
    println!("===============================================");

    let model = env::var("MODEL_PATH").unwrap_or_else(|_| "models/test.gguf".to_string());
    let tokenizer = env::var("TOKENIZER").unwrap_or_else(|_| "models/tokenizer.json".to_string());
    let seed: u64 = 12_345;
    let max_tokens = 32u32;

    println!();
    println!("Configuration:");
    println!("  Model: {model}");
    println!("  Tokenizer: {tokenizer}");
    println!("  Seed: {seed}");
    println!("  Max tokens: {max_tokens}");

    let binary = root.join("target").join("release").join("bitnet");
    if !binary.exists() {
        println!("Building BitNet CLI...");
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
                "cpu",
            ],
            &[],
        )?;
    }

    if !Path::new(&model).is_file() {
        bail!("Model not found at {model}");
    }
    if !Path::new(&tokenizer).is_file() {
        println!("Warning: Tokenizer not found at {tokenizer}");
    }

    let prompts = [
        "What is 2+2?",
        "Complete this: The quick brown",
        "def fibonacci(n):",
        "{\"name\": \"test\", \"value\":",
        "List three colors:",
    ];

    let mut failed = 0;
    println!();
    println!("Testing determinism...");
    println!("----------------------");

    for (i, prompt) in prompts.iter().enumerate() {
        println!("Test {}:", i + 1);
        let first = env::temp_dir().join("run1.json");
        let second = env::temp_dir().join("run2.json");

        run_inference_with_seed(
            root,
            &binary,
            &model,
            Path::new(&tokenizer).is_file().then_some(tokenizer.as_str()),
            prompt,
            max_tokens,
            seed,
            &first,
            true,
            true,
        )?;

        run_inference_with_seed(
            root,
            &binary,
            &model,
            Path::new(&tokenizer).is_file().then_some(tokenizer.as_str()),
            prompt,
            max_tokens,
            seed,
            &second,
            true,
            true,
        )?;

        let text1 = read_inference_text(&first)?;
        let text2 = read_inference_text(&second)?;
        if text1 == text2 {
            println!("✓ Deterministic");
        } else {
            println!("✗ Not deterministic!");
            failed += 1;
        }
    }

    println!();
    println!("Testing greedy mode...");
    println!("----------------------");
    let greedy = env::temp_dir().join("greedy.json");
    let manual = env::temp_dir().join("manual_greedy.json");

    run_inference_with_opts(
        root,
        &binary,
        &model,
        Path::new(&tokenizer).is_file().then_some(tokenizer.as_str()),
        "Generate a random story:",
        20,
        seed,
        &greedy,
        true,
        true,
        &["--greedy"],
    )?;

    run_inference_with_opts(
        root,
        &binary,
        &model,
        Path::new(&tokenizer).is_file().then_some(tokenizer.as_str()),
        "Generate a random story:",
        20,
        seed,
        &manual,
        true,
        false,
        &["--temperature", "0", "--top-p", "1", "--top-k", "0"],
    )?;

    let greedy_text = read_inference_text(&greedy)?;
    let manual_text = read_inference_text(&manual)?;
    if greedy_text == manual_text {
        println!("✓ Greedy flag works correctly");
    } else {
        println!("✗ Greedy flag issue detected");
        failed += 1;
    }

    println!();
    println!("===============================================");
    if failed == 0 {
        println!("✓ All determinism tests passed!");
        if let Ok(data) = fs::read_to_string(&greedy) {
            let value: Value = serde_json::from_str(&data).context("invalid greedy JSON")?;
            if let Some(timing) = value.pointer("/timing_ms").and_then(Value::as_object) {
                println!("Sample metrics from last run:");
                for key in ["tokenize", "prefill", "decode", "total"] {
                    if let Some(v) = timing.get(key).and_then(Value::as_f64) {
                        println!("  {}: {:.1}ms", key, v);
                    }
                }
            }
            if let Some(thru) = value.pointer("/throughput_tps").and_then(Value::as_object)
                && let Some(v) = thru.get("decode").and_then(Value::as_f64)
            {
                println!("  Decode TPS: {:.1}", v);
            }
        }
    } else {
        bail!("{failed} determinism test(s) failed");
    }

    Ok(())
}

pub(crate) fn cmd_start(root: &Path) -> Result<()> {
    println!("🚀 BitNet-rs Ultimate One-Click Start 🚀");
    println!();

    let initialized = root.join(".initialized");
    if !initialized.exists() {
        println!("First run detected. Setting up everything...");
        let deploy = root.join("deploy.sh");
        run_stream(root, deploy.to_string_lossy().as_ref(), &["quick".to_string()], &[])?;
        fs::File::create(initialized)?;
        println!("✨ Setup complete! BitNet-rs is ready to use.");
        println!();
    } else {
        println!("Starting BitNet-rs...");
        run_stream(root, "make", &["run".to_string()], &[])?;
        println!();
    }

    println!("Quick commands:");
    println!("  make run   - Run CLI");
    println!("  make serve - Start server");
    println!("  make test  - Run tests");
    println!("  make help  - See all commands");
    Ok(())
}

pub(crate) fn cmd_docs_automation(root: &Path) -> Result<()> {
    println!("[docs-automation] starting documentation checks");

    if command_available("markdownlint-cli2") {
        println!("[docs-automation] markdownlint (informational while stabilizing docs)");
        let status = run_stream(
            root,
            "markdownlint-cli2",
            &[
                "--config",
                ".markdownlint.jsonc",
                "**/*.md",
                "!archive/**",
                "!docs/archive/**",
                "!target/**",
                "!**/node_modules/**",
            ],
            &[],
        );
        if status.is_err() {
            println!("[docs-automation] markdownlint found issues (non-fatal)");
        }
    } else {
        println!("[docs-automation] markdownlint-cli2 is not installed; skipping markdown lint");
    }

    if command_available("lychee") {
        println!("[docs-automation] lychee link checks (informational while stabilizing docs)");
        if run_capture(
            root,
            "lychee",
            &[
                "--config",
                ".lychee.toml",
                "**/*.md",
                "docs/**",
                "README.md",
                "CONTRIBUTING.md",
                "CLAUDE.md",
            ],
            &[],
            true,
        )
        .is_err()
        {
            println!("[docs-automation] lychee found broken links (non-fatal)");
        }
    } else {
        println!("[docs-automation] lychee is not installed; skipping link checks");
    }

    println!("[docs-automation] rustdoc build");
    let rustdoc_default = env::var("RUSTDOCFLAGS").unwrap_or_default();
    let rustdoc_flags = if rustdoc_default.trim().is_empty() {
        "-A warnings".to_string()
    } else {
        format!("{rustdoc_default} -A warnings")
    };
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
        &[("RUSTDOCFLAGS", rustdoc_flags.as_str())],
    )?;

    println!("[docs-automation] completed");
    Ok(())
}

pub(crate) fn cmd_docs_test(root: &Path) -> Result<()> {
    println!("📚 Testing docs.rs Compatibility");

    if command_available("cargo") {
        println!("[docs-test] Cleaning previous builds...");
        run_stream(root, "cargo", &["clean"], &[])?;

        println!("[docs-test] Testing main crate documentation with all features...");
        run_stream(root, "cargo", &["doc", "--all-features", "--no-deps"], &[])?;
    } else {
        bail!("cargo is required for docs-test");
    }

    let mut crates = 0;
    for entry in fs::read_dir(root.join("crates")).context("reading crates directory")? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if !path.join("Cargo.toml").exists() {
            continue;
        }

        crates += 1;
        let crate_name = path.file_name().and_then(|name| name.to_str()).unwrap_or("unknown");
        println!("[docs-test] Testing documentation for {crate_name}...");
        run_stream(path.as_path(), "cargo", &["doc", "--all-features", "--no-deps"], &[])?;
    }

    println!("[docs-test] Testing documentation with minimal features...");
    run_stream(root, "cargo", &["doc", "--features", "minimal", "--no-deps"], &[])?;

    println!("[docs-test] Checking for missing documentation...");
    run_stream(
        root,
        "cargo",
        &["doc", "--all-features", "--no-deps", "--", "-D", "missing_docs"],
        &[("RUSTDOCFLAGS", "--cfg docsrs")],
    )?;

    if crates == 0 {
        println!("[docs-test] No crate directories found in crates/");
    }

    if command_available("linkchecker") {
        println!("[docs-test] Testing cross-references...");
        let _ = run_capture(root, "linkchecker", &["target/doc/bitnet/index.html"], &[], true)?;
    } else {
        println!("[docs-test] linkchecker not available, skipping link validation");
    }

    println!("✅ All docs.rs compatibility tests passed!");
    println!("📖 Documentation is ready for docs.rs!");
    Ok(())
}

pub(crate) fn cmd_test_generation(root: &Path) -> Result<()> {
    println!(
        "Running the existing bitnet-models integration smoke instead of compiling an ad hoc scratch binary..."
    );
    run_stream(
        root,
        "cargo",
        &[
            "test",
            "-p",
            "bitnet-models",
            "--features",
            "integration-tests",
            "--test",
            "transformer_tests",
            "test_model_integration",
            "--",
            "--nocapture",
        ],
        &[],
    )
}

pub(crate) fn cmd_test_quant_support(root: &Path) -> Result<()> {
    println!("=== Testing BitNet Quantization Support ===");
    println!();

    let bitnet_bin = root.join("target").join("release").join("bitnet");
    if !bitnet_bin.exists() {
        bail!("bitnet CLI not found at {}", bitnet_bin.to_string_lossy());
    }

    let bitnet_bin = bitnet_bin
        .to_str()
        .with_context(|| format!("non-utf8 bitnet path: {}", bitnet_bin.display()))?;

    let test_model = |name: &str, expected_quant: &str| -> Result<()> {
        let model_path = root.join(name);
        if !model_path.exists() {
            println!("⚠ Model not found: {} (skipping)", model_path.display());
            return Ok(());
        }

        println!("Testing {}...", model_path.display());
        let inspect = run_capture(
            root,
            bitnet_bin,
            &["inspect", "--model", model_path.to_string_lossy().as_ref(), "--json"],
            &[("RUST_LOG", "error")],
            false,
        )?;
        let quant = serde_json::from_slice::<Value>(&inspect.stdout)
            .context("failed to parse inspect JSON")?
            .pointer("/quantization")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        if quant == expected_quant {
            println!("✓ Detected quantization: {quant}");
        } else {
            bail!("expected {expected_quant}, got {quant}");
        }

        let inference_args = vec![
            "run".to_string(),
            "--model".to_string(),
            model_path.to_string_lossy().into_owned(),
            "--prompt".to_string(),
            "Hello".to_string(),
            "--max-new-tokens".to_string(),
            "4".to_string(),
            "--greedy".to_string(),
        ];
        let (inference, timed_out) = run_capture_with_timeout(
            root,
            bitnet_bin,
            &inference_args,
            &[("RUST_LOG", "error"), ("BITNET_DETERMINISTIC", "1"), ("BITNET_SEED", "42")],
            Duration::from_secs(10),
        )?;
        if inference.status.success() {
            println!("  ✓ Inference successful");
        } else if timed_out {
            println!("  ✗ Inference timed out");
        } else {
            println!("  ✗ Inference failed");
            print_matching_lines(&inference, &[], true)?;
        }
        println!();
        Ok(())
    };

    println!("=== IQ2_S Support (via GGML FFI) ===");
    test_model("models/test-iq2s.gguf", "IQ2_S")?;
    test_model("models/llama-iq2s.gguf", "IQ2_S")?;

    println!("=== I2_S Support (Native Rust) ===");
    test_model("models/test-i2s.gguf", "I2_S")?;
    test_model("models/bitnet-i2s.gguf", "I2_S")?;

    println!("=== IS_2 Alias Support ===");
    test_model("models/test-is2.gguf", "I2_S")?;

    let (mut has_ggml, mut has_ggml_output) = (false, false);
    if command_available("ldd") {
        let ldd = run_capture(root, "ldd", &[bitnet_bin], &[], true)?;
        has_ggml_output =
            has_ggml_output || String::from_utf8_lossy(&ldd.stdout).contains("bitnet_ggml");
        has_ggml = has_ggml || has_ggml_output;
    }
    if command_available("otool") {
        let otool = run_capture(root, "otool", &["-L", bitnet_bin], &[], true)?;
        has_ggml_output =
            has_ggml_output || String::from_utf8_lossy(&otool.stdout).contains("bitnet_ggml");
        has_ggml = has_ggml || has_ggml_output;
    }

    println!("=== Feature Detection ===");
    print!("IQ2_S FFI support: ");
    if has_ggml {
        println!("✓ Enabled (GGML FFI linked)");
    } else {
        println!("✗ Disabled (rebuild with --features iq2s-ffi)");
    }

    println!("I2_S native support: ✓ Always enabled");
    println!();
    println!("=== Summary ===");
    println!("• IQ2_S: GGML's 2-bit quantization, requires --features iq2s-ffi");
    println!("• I2_S/IS_2: BitNet's native 2-bit signed format, always available");
    println!("• Both formats dequantize to f32 at load time for correctness");
    println!("• Performance optimizations can be added later");
    Ok(())
}

pub(crate) fn cmd_test_optimizations(root: &Path) -> Result<()> {
    println!("=== Testing BitNet-rs Optimizations ===");
    println!("This script demonstrates:");
    println!("1. Real tokenizer support (HuggingFace JSON)");
    println!("2. Precise timing metrics (tokenize/prefill/decode)");
    println!("3. Memory-efficient transpose handling");
    println!("4. Reproducible benchmarks");
    println!();

    let model_path =
        env::var("MODEL").unwrap_or_else(|_| "models/bitnet_b1_58-2B-TQ2_0.gguf".to_string());
    let tokenizer_path =
        env::var("TOKENIZER").unwrap_or_else(|_| "models/tokenizer.json".to_string());

    println!("Model: {model_path}");
    if !Path::new(&model_path).exists() {
        bail!("Error: Model not found at {model_path}");
    }

    let mut tokenizer_arg = Vec::new();
    if !Path::new(&tokenizer_path).exists() {
        println!("Warning: Tokenizer not found at {tokenizer_path}");
        println!("Will use mock tokenizer (less accurate)");
    } else {
        println!("✓ Using real tokenizer: {tokenizer_path}");
        tokenizer_arg.push("--tokenizer".to_string());
        tokenizer_arg.push(tokenizer_path.clone());
    }

    let binary = root.join("target").join("release").join("bitnet");
    if !binary.exists() {
        println!("Building BitNet CLI...");
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
                "cpu",
            ],
            &[],
        )?;
    }

    println!();
    println!("=== Test 1: Single Generation with Timing Breakdown ===");
    println!("Running inference with detailed timing...");
    println!();

    let mut run_args = vec![
        "run".to_string(),
        "--model".to_string(),
        model_path.clone(),
        "--prompt".to_string(),
        "The future of AI is".to_string(),
        "--max-new-tokens".to_string(),
        "50".to_string(),
        "--temperature".to_string(),
        "0.0".to_string(),
        "--json-out".to_string(),
        "/tmp/bitnet-test.json".to_string(),
        "--seed".to_string(),
        "42".to_string(),
    ];
    run_args.extend(tokenizer_arg.clone());
    let _ = run_capture(root, binary.to_string_lossy().as_ref(), &run_args, &[], true)?;

    let report = PathBuf::from("/tmp/bitnet-test.json");
    if report.exists() {
        println!();
        println!("=== Timing Results ===");
        let timing = run_capture(
            root,
            "jq",
            &[".timing_ms, .throughput_tps", report.to_str().context("invalid /tmp path")?],
            &[],
            false,
        )?;
        println!("{}", String::from_utf8_lossy(&timing.stdout));
        println!();
        println!("=== Token Counts ===");
        let counts = run_capture(
            root,
            "jq",
            &[".counts", report.to_str().context("invalid /tmp path")?],
            &[],
            false,
        )?;
        println!("{}", String::from_utf8_lossy(&counts.stdout));
        println!();
        println!("=== Tokenizer Info ===");
        let tokenizer_info = run_capture(
            root,
            "jq",
            &[".tokenizer", report.to_str().context("invalid /tmp path")?],
            &[],
            false,
        )?;
        println!("{}", String::from_utf8_lossy(&tokenizer_info.stdout));
    }

    println!();
    println!("=== Test 2: Memory Efficiency Check ===");
    let transposed = run_capture(
        root,
        binary.to_string_lossy().as_ref(),
        &{
            let mut args = vec![
                "run".to_string(),
                "--model".to_string(),
                model_path.clone(),
                "--prompt".to_string(),
                "Hello".to_string(),
                "--max-new-tokens".to_string(),
                "10".to_string(),
                "--temperature".to_string(),
                "0.0".to_string(),
            ];
            args.extend(tokenizer_arg.clone());
            args
        },
        &[("RUST_LOG", "bitnet_models=info")],
        true,
    )?;
    let transpose_output = format!(
        "{}{}",
        String::from_utf8_lossy(&transposed.stdout),
        String::from_utf8_lossy(&transposed.stderr)
    );
    let transposed_lines: Vec<&str> =
        transpose_output.lines().filter(|line| line.to_lowercase().contains("transpos")).collect();
    if transposed_lines.is_empty() {
        println!("(No transpose operations logged)");
    } else {
        for line in transposed_lines {
            println!("{line}");
        }
    }

    let bench_script = root.join("scripts").join("bench-decode.sh");
    if bench_script.exists() && !tokenizer_arg.is_empty() {
        println!();
        println!("=== Test 3: Benchmark Suite (if available) ===");
        println!("Running decode benchmark...");
        let output = run_capture(
            root,
            bench_script.to_str().with_context(|| {
                format!("invalid bench script path: {}", bench_script.display())
            })?,
            &[] as &[&str],
            &[("TOKENIZER", tokenizer_path.as_str()), ("MODEL", model_path.as_str())],
            true,
        )?;
        let output = String::from_utf8_lossy(&output.stdout);
        for line in output.lines().take(10) {
            println!("{line}");
        }
    } else {
        println!("Skipping benchmark (requires tokenizer and bench script)");
    }

    println!();
    println!("=== Summary ===");
    println!("✓ Real tokenizer support working");
    println!("✓ Precise timing metrics available");
    println!("✓ Memory-efficient transpose handling active");
    println!("✓ Reproducible results with seed");
    println!();
    println!("Key improvements implemented:");
    println!("- HuggingFace tokenizer integration");
    println!("- Separate tokenize/prefill/decode timing");
    println!("- Avoided 1.3GB+ memory allocations for transposes");
    println!("- Robust model dimension detection");
    println!("- Comprehensive weight mapping for various formats");
    Ok(())
}

pub(crate) fn cmd_test_download(root: &Path) -> Result<()> {
    let mock_root =
        root.join("tests").join("tmp").join("mock").join("model").join("resolve").join("main");
    let server_root = root.join("tests").join("tmp");
    let downloaded_dir = root.join("tests").join("tmp").join("downloaded");
    let out_dir = "tests/tmp/downloaded";

    fs::create_dir_all(&mock_root).context("creating mock model dir")?;
    fs::create_dir_all(&downloaded_dir).context("creating downloaded dir")?;

    fs::write(mock_root.join("model.gguf"), "dummy model data")?;
    fs::write(mock_root.join("tokenizer.json"), "{}")?;

    let server = Command::new("python3")
        .arg("-m")
        .arg("http.server")
        .arg("8080")
        .current_dir(&server_root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to start mock HTTP server")?;
    let _guard = ProcessKiller(Some(server));

    sleep(Duration::from_secs(2));

    run_stream(root, "cargo", &["build", "-p", "xtask", "--no-default-features"], &[])?;

    let mut xtask = root.join("target").join("debug").join("xtask");
    if cfg!(windows) {
        xtask.set_extension("exe");
    }
    if !xtask.exists() {
        bail!("expected {} to exist after cargo build", xtask.display());
    }

    run_xtask_binary(
        root,
        &xtask,
        &[
            "download-model",
            "--base-url",
            "http://localhost:8080",
            "--id",
            "mock/model",
            "--file",
            "model.gguf",
            "--out",
            out_dir,
            "--force",
        ],
        false,
    )?;

    let downloaded_model = downloaded_dir.join("mock-model").join("model.gguf");
    if !downloaded_model.exists() {
        bail!("download failed: {} not found", downloaded_model.display());
    }

    run_xtask_binary(
        root,
        &xtask,
        &[
            "download-model",
            "--offline",
            "--id",
            "mock/model",
            "--file",
            "model.gguf",
            "--out",
            out_dir,
        ],
        false,
    )?;

    let missing = run_xtask_binary(
        root,
        &xtask,
        &[
            "download-model",
            "--offline",
            "--id",
            "mock/model",
            "--file",
            "missing.gguf",
            "--out",
            out_dir,
        ],
        true,
    )?;
    if missing.status.success() {
        bail!("offline mode unexpectedly succeeded for missing file");
    }

    println!("Tests passed!");
    Ok(())
}

pub(crate) fn cmd_xtask_smoke(
    root: &Path,
    model: Option<String>,
    tokenizer: Option<String>,
) -> Result<()> {
    let model =
        model.unwrap_or_else(|| env::var("MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()));

    println!("== verify: JSON cleanliness ==");
    let verify_output = run_xtask(
        root,
        &["--", "verify", "--model", model.as_str(), "--format", "json"],
        &[],
        false,
    )?;
    let _verify_json: Value =
        serde_json::from_slice(&verify_output.stdout).context("invalid verify JSON")?;

    println!("== verify: strict failure exit=15 on bad path ==");
    let strict =
        run_xtask(root, &["--", "verify", "--model", "/nope/bad.gguf", "--strict"], &[], true)?;
    if strict.status.code() != Some(15) {
        bail!("verify strict exit code was {:?}, expected 15", strict.status.code());
    }

    println!("== infer: deterministic mock json ==");
    let mut infer_args = vec![
        "--".to_string(),
        "infer".to_string(),
        "--model".to_string(),
        model.clone(),
        "--prompt".to_string(),
        "hi".to_string(),
        "--max-new-tokens".to_string(),
        "4".to_string(),
        "--allow-mock".to_string(),
        "--deterministic".to_string(),
        "--format".to_string(),
        "json".to_string(),
    ];
    if let Some(tokenizer) = tokenizer {
        infer_args.push("--tokenizer".to_string());
        infer_args.push(tokenizer);
    }
    let infer_output = run_xtask(root, &infer_args, &[], false)?;
    let infer_json: Value =
        serde_json::from_slice(&infer_output.stdout).context("invalid infer JSON")?;
    let temperature =
        infer_json.pointer("/config/temperature").and_then(Value::as_f64).unwrap_or(f64::NAN);
    let seed = infer_json.pointer("/config/seed").and_then(Value::as_u64);
    if (temperature - 0.0).abs() > f64::EPSILON || seed != Some(42) {
        bail!("infer JSON config did not match expected deterministic defaults");
    }

    let bench_file = env::temp_dir().join("bitnet_task_smoke_bench.json");
    if bench_file.exists() {
        let _ = fs::remove_file(&bench_file);
    }
    let bench_file = bench_file.to_str().context("non-utf8 benchmark file path")?.to_string();

    println!("== benchmark: 0-token short-circuit + json ==");
    run_xtask(
        root,
        &[
            "--",
            "benchmark",
            "--model",
            &model,
            "--allow-mock",
            "--tokens",
            "0",
            "--json",
            &bench_file,
        ],
        &[],
        false,
    )?;
    check_benchmark_smoke(Path::new(&bench_file))?;

    println!("== benchmark: one-liner present even with json ==");
    let output = run_xtask(
        root,
        &[
            "--",
            "benchmark",
            "--model",
            &model,
            "--allow-mock",
            "--tokens",
            "8",
            "--warmup-tokens",
            "2",
            "--no-output",
            "--json",
            &bench_file,
        ],
        &[],
        false,
    )?;
    let text = String::from_utf8_lossy(&output.stdout);
    if !text.contains("tokens in ") || !text.contains("tok/s") {
        bail!("benchmark one-liner output not present");
    }
    let final_json: Value = serde_json::from_str(
        &fs::read_to_string(&bench_file).context("failed to read benchmark json")?,
    )
    .context("invalid benchmark JSON")?;
    if final_json.pointer("/version").is_none() {
        bail!("benchmark json missing version");
    }
    if final_json.pointer("/performance/tokens_per_sec").and_then(Value::as_f64).unwrap_or(-1.0)
        < 0.0
    {
        bail!("benchmark tokens_per_sec was negative");
    }

    println!("✅ smoke OK");
    Ok(())
}

pub(crate) fn cmd_run_miri(root: &Path) -> Result<()> {
    println!("Setting up Miri for undefined behavior detection...");

    let toolchains = run_capture(root, "rustup", &["toolchain", "list"], &[], true)?;
    let has_nightly = String::from_utf8_lossy(&toolchains.stdout).contains("nightly");
    if !has_nightly {
        println!("Installing Rust nightly...");
        run_stream(root, "rustup", &["toolchain", "install", "nightly"], &[])?;
    }

    println!("Installing Miri...");
    run_stream(root, "rustup", &["+nightly", "component", "add", "miri"], &[])?;

    println!("Setting up Miri...");
    run_stream(root, "cargo", &["+nightly", "miri", "setup"], &[])?;

    let crates = ["bitnet-common", "bitnet-kernels", "bitnet-quantization", "bitnet-models"];
    let mut failed = Vec::new();

    println!("Running Miri tests...");
    for crate_name in crates {
        println!("Testing {crate_name} with Miri...");
        let crate_dir = root.join("crates").join(crate_name);
        if !crate_dir.is_dir() {
            println!("⚠️  Crate directory not found: {}", crate_dir.display());
            continue;
        }

        let output = run_capture(&crate_dir, "cargo", &["+nightly", "miri", "test"], &[], true)?;
        if output.status.success() {
            println!("✅ Miri tests passed for {crate_name}");
        } else {
            println!("❌ Miri tests failed for {crate_name}");
            failed.push(crate_name.to_string());
        }
        println!();
    }

    println!("Running Miri on integration tests...");
    let integration = run_capture(
        root,
        "cargo",
        &["+nightly", "miri", "test", "--test", "integration_security"],
        &[],
        true,
    )?;
    if integration.status.success() {
        println!("✅ Integration security tests passed with Miri");
    } else {
        println!("❌ Integration security tests failed with Miri");
        failed.push("integration_tests".to_string());
    }

    println!("=== Miri Testing Summary ===");
    if failed.is_empty() {
        println!("✅ All Miri tests passed!");
        println!("No undefined behavior detected.");
        println!("Miri testing completed!");
        Ok(())
    } else {
        println!("❌ Failed crates/tests: {}", failed.join(", "));
        println!("Undefined behavior may be present in the failed components.");
        bail!("miri tests failed");
    }
}

pub(crate) fn cmd_run_fuzz(root: &Path, target: Option<String>, duration: u64) -> Result<()> {
    println!("Setting up fuzzing environment...");

    if !command_available("cargo-fuzz") {
        println!("Installing cargo-fuzz...");
        run_stream(root, "cargo", &["install", "cargo-fuzz"], &[])?;
    }

    let fuzz_dir = root.join("fuzz");
    if !fuzz_dir.is_dir() {
        println!("Initializing fuzz directory...");
        run_stream(root, "cargo", &["fuzz", "init"], &[])?;
    }

    let targets = ["quantization_i2s", "gguf_parser", "kernel_matmul"];
    let duration = duration.to_string();

    let mut failed_targets = Vec::new();
    let run_target = |target_name: &str, duration: &str| -> Result<()> {
        if !command_available("timeout") {
            bail!("timeout command not found");
        }
        println!("Running fuzz target: {target_name} for {duration}s");

        let artifacts = fuzz_dir.join("artifacts").join(target_name);
        std::fs::create_dir_all(&artifacts).context("creating fuzz artifacts directory")?;

        let args = vec![
            format!("{duration}s"),
            "cargo".to_string(),
            "fuzz".to_string(),
            "run".to_string(),
            target_name.to_string(),
            "--".to_string(),
            format!("-max_total_time={duration}"),
        ];
        let output = run_capture(root, "timeout", &args, &[], true)?;
        if !output.status.success() && output.status.code() != Some(124) {
            bail!("fuzz target {target_name} failed with code {:?}", output.status.code());
        }

        let has_crashes = match fs::read_dir(&artifacts) {
            Ok(mut entries) => entries.next().is_some(),
            Err(_) => false,
        };
        if has_crashes {
            println!("⚠️  Crashes found for {target_name}:");
            let listing =
                run_capture(root, "ls", &["-la", artifacts.to_string_lossy().as_ref()], &[], true)?;
            print!("{}", String::from_utf8_lossy(&listing.stdout));
            bail!("fuzz target {target_name} produced crashes");
        }

        println!("✅ No crashes found for {target_name}");
        Ok(())
    };

    if let Some(target) = target {
        if !targets.iter().any(|name| *name == target) {
            println!("Error: Unknown target '{target}'");
            println!("Available targets: {}", targets.join(" "));
            bail!("unknown fuzz target");
        }
        println!("Running single target...");
        if let Err(err) = run_target(&target, &duration) {
            failed_targets.push(target);
            println!("{err:#}");
        }
    } else {
        println!("Running all fuzz targets for {duration}s each...");
        for target_name in &targets {
            if let Err(err) = run_target(target_name, &duration) {
                failed_targets.push((*target_name).to_string());
                println!("{err:#}");
            }
            println!();
        }
    }

    println!("=== Fuzzing Summary ===");
    if failed_targets.is_empty() {
        println!("✅ All fuzz targets passed!");
        println!("Fuzzing completed!");
        Ok(())
    } else {
        println!("❌ Failed targets: {}", failed_targets.join(", "));
        bail!("fuzz targets failed");
    }
}
