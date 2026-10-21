use super::*;
use sha2::{Digest, Sha256};
use std::io::{BufReader, Read};

pub(crate) fn cmd_bitnet_accept(root: &Path, model: &str, tokenizer: &str) -> Result<()> {
    println!("== verify (human) ==");
    run_stream(
        root,
        "cargo",
        &[
            "run",
            "-p",
            "xtask",
            "--",
            "verify",
            "--model",
            model,
            "--tokenizer",
            tokenizer,
            "--format",
            "human",
        ],
        &[],
    )?;

    println!("== infer (auto template, 1 step, DEBUG_ATTN=1) ==");
    run_stream(
        root,
        "cargo",
        &[
            "run",
            "-p",
            "xtask",
            "--features",
            "inference",
            "--",
            "infer",
            "--model",
            model,
            "--tokenizer",
            tokenizer,
            "--prompt",
            "The capital of France is",
            "--max-new-tokens",
            "1",
            "--deterministic",
            "--template",
            "auto",
        ],
        &[("DEBUG_ATTN", "1")],
    )?;

    println!("== benchmark (prefill vs decode) ==");
    let bench_file = env::temp_dir().join("bitnet_task_bench.json");
    let bench_file = bench_file.to_string_lossy().into_owned();
    let _ = fs::remove_file(&bench_file);
    run_stream(
        root,
        "cargo",
        &[
            "run",
            "-p",
            "xtask",
            "--features",
            "inference",
            "--",
            "benchmark",
            "--model",
            model,
            "--tokenizer",
            tokenizer,
            "--prompt",
            "Write two short lines.",
            "--tokens",
            "64",
            "--warmup-tokens",
            "16",
            "--no-output",
            "--json",
            &bench_file,
        ],
        &[],
    )?;

    let output = fs::read_to_string(bench_file).context("failed to read benchmark json output")?;
    let json: Value = serde_json::from_str(&output).context("invalid benchmark json output")?;
    let tps = json.pointer("/performance/tokens_per_sec");
    let prefill = json.pointer("/timing/prefill_ms");
    let decode = json.pointer("/timing/decode_ms");
    println!("tokens_per_sec:");
    println!("{}", tps.unwrap_or(&Value::Null));
    println!("prefill/decode ms:");
    println!("{} {}", prefill.unwrap_or(&Value::Null), decode.unwrap_or(&Value::Null));
    Ok(())
}

pub(crate) fn cmd_resolve_model_path(root: &Path, model: &str) -> Result<()> {
    let direct = root.join(model);
    if direct.is_file() {
        println!("{}", direct.display());
        return Ok(());
    }
    let nested = root.join("models").join(model);
    if nested.is_file() {
        println!("{}", nested.display());
        return Ok(());
    }
    bail!("Error: model not found: {model}");
}

pub(crate) fn cmd_vendor_ggml_quants(root: &Path, commit: String) -> Result<()> {
    let commit = if commit.is_empty() { "master".to_string() } else { commit };
    println!("Vendoring GGML quants from commit: {commit}");

    if commit == "master" {
        println!(
            "Warning: Using master branch. Consider pinning a specific commit for reproducibility."
        );
        println!("Example: bitnet-task vendor-ggml-quants <commit>");
    }

    let base = format!("https://raw.githubusercontent.com/ggerganov/llama.cpp/{commit}");
    let dest = root.join("crates").join("bitnet-ggml-ffi").join("csrc");
    println!("Fetching to: {}", dest.display());

    fs::create_dir_all(&dest).context("creating GGML destination directory")?;

    let files = [
        ("ggml.h", "ggml.h"),
        ("ggml-quants.h", "ggml-quants.h"),
        ("ggml-quants.c", "ggml-quants.c"),
    ];
    for (remote, local) in files {
        println!("Downloading {local}...");
        let url = format!("{base}/{remote}");
        run_stream(
            root,
            "curl",
            &["-fsSL", url.as_str(), "-o", dest.join(local).to_string_lossy().as_ref()],
            &[],
        )?;
    }

    fs::write(dest.join("GGML_VERSION"), format!("{commit}\n"))
        .context("writing GGML_VERSION file")?;

    println!("Successfully vendored GGML quants from commit {commit}");
    println!("Files written to: {}/", dest.display());
    for entry in fs::read_dir(&dest)? {
        let entry = entry?;
        println!("{}", entry.path().display());
    }
    Ok(())
}

pub(crate) fn cmd_generate_policy(root: &Path, model: &str, output: &str) -> Result<()> {
    let model_path =
        if Path::new(model).is_absolute() { PathBuf::from(model) } else { root.join(model) };
    if !model_path.exists() {
        bail!("Model file not found: {}", model_path.display());
    }

    let digest = sha256_hex(&model_path)?;
    let fingerprint = format!("sha256-{digest}");

    let output_path =
        if Path::new(output).is_absolute() { PathBuf::from(output) } else { root.join(output) };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).context("creating output directory")?;
    }

    println!("Computing fingerprint for: {}", model_path.display());
    println!("Fingerprint: {fingerprint}");

    let policy = format!(
        "version: 1\nmodels:\n  - fingerprint: \"{fingerprint}\"\n    notes: \"BitNet model with inverted I2_S scales in Q/K/V projections\"\n    corrections:\n      # Override I2_S dequantization for attention projections\n      # Use inv=true to invert the scales (1/scale instead of scale)\n      - type: I2S_DEQUANT_OVERRIDE\n        tensors:\n          # LLaMA/HF-style names\n          - \"q_proj.weight\"\n          - \"k_proj.weight\"\n          - \"v_proj.weight\"\n          # Microsoft BitNet-style names\n          - \"wq.weight\"\n          - \"wk.weight\"\n          - \"wv.weight\"\n          # Alternative naming patterns\n          - \"attn_q.weight\"\n          - \"attn_k.weight\"\n          - \"attn_v.weight\"\n        inv: true\n        k: 1.0\n"
    );
    fs::write(&output_path, policy)
        .with_context(|| format!("writing {}", output_path.display()))?;
    println!("Policy file written to: {}", output_path.display());
    println!();
    println!("To use this policy:");
    println!("  export BITNET_CORRECTION_POLICY={}", output_path.display());
    println!("  export BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1");
    println!("  RUST_LOG=info,bitnet_models=debug ./scripts/debug_inference.sh \\");
    println!("    \"{}\" \\", model_path.to_string_lossy());
    println!("    models/llama3-tokenizer/tokenizer.json \\");
    println!("    \"Answer in one short sentence: Why is the sky blue?\"");
    Ok(())
}

fn sha256_hex(path: &Path) -> Result<String> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];

    loop {
        let read =
            reader.read(&mut buffer).with_context(|| format!("reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn cmd_show_quant_status(root: &Path) -> Result<()> {
    let mut cli_bin = root.join("target").join("release").join("bitnet");
    if cfg!(windows) {
        cli_bin.set_extension("exe");
    }
    if !cli_bin.exists() {
        println!("⚠️  Release binary not found. Building...");
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
                "cpu,iq2s-ffi",
            ],
            &[],
        )?;
    }

    println!("=== BitNet-rs Quantization Support Status ===");
    println!();
    println!("📊 Quantization Format Support:");
    println!("================================");
    println!();
    println!("✅ I2_S (BitNet Native 2-bit signed)");
    println!("   - Implementation: Pure Rust");
    println!("   - Dependencies: None");
    println!("   - Feature flag: Always available with 'cpu'");
    println!("   - Block size: 256 elements, 66 bytes");
    println!("   - Status: FULLY IMPLEMENTED");
    println!();
    println!("✅ IQ2_S (GGML/llama.cpp compatible)");
    println!("   - Implementation: GGML FFI (C bridge)");
    println!("   - Dependencies: Vendored GGML files");
    println!("   - Feature flag: iq2s-ffi");
    println!("   - Block size: Determined at runtime from GGML");
    println!("   - Status: IMPLEMENTED (needs real GGML vendor)");
    println!();
    println!("🔄 Other Formats (Planned):");
    println!("   - Q4_0, Q4_1: 4-bit quantization");
    println!("   - Q5_0, Q5_1: 5-bit quantization");
    println!("   - Q8_0: 8-bit quantization");
    println!("   - K-quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K");
    println!();
    println!("📦 Build Configurations:");
    println!("========================");
    println!();
    println!("1. CPU with I2_S only (no external deps):");
    println!("   cargo build --release --no-default-features --features cpu");
    println!();
    println!("2. CPU with both I2_S and IQ2_S:");
    println!("   cargo build --release --no-default-features --features 'cpu,iq2s-ffi'");
    println!();
    println!("3. GPU/CUDA support:");
    println!("   cargo build --release --no-default-features --features cuda");
    println!();
    println!("🧪 Testing Quantization:");
    println!("=======================");
    println!();
    println!("# Run I2_S tests (native):");
    println!("cargo test -p bitnet-models --tests -- i2s");
    println!();
    println!("# Run IQ2_S tests (FFI):");
    println!("cargo test -p bitnet-models --tests --features iq2s-ffi -- iq2s");
    println!();
    println!("# Test with a model:");
    println!("BITNET_DETERMINISTIC=1 BITNET_SEED=42 \\");
    println!("  {} run --model <path/to/model.gguf> \\", cli_bin.display());
    println!("  --prompt 'Hello' --max-new-tokens 8");
    println!();
    println!("⚠️  Important Notes:");
    println!("===================");
    println!("- IQ2_S currently uses stub GGML implementation");
    println!("- Run 'cargo xtask vendor-ggml --commit <sha>' to get real GGML");
    println!("- Both I2_S and IQ2_S dequantize to f32 at load time (correctness-first)");
    println!("- Performance optimizations (on-the-fly dequant) planned post-alpha");
    println!();
    println!("✨ Next Steps:");
    println!("=============");
    println!("1. Vendor real GGML files from llama.cpp");
    println!("2. Add pure-Rust IQ2_S implementation");
    println!("3. Run parity tests between FFI and Rust paths");
    println!("4. Enable on-the-fly dequantization for memory efficiency");
    Ok(())
}
