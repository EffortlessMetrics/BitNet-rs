use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
};
use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, RANGE};
use sha2::{Digest, Sha256};

#[derive(Parser)]
#[command(name = "xtask", about = "Developer tasks for BitNet.rs")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download a GGUF model from Hugging Face (supports HF_TOKEN)
    DownloadModel {
        /// HF repo id (e.g., microsoft/bitnet-b1.58-2B-4T-gguf)
        #[arg(long, default_value = "microsoft/bitnet-b1.58-2B-4T-gguf")]
        id: String,
        /// File within repo (e.g., ggml-model-i2_s.gguf)
        #[arg(long, default_value = "ggml-model-i2_s.gguf")]
        file: String,
        /// Output directory
        #[arg(long, default_value = "models")]
        out: PathBuf,
        /// Optional expected sha256 (hex)
        #[arg(long)]
        sha256: Option<String>,
        /// Overwrite if exists
        #[arg(long, default_value_t = false)]
        force: bool,
    },

    /// Fetch & build microsoft/BitNet C++ (pinned tag)
    FetchCpp {
        /// Tag or rev to fetch (default: b1-65-ggml)
        #[arg(long, default_value = "b1-65-ggml")]
        tag: String,
        /// Force rebuild
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Clean rebuild
        #[arg(long, default_value_t = false)]
        clean: bool,
    },

    /// Run deterministic cross-validation tests
    Crossval {
        /// Path to GGUF model (default: models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf)
        #[arg(long)]
        model: Option<PathBuf>,
        /// Path to C++ checkout (default: $HOME/.cache/bitnet_cpp)
        #[arg(long)]
        cpp_dir: Option<PathBuf>,
        /// Release build
        #[arg(long, default_value_t = true)]
        release: bool,
        /// Extra args to pass to cargo test after `--`
        #[arg(last = true)]
        extra: Vec<String>,
    },

    /// Run full cross-validation workflow (download + fetch + test)
    FullCrossval {
        /// Force redownload/rebuild
        #[arg(long, default_value_t = false)]
        force: bool,
    },

    /// Generate test fixtures (keeping existing functionality)
    GenFixtures {
        /// Size of fixture (tiny, small, medium)
        #[arg(long, default_value = "small")]
        size: String,
        /// Output directory
        #[arg(long, default_value = "crossval/fixtures/")]
        output: PathBuf,
    },

    /// Setup cross-validation environment
    SetupCrossval,

    /// Clean all caches
    CleanCache,

    /// Check feature flag consistency
    CheckFeatures,

    /// Run performance benchmarks
    Benchmark {
        /// Platform to test
        #[arg(long, default_value = "current")]
        platform: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::DownloadModel {
            id,
            file,
            out,
            sha256,
            force,
        } => download_model_cmd(&id, &file, &out, sha256.as_deref(), force),
        Cmd::FetchCpp { tag, force, clean } => fetch_cpp_cmd(&tag, force, clean),
        Cmd::Crossval {
            model,
            cpp_dir,
            release,
            extra,
        } => {
            let model_path = model.unwrap_or_else(|| {
                PathBuf::from("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf")
            });
            crossval_cmd(&model_path, cpp_dir.as_deref(), release, &extra)
        }
        Cmd::FullCrossval { force } => full_crossval_cmd(force),
        Cmd::GenFixtures { size, output } => gen_fixtures(&size, &output),
        Cmd::SetupCrossval => setup_crossval(),
        Cmd::CleanCache => clean_cache(),
        Cmd::CheckFeatures => check_features(),
        Cmd::Benchmark { platform } => run_benchmark(&platform),
    }
}

fn download_model_cmd(
    id: &str,
    file: &str,
    out_dir: &Path,
    sha256_hex: Option<&str>,
    force: bool,
) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    let dest_dir = out_dir.join(id.replace('/', "-"));
    fs::create_dir_all(&dest_dir)?;
    let dest = dest_dir.join(file);

    let url = format!("https://huggingface.co/{id}/resolve/main/{file}");
    let token = std::env::var("HF_TOKEN").ok();

    if dest.exists() && !force {
        eprintln!("✓ File exists: {} (use --force to overwrite)", dest.display());
        if let Some(want) = sha256_hex {
            verify_sha256(&dest, want)?;
            println!("✓ SHA256 verified");
        }
        return Ok(());
    }

    println!("📥 Downloading from Hugging Face:");
    println!("   Repository: {}", id);
    println!("   File: {}", file);
    println!("   Destination: {}", dest.display());
    if token.is_some() {
        println!("   Using HF_TOKEN for authentication");
    }

    let client = Client::builder()
        .timeout(None)
        .build()?;
    
    // HEAD request to get file size
    let mut head_req = client.head(&url);
    if let Some(t) = &token {
        head_req = head_req.header(AUTHORIZATION, format!("Bearer {t}"));
    }
    
    let size = head_req
        .send()
        .and_then(|r| r.error_for_status())
        .ok()
        .and_then(|r| {
            r.headers()
                .get(CONTENT_LENGTH)?
                .to_str()
                .ok()?
                .parse::<u64>()
                .ok()
        });

    let tmp = dest.with_extension("part");
    let mut start = 0u64;
    
    // Check for partial download
    if tmp.exists() {
        start = fs::metadata(&tmp)?.len();
        if let Some(total) = size {
            println!("   Resuming from {:.2} MB / {:.2} MB", 
                start as f64 / 1_048_576.0,
                total as f64 / 1_048_576.0);
        }
    }
    
    let mut req = client.get(&url);
    if let Some(t) = &token {
        req = req.header(AUTHORIZATION, format!("Bearer {t}"));
    }
    if start > 0 {
        req = req.header(RANGE, format!("bytes={start}-"));
    }
    
    let mut resp = req.send()?.error_for_status()?;

    let pb = ProgressBar::new(size.unwrap_or(0));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}"
        )?
        .progress_chars("##-")
    );
    
    if start > 0 {
        pb.set_position(start);
        pb.set_message("resuming");
    }

    let mut file_out = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(start > 0)
        .open(&tmp)?;
    
    let mut downloaded = start;
    let mut buf = vec![0u8; 1024 * 256]; // 256KB buffer
    
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file_out.write_all(&buf[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }
    
    file_out.flush()?;
    pb.finish_with_message("download complete");

    fs::rename(&tmp, &dest)?;

    if let Some(want) = sha256_hex {
        print!("🔒 Verifying SHA256... ");
        std::io::stdout().flush()?;
        verify_sha256(&dest, want)?;
        println!("✓ OK");
    }
    
    println!("✅ Saved: {}", dest.display());
    if let Some(size) = size {
        println!("   Size: {:.2} MB", size as f64 / 1_048_576.0);
    }
    
    Ok(())
}

fn verify_sha256(path: &Path, expected_hex: &str) -> Result<()> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024]; // 1MB buffer
    
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    
    let got = hex::encode(hasher.finalize());
    if got != expected_hex.to_lowercase() {
        return Err(anyhow!(
            "SHA256 mismatch for {}:\n  expected {}\n  got      {}",
            path.display(),
            expected_hex,
            got
        ));
    }
    
    Ok(())
}

fn fetch_cpp_cmd(tag: &str, force: bool, clean: bool) -> Result<()> {
    let script = PathBuf::from("ci/fetch_bitnet_cpp.sh");
    if !script.exists() {
        return Err(anyhow!(
            "Script not found: {}. Are you in the BitNet-rs root directory?",
            script.display()
        ));
    }
    
    if cfg!(windows) {
        eprintln!("⚠️  Note: On Windows, run this command under WSL or Git Bash");
    }
    
    println!("🔧 Fetching Microsoft BitNet C++ implementation");
    println!("   Tag: {}", tag);
    println!("   Force: {}", force);
    println!("   Clean: {}", clean);
    
    let mut args = vec!["--tag".to_string(), tag.to_string()];
    if force {
        args.push("--force".to_string());
    }
    if clean {
        args.push("--clean".to_string());
    }
    
    run(
        "bash",
        std::iter::once(script.to_string_lossy().to_string())
            .chain(args)
            .collect(),
    )
}

fn crossval_cmd(
    model: &Path,
    cpp_dir: Option<&Path>,
    release: bool,
    extra: &[String],
) -> Result<()> {
    if !model.exists() {
        return Err(anyhow!(
            "Model not found: {}\nTip: Run `cargo xtask download-model` first",
            model.display()
        ));
    }
    
    let cpp = cpp_dir
        .map(|p| p.to_path_buf())
        .or_else(|| std::env::var_os("BITNET_CPP_DIR").map(PathBuf::from))
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap()
                .join(".cache/bitnet_cpp")
        });

    if !cpp.exists() {
        eprintln!("⚠️  Warning: BITNET_CPP_DIR not found at {}", cpp.display());
        eprintln!("   Tip: Run `cargo xtask fetch-cpp` first");
    }

    println!("🧪 Running cross-validation tests");
    println!("   Model: {}", model.display());
    println!("   C++ dir: {}", cpp.display());
    println!("   Release: {}", release);
    println!("   Deterministic: yes (single-threaded)");

    // Build the cargo test command
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .args(["-p", "bitnet-crossval", "--features", "crossval"]);
    
    if release {
        cmd.arg("--release");
    }
    
    // Set environment for determinism
    cmd.env("BITNET_CPP_DIR", &cpp)
        .env("CROSSVAL_GGUF", model)
        .env("OMP_NUM_THREADS", "1")
        .env("GGML_NUM_THREADS", "1");
    
    // Add test runner args
    cmd.arg("--")
        .args(["--nocapture", "--test-threads=1"])
        .args(extra);

    run_cmd(&mut cmd)
}

fn full_crossval_cmd(force: bool) -> Result<()> {
    println!("🚀 Running full cross-validation workflow");
    println!();
    
    // Step 1: Download model
    println!("Step 1/3: Downloading model");
    download_model_cmd(
        "microsoft/bitnet-b1.58-2B-4T-gguf",
        "ggml-model-i2_s.gguf",
        &PathBuf::from("models"),
        None, // Add SHA256 if available
        force,
    )?;
    
    println!();
    
    // Step 2: Fetch C++ implementation
    println!("Step 2/3: Fetching C++ implementation");
    fetch_cpp_cmd("b1-65-ggml", force, false)?;
    
    println!();
    
    // Step 3: Run tests
    println!("Step 3/3: Running cross-validation tests");
    let model = PathBuf::from("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
    crossval_cmd(&model, None, true, &[])?;
    
    println!();
    println!("✅ Full cross-validation workflow complete!");
    
    Ok(())
}

// Keep existing functionality from original xtask
fn gen_fixtures(size: &str, output_dir: &Path) -> Result<()> {
    println!("🔧 Generating deterministic test model fixtures...");
    println!("  Size: {}", size);
    println!("  Output: {}", output_dir.display());
    
    fs::create_dir_all(output_dir)?;
    
    // Simple fixture generation (existing logic)
    let fixture_content = format!(
        r#"{{
  "model_type": "bitnet_b1_58",
  "vocab_size": {},
  "hidden_size": {},
  "num_layers": {},
  "test_metadata": {{
    "fixture_type": "{}",
    "deterministic": true,
    "seed": 42
  }}
}}"#,
        match size {
            "tiny" => 1000,
            "small" => 5000,
            _ => 32000,
        },
        match size {
            "tiny" => 64,
            "small" => 256,
            _ => 512,
        },
        match size {
            "tiny" => 2,
            "small" => 4,
            _ => 8,
        },
        size
    );
    
    let fixture_path = output_dir.join(format!("{}_model.json", size));
    fs::write(&fixture_path, fixture_content)?;
    
    println!("✅ Created: {}", fixture_path.display());
    Ok(())
}

fn setup_crossval() -> Result<()> {
    println!("🔧 Setting up cross-validation environment...");
    
    // Generate test fixtures
    println!("  Generating test fixtures...");
    gen_fixtures("small", &PathBuf::from("crossval/fixtures/"))?;
    
    // Build with crossval features
    println!("  Building with cross-validation features...");
    let status = Command::new("cargo")
        .args(&["build", "--features", "crossval"])
        .status()?;
    
    if !status.success() {
        return Err(anyhow!("Failed to build with crossval features"));
    }
    
    println!("✅ Cross-validation environment setup complete!");
    println!();
    println!("You can now run:");
    println!("  cargo test -p bitnet-crossval --features crossval");
    
    Ok(())
}

fn clean_cache() -> Result<()> {
    println!("🧹 Cleaning all caches and temporary files...");
    
    let cache_dirs = [
        PathBuf::from("target/"),
        dirs::home_dir().unwrap().join(".cache/bitnet_cpp/"),
        PathBuf::from("crossval/fixtures/"),
    ];
    
    for dir in &cache_dirs {
        if dir.exists() {
            println!("  Cleaning: {}", dir.display());
            fs::remove_dir_all(dir)?;
            println!("    ✅ Removed");
        } else {
            println!("  Skipping: {} (does not exist)", dir.display());
        }
    }
    
    println!("✅ Cache cleanup complete!");
    Ok(())
}

fn check_features() -> Result<()> {
    println!("🔍 Checking feature flag consistency...");
    
    let cargo_toml = fs::read_to_string("Cargo.toml")?;
    
    if cargo_toml.contains("default = [") && cargo_toml.contains("\"crossval\"") {
        return Err(anyhow!(
            "crossval feature is enabled by default! This will slow down builds."
        ));
    }
    
    println!("  ✅ crossval feature is not in default features");
    println!("✅ Feature flag consistency check passed!");
    
    Ok(())
}

fn run_benchmark(platform: &str) -> Result<()> {
    println!("🚀 Running performance benchmarks...");
    println!("  Platform: {}", platform);
    
    let status = Command::new("cargo")
        .args(&["bench", "--workspace", "--features", "cpu"])
        .status()?;
    
    if !status.success() {
        return Err(anyhow!("Benchmarks failed"));
    }
    
    println!("✅ Benchmarks completed successfully!");
    Ok(())
}

fn run(bin: &str, args: Vec<String>) -> Result<()> {
    let mut cmd = Command::new(bin);
    cmd.args(args);
    run_cmd(&mut cmd)
}

fn run_cmd(cmd: &mut Command) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("Failed to spawn: {:?}", cmd))?;
    
    if !status.success() {
        return Err(anyhow!("Command failed with status: {:?}", status));
    }
    
    Ok(())
}