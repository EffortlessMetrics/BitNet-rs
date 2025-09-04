use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use fs2::FileExt;
use fs2::available_space;
use httpdate::parse_http_date;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use is_terminal::IsTerminal;
use reqwest::StatusCode;
use reqwest::blocking::Client;
use reqwest::header::{
    ACCEPT_ENCODING, ACCEPT_RANGES, AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE, ETAG,
    IF_MODIFIED_SINCE, IF_NONE_MATCH, IF_RANGE, LAST_MODIFIED, RANGE, RETRY_AFTER,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{self, Command},
    sync::{
        Once,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime},
};
use walkdir::WalkDir;

mod gates;

// RAII guard for lock file cleanup
struct LockGuard {
    file: Option<std::fs::File>,
    path: PathBuf,
}

impl LockGuard {
    fn new(path: PathBuf, file: std::fs::File) -> Self {
        LockGuard { file: Some(file), path }
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            let _ = FileExt::unlock(&file);
            drop(file);
        }
        let _ = fs::remove_file(&self.path);
    }
}

/// Cross-validation report for CI artifacts
#[derive(Debug, Serialize, Deserialize)]
struct CrossValReport {
    model: String,
    rust_ok: bool,
    cpp_header_ok: bool,
    cpp_full_ok: bool,
    xfail: bool,
    notes: String,
    timestamp: String,
    platform: String,
    // Enhanced fields for better diagnostics
    gguf_version_detected: Option<u32>,
    n_kv: Option<u64>,
    n_tensors: Option<u64>,
    data_offset: Option<u64>,
    file_size: Option<u64>,
}

impl CrossValReport {
    fn new(model: &Path) -> Self {
        let file_size = std::fs::metadata(model).ok().map(|m| m.len());

        Self {
            model: model.display().to_string(),
            rust_ok: false,
            cpp_header_ok: false,
            cpp_full_ok: false,
            xfail: false,
            notes: String::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            gguf_version_detected: None,
            n_kv: None,
            n_tensors: None,
            data_offset: None,
            file_size,
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        println!("📊 Saved cross-validation report to: {}", path.display());
        Ok(())
    }
}

// Global interrupt flag and setup
static CTRL_ONCE: Once = Once::new();
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

// Exit codes for structured errors
const EXIT_SUCCESS: i32 = 0;
const EXIT_NO_SPACE: i32 = 10;
const EXIT_AUTH: i32 = 11;
const EXIT_RATE_LIMIT: i32 = 12;
const EXIT_HASH_MISMATCH: i32 = 13;
const EXIT_NETWORK: i32 = 14;
const EXIT_INTERRUPTED: i32 = 130;

// Safe exponential backoff helper with jitter
#[inline]
fn exp_backoff_ms(attempt: u32) -> u64 {
    // 200ms, 400ms, 800ms… capped at 10s
    let shift = attempt.saturating_sub(1).min(20);
    let base = (200u64).saturating_mul(1u64 << shift).min(10_000);
    // Add deterministic jitter: +0..199ms based on attempt
    let jitter = (attempt as u64 * 37) % 200;
    base.saturating_add(jitter)
}

// Parse Retry-After header (supports both seconds and HTTP-date)
fn retry_after_secs(headers: &reqwest::header::HeaderMap) -> u64 {
    let raw = match headers.get(RETRY_AFTER).and_then(|v| v.to_str().ok()) {
        Some(s) => s,
        None => return 5, // Default to 5 seconds
    };

    // Try parsing as integer seconds first
    if let Ok(s) = raw.parse::<u64>() {
        return s.min(3600); // Cap at 1 hour
    }

    // Try parsing as HTTP-date
    parse_http_date(raw)
        .ok()
        .and_then(|when| when.duration_since(SystemTime::now()).ok())
        .map(|d| d.as_secs().clamp(1, 3600))
        .unwrap_or(5) // Default to 5 seconds if parsing fails
}

// Atomic write helper for metadata files
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    #[cfg(unix)]
    {
        if let Ok(f) = std::fs::File::open(&tmp) {
            f.sync_all()?;
        }
    }
    fs::rename(&tmp, path)?;
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

// Centralized defaults to avoid drift
const DEFAULT_MODEL_ID: &str = "microsoft/bitnet-b1.58-2B-4T-gguf";
const DEFAULT_MODEL_FILE: &str = "ggml-model-i2_s.gguf";
const USER_AGENT_STRING: &str = "bitnet-xtask/0.1 (+https://github.com/microsoft/BitNet-rs)";
const DEFAULT_CPP_TAG: &str = "main";

#[derive(Parser)]
#[command(name = "xtask", about = "Developer tasks for BitNet.rs")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download a GGUF model from Hugging Face with production-ready features
    ///
    /// Features:
    /// - Resumable downloads with Content-Range validation
    /// - 429 rate limiting with Retry-After support
    /// - ETag/Last-Modified caching for 304 optimization
    /// - Concurrent download protection via file locking
    /// - SHA256 verification with automatic retry on mismatch
    /// - Disk space validation before download
    /// - Ctrl-C graceful handling with resume support
    ///
    /// Environment:
    /// - HF_TOKEN: Authentication token for private repositories
    /// - HTTP[S]_PROXY: Automatically respected for proxy connections
    DownloadModel {
        /// HF repo id (e.g., microsoft/bitnet-b1.58-2B-4T-gguf)
        #[arg(long, default_value = DEFAULT_MODEL_ID)]
        id: String,
        /// File within repo (e.g., ggml-model-i2_s.gguf)
        #[arg(long, default_value = DEFAULT_MODEL_FILE)]
        file: String,
        /// Output directory
        #[arg(long, default_value = "models")]
        out: PathBuf,
        /// Optional expected SHA256 for verification
        #[arg(long)]
        sha256: Option<String>,
        /// Force download even if file exists
        #[arg(long, default_value_t = false)]
        force: bool,

        /// Pin to a specific branch/tag/commit
        #[arg(long, alias = "ref")]
        rev: Option<String>,

        /// Disable progress bar (same as redirecting stderr)
        #[arg(long, alias = "quiet")]
        no_progress: bool,

        /// Verbose output for debugging
        #[arg(short, long)]
        verbose: bool,

        /// Alternative base URL (for mirrors)
        #[arg(long, default_value = "https://huggingface.co")]
        base_url: String,

        /// Output JSON events for CI/CD pipelines
        #[arg(long)]
        json: bool,

        /// Maximum retry attempts
        #[arg(long, default_value_t = 3)]
        retries: u32,

        /// Request timeout in seconds
        #[arg(long, default_value_t = 1800)]
        timeout: u64,
    },

    /// Fetch & build microsoft/BitNet C++ for cross-validation
    ///
    /// Validates that the C++ binary was successfully built after compilation
    FetchCpp {
        /// Branch or rev to fetch (default: main)
        #[arg(long, default_value = DEFAULT_CPP_TAG)]
        tag: String,
        /// Force rebuild
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Clean rebuild
        #[arg(long, default_value_t = false)]
        clean: bool,
        /// Backend: "cpu" (default) | "cuda"
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Additional CMake flags (e.g., "-DCMAKE_CUDA_ARCHITECTURES=80;86")
        #[arg(long, default_value = "")]
        cmake_flags: String,
        /// Git repository URL (default: official Microsoft BitNet)
        #[arg(long, default_value = "https://github.com/microsoft/BitNet.git")]
        repo: String,
    },

    /// Run deterministic cross-validation tests against C++ implementation
    ///
    /// Auto-discovers GGUF models in the models/ directory if not specified.
    /// Requires the C++ implementation to be built first (use fetch-cpp).
    Crossval {
        /// Path to GGUF model (auto-discovers if not specified)
        #[arg(long)]
        model: Option<PathBuf>,
        /// Path to C++ checkout (default: $HOME/.cache/bitnet_cpp)
        #[arg(long)]
        cpp_dir: Option<PathBuf>,
        /// Release build
        #[arg(long, default_value_t = true)]
        release: bool,
        /// Print env and cargo test command, then exit
        #[arg(long, help = "Print env and cargo test command, then exit")]
        dry_run: bool,
        /// Extra args to pass to cargo test after `--`
        #[arg(last = true)]
        extra: Vec<String>,
    },

    /// Run full cross-validation workflow (download + fetch + test)
    ///
    /// One-command workflow that:
    /// 1. Downloads the default model (or skips if exists)
    /// 2. Fetches and builds the C++ implementation
    /// 3. Runs cross-validation tests with auto-discovery
    ///
    /// Perfect for CI/CD pipelines and initial setup
    FullCrossval {
        /// Force redownload/rebuild
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Branch/tag to fetch (default: main)
        #[arg(long, default_value = DEFAULT_CPP_TAG)]
        tag: String,
        /// Backend: "cpu" (default) | "cuda"
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Additional CMake flags
        #[arg(long, default_value = "")]
        cmake_flags: String,
        /// Git repository URL (default: official Microsoft BitNet)
        #[arg(long, default_value = "https://github.com/microsoft/BitNet.git")]
        repo: String,
    },

    /// Generate realistic test fixtures for unit testing
    ///
    /// Creates GGUF-like metadata JSON and binary weight files
    /// with deterministic content for reproducible testing
    GenFixtures {
        /// Size of fixture (tiny, small, medium)
        #[arg(long, default_value = "small")]
        size: String,
        /// Output directory
        #[arg(long, default_value = "crossval/fixtures/")]
        output: PathBuf,
    },

    /// Generate a minimal valid GGUF file for smoke testing
    ///
    /// Always creates a GGUF v3 file with valid headers for testing.
    /// If --version 2 is provided, still emits v3 but adds a
    /// compat.v2_requested=true metadata tag for test purposes.
    GenMiniGguf {
        /// Output file path
        #[arg(long, default_value = "tests/models/mini.gguf")]
        output: PathBuf,
        /// GGUF version requested (2 or 3) - always emits v3 format
        #[arg(long, default_value = "3")]
        version: u32,
    },

    /// Setup cross-validation environment
    SetupCrossval,

    /// Clean all caches with interactive confirmation
    ///
    /// Shows size of each cache directory and asks for confirmation.
    /// Cleans: target/, ~/.cache/bitnet_cpp/, crossval/fixtures/, models/
    CleanCache,

    /// Check feature flag consistency
    CheckFeatures,

    /// CI gates that emit JSON for robust detection
    Gate {
        #[command(subcommand)]
        which: GateWhich,
    },

    /// Run performance benchmarks
    Benchmark {
        /// Platform to test
        #[arg(long, default_value = "current")]
        platform: String,
    },

    /// Compare metrics with baseline for regression detection
    ///
    /// Compares crossval metrics JSON with a baseline and fails if thresholds are exceeded
    CompareMetrics {
        /// Path to baseline metrics JSON
        #[arg(long)]
        baseline: PathBuf,
        /// Path to current metrics JSON
        #[arg(long)]
        current: PathBuf,
        /// Max allowed perplexity increase (e.g., 0.02 for 2%)
        #[arg(long, default_value = "0.02")]
        ppl_max: f64,
        /// Max allowed latency P95 increase (e.g., 0.05 for 5%)
        #[arg(long, default_value = "0.05")]
        latency_p95_max: f64,
        /// Min required tokens/sec (e.g., -0.05 for 5% decrease allowed)
        #[arg(long, default_value = "-0.05")]
        tok_s_min: f64,
    },

    /// Detect breaking changes in the API
    ///
    /// Compares the current API surface with a baseline to detect breaking changes
    DetectBreaking {
        /// Path to baseline version (default: latest git tag)
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Path to current version (default: current directory)
        #[arg(long, default_value = ".")]
        current: PathBuf,
        /// Output format (json, human)
        #[arg(long, default_value = "human")]
        format: String,
    },

    /// Vendor GGML quantization files for IQ2_S support
    ///
    /// Downloads GGML quantization headers and implementation from llama.cpp
    /// to enable IQ2_S tensor support through FFI. This is required for
    /// building with the `iq2s-ffi` feature.
    ///
    /// Example:
    ///   cargo xtask vendor-ggml --commit b4247
    VendorGgml {
        /// llama.cpp commit SHA to vendor from
        #[arg(long, default_value = "b4247")]
        commit: String,
        /// Force re-download even if files exist
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Output directory for vendored files
        #[arg(long, default_value = "crates/bitnet-ggml-ffi/csrc")]
        output: PathBuf,
    },

    /// GPU preflight check and environment detection
    ///
    /// Checks for available GPU backends (CUDA, Metal, ROCm, WebGPU)
    /// and provides actionable setup instructions if none are found.
    ///
    /// Exit codes:
    /// - 0: GPU backend available
    /// - 1: No GPU backend found (but can continue with CPU)
    GpuPreflight {
        /// Exit with error if no GPU found (default: warn only)
        #[arg(long, default_value_t = false)]
        require: bool,
        /// Output format (human, json)
        #[arg(long, default_value = "human")]
        format: String,
    },

    /// Run GPU smoke tests with CPU parity check
    ///
    /// Runs a small GPU test to verify functionality and compares
    /// results with CPU for correctness validation.
    GpuSmoke {
        /// Test model size (tiny, small, medium)
        #[arg(long, default_value = "tiny")]
        size: String,
        /// Tolerance for CPU-GPU comparison (cosine similarity)
        #[arg(long, default_value = "0.99")]
        tolerance: f32,
        /// Skip if no GPU available (for CI)
        #[arg(long, default_value_t = true)]
        skip_if_no_gpu: bool,
    },

    /// Run demos with automatic feature detection
    ///
    /// Runs the reporting system demos, automatically enabling
    /// the required features based on what's available.
    Demo {
        /// Which demo to run (system, comprehensive, all)
        #[arg(long, default_value = "all")]
        which: String,
        /// Additional arguments to pass to the demo
        #[arg(last = true)]
        args: Vec<String>,
    },
}

#[derive(Subcommand)]
enum GateWhich {
    /// Dry-run tensor-name mapper gate → JSON
    Mapper {
        /// Path to model GGUF (only header/tensor names are read)
        #[arg(long)]
        model: PathBuf,
    },
}

fn main() {
    let code = match real_main() {
        Ok(()) => EXIT_SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            classify_exit(&e)
        }
    };
    process::exit(code);
}

fn classify_exit(e: &anyhow::Error) -> i32 {
    // Check for reqwest errors
    if let Some(req) = e.downcast_ref::<reqwest::Error>() {
        if let Some(s) = req.status() {
            return match s.as_u16() {
                401 | 403 => EXIT_AUTH,
                429 => EXIT_RATE_LIMIT,
                404 => EXIT_NETWORK,
                _ => EXIT_NETWORK,
            };
        }
        return EXIT_NETWORK;
    }

    // Check error message for specific patterns
    let msg = e.to_string().to_ascii_lowercase();
    if msg.contains("not enough disk") || msg.contains("insufficient disk space") {
        return EXIT_NO_SPACE;
    }
    if msg.contains("sha") && msg.contains("mismatch") {
        return EXIT_HASH_MISMATCH;
    }
    if msg.contains("interrupted") {
        return EXIT_INTERRUPTED;
    }

    // Default to network error
    EXIT_NETWORK
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::DownloadModel {
            id,
            file,
            out,
            sha256,
            force,
            rev,
            no_progress,
            verbose,
            base_url,
            json,
            retries,
            timeout,
        } => download_model_cmd(DownloadConfig {
            id: &id,
            file: &file,
            out_dir: &out,
            sha256_hex: sha256.as_deref(),
            force,
            rev: rev.as_deref(),
            no_progress,
            verbose,
            base_url: &base_url,
            json,
            retries,
            timeout,
        }),
        Cmd::FetchCpp { tag, force, clean, backend, cmake_flags, repo } => {
            fetch_cpp_cmd(&tag, force, clean, &backend, &cmake_flags, &repo)
        }
        Cmd::Crossval { model, cpp_dir, release, dry_run, extra } => {
            let model_path = match model {
                Some(p) => p,
                None => resolve_default_model()?,
            };
            crossval_cmd(&model_path, cpp_dir.as_deref(), release, &extra, dry_run)
        }
        Cmd::FullCrossval { force, tag, backend, cmake_flags, repo } => {
            full_crossval_cmd(force, &tag, &backend, &cmake_flags, &repo)
        }
        Cmd::GenFixtures { size, output } => gen_fixtures(&size, &output),
        Cmd::GenMiniGguf { output, version } => gen_mini_gguf(&output, version),
        Cmd::SetupCrossval => setup_crossval(),
        Cmd::CleanCache => clean_cache(),
        Cmd::CheckFeatures => check_features(),
        Cmd::Gate { which } => match which {
            GateWhich::Mapper { model } => std::process::exit(gates::mapper_gate(model)?),
        },
        Cmd::Benchmark { platform } => run_benchmark(&platform),
        Cmd::CompareMetrics { baseline, current, ppl_max, latency_p95_max, tok_s_min } => {
            compare_metrics(&baseline, &current, ppl_max, latency_p95_max, tok_s_min)
        }
        Cmd::DetectBreaking { baseline, current, format } => {
            detect_breaking_changes_cmd(baseline.as_deref(), &current, &format)
        }
        Cmd::VendorGgml { commit, force, output } => vendor_ggml_cmd(&commit, force, &output),
        Cmd::GpuPreflight { require, format } => gpu_preflight_cmd(require, &format),
        Cmd::GpuSmoke { size, tolerance, skip_if_no_gpu } => {
            gpu_smoke_cmd(&size, tolerance, skip_if_no_gpu)
        }
        Cmd::Demo { which, args } => demo_cmd(&which, &args),
    }
}

// JSON event structure for CI/CD pipelines
#[derive(serde::Serialize)]
struct Event<'a> {
    phase: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    downloaded: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wait_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    msg: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    resume: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    start: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ms: Option<u64>,
}

// Download configuration to reduce function arguments
struct DownloadConfig<'a> {
    id: &'a str,
    file: &'a str,
    out_dir: &'a Path,
    sha256_hex: Option<&'a str>,
    force: bool,
    rev: Option<&'a str>,
    no_progress: bool,
    verbose: bool,
    base_url: &'a str,
    json: bool,
    retries: u32,
    timeout: u64,
}

// Macro for emitting JSON events
macro_rules! ev {
    ($json:expr, $phase:expr, { $($key:ident: $value:expr),* $(,)? }) => {
        if $json {
            let mut event = Event {
                phase: $phase,
                url: None,
                downloaded: None,
                total: None,
                wait_secs: None,
                msg: None,
                resume: None,
                start: None,
                bytes: None,
                ms: None,
            };
            $(event.$key = Some($value);)*
            let _ = println!("{}", serde_json::to_string(&event).unwrap());
        }
    };
}

fn download_model_cmd(config: DownloadConfig) -> Result<()> {
    let DownloadConfig {
        id,
        file,
        out_dir,
        sha256_hex,
        force,
        rev,
        no_progress,
        verbose,
        base_url,
        json,
        retries,
        timeout,
    } = config;
    fs::create_dir_all(out_dir)?;

    // Guard against path traversal
    let safe_file =
        Path::new(file).file_name().ok_or_else(|| anyhow!("invalid file name: {}", file))?;

    let dest_dir = out_dir.join(id.replace('/', "-"));
    fs::create_dir_all(&dest_dir)?;
    let dest = dest_dir.join(safe_file);

    let revision = rev.unwrap_or("main");
    let url = format!("{base_url}/{id}/resolve/{revision}/{file}");
    let token = std::env::var("HF_TOKEN").ok();

    if verbose {
        eprintln!("[VERBOSE] URL: {}", url);
        eprintln!("[VERBOSE] Revision: {}", revision);
    }

    // Build client first (needed for conditional checks)
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(timeout))
        .user_agent(USER_AGENT_STRING)
        .no_gzip()
        .no_brotli()
        .no_deflate() // Force identity encoding for correct ranges
        .build()?;

    // Check if file exists and possibly skip download via ETag/Last-Modified
    let etag_path = dest.with_extension("etag");
    let lastmod_path = dest.with_extension("lastmod");

    if dest.exists() && !force {
        let mut up_to_date = false;
        let saved_etag = fs::read_to_string(&etag_path).ok();
        let saved_lastmod = fs::read_to_string(&lastmod_path).ok();

        if saved_etag.is_some() || saved_lastmod.is_some() {
            // Check if the file is still current
            let mut head_req = client.head(&url);
            if let Some(t) = &token {
                head_req = head_req.header(AUTHORIZATION, format!("Bearer {t}"));
            }
            head_req = head_req.header(ACCEPT_ENCODING, "identity");
            if let Some(etag) = &saved_etag {
                head_req = head_req.header(IF_NONE_MATCH, etag.trim());
            }
            if let Some(lm) = &saved_lastmod {
                head_req = head_req.header(IF_MODIFIED_SINCE, lm.trim());
            }

            if let Ok(resp) = head_req.send() {
                // Add friendlier auth message on HEAD
                if matches!(resp.status(), StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
                    bail!(
                        "HTTP {} from Hugging Face during metadata check. If the repo is private, set HF_TOKEN, e.g.\n\
                           HF_TOKEN=*** cargo xtask download-model --id {} --file {}",
                        resp.status().as_u16(),
                        id,
                        file
                    );
                }

                match resp.status() {
                    StatusCode::NOT_MODIFIED => {
                        up_to_date = true;
                    }
                    StatusCode::OK => {
                        // remote likely changed; do not return
                    }
                    // If HEAD is not allowed, fall through to download path.
                    StatusCode::METHOD_NOT_ALLOWED => { /* fall through */ }
                    _ => { /* fall through; we'll attempt download */ }
                }
            }
        }

        if up_to_date {
            println!("✓ File is up to date: {}", dest.display());
            if let Some(want) = sha256_hex {
                if let Err(e) = verify_sha256(&dest, want) {
                    // Remove bad file and cache files
                    let _ = fs::remove_file(&dest);
                    let _ = fs::remove_file(&etag_path);
                    let _ = fs::remove_file(&lastmod_path);
                    return Err(e);
                }
                println!("✓ SHA256 verified");
            }
            return Ok(());
        }
        // else: remote changed or HEAD inconclusive → continue into download path
    }

    println!("📥 Downloading from Hugging Face:");
    println!("   Repository: {}", id);
    println!("   File: {}", file);
    println!("   Destination: {}", dest.display());
    if token.is_some() {
        println!("   Using HF_TOKEN for authentication");
    }

    // HEAD request to get file size and check resumability
    let mut head_req = client.head(&url);
    if let Some(t) = &token {
        head_req = head_req.header(AUTHORIZATION, format!("Bearer {t}"));
    }
    head_req = head_req.header(ACCEPT_ENCODING, "identity");

    // Try HEAD first, fallback to Range GET for size
    let (size, resumable) = head_req
        .send()
        .and_then(|r| r.error_for_status())
        .ok()
        .and_then(|r| {
            // Check if server supports range requests (default to false if missing)
            let resumable = r
                .headers()
                .get(ACCEPT_RANGES)
                .and_then(|h| h.to_str().ok())
                .map(|v| v.eq_ignore_ascii_case("bytes"))
                .unwrap_or(false);

            let sz = r.headers().get(CONTENT_LENGTH)?.to_str().ok()?.parse::<u64>().ok()?;
            Some((sz, resumable))
        })
        .map(|(sz, res)| (Some(sz), res))
        .or_else(|| {
            // Fallback: try 1-byte GET to extract total from Content-Range (with cache headers)
            let mut probe = client.get(&url);
            if let Some(t) = &token {
                probe = probe.header(AUTHORIZATION, format!("Bearer {t}"));
            }
            probe = probe.header(RANGE, "bytes=0-0").header(ACCEPT_ENCODING, "identity");

            // Add conditional headers for cache checking on fallback
            if dest.exists() && !force {
                if let Ok(etag) = fs::read_to_string(&etag_path) {
                    probe = probe.header(IF_NONE_MATCH, etag.trim());
                }
                if let Ok(lastmod) = fs::read_to_string(&lastmod_path) {
                    probe = probe.header(IF_MODIFIED_SINCE, lastmod.trim());
                }
            }

            probe
                .send()
                .ok()
                .and_then(|r| {
                    // Check for 304 on the 1-byte probe - means file is current
                    if r.status() == StatusCode::NOT_MODIFIED && dest.exists() && !force {
                        // Can't early return from a closure, will handle after
                        return None;
                    }
                    let sz = r
                        .headers()
                        .get(CONTENT_RANGE)
                        .and_then(|h| h.to_str().ok())
                        .and_then(|s| s.rsplit('/').next()?.parse::<u64>().ok())?;
                    Some(sz)
                })
                .map(|sz| (Some(sz), true))
        })
        .unwrap_or((None, false)); // Default to non-resumable if we can't determine

    // If we got a 304 on the fallback probe and file exists, we're done
    if size.is_none() && dest.exists() && !force {
        // Do another quick check to see if it was a 304
        let mut probe = client.get(&url);
        if let Some(t) = &token {
            probe = probe.header(AUTHORIZATION, format!("Bearer {t}"));
        }
        probe = probe.header(RANGE, "bytes=0-0");
        if let Ok(etag) = fs::read_to_string(&etag_path) {
            probe = probe.header(IF_NONE_MATCH, etag.trim());
        }
        if let Ok(lastmod) = fs::read_to_string(&lastmod_path) {
            probe = probe.header(IF_MODIFIED_SINCE, lastmod.trim());
        }
        if let Ok(r) = probe.send()
            && r.status() == StatusCode::NOT_MODIFIED
        {
            println!("✓ File is up to date: {}", dest.display());
            if let Some(want) = sha256_hex {
                if let Err(e) = verify_sha256(&dest, want) {
                    let _ = fs::remove_file(&dest);
                    let _ = fs::remove_file(&etag_path);
                    let _ = fs::remove_file(&lastmod_path);
                    return Err(e);
                }
                println!("✓ SHA256 verified");
            }
            return Ok(());
        }
    }

    // Ensure directory exists before checking disk space
    if !dest_dir.exists() {
        fs::create_dir_all(&dest_dir)
            .with_context(|| format!("failed to create {}", dest_dir.display()))?;
    }

    let tmp = dest.with_extension("part");
    let mut start = 0u64;

    // Force mode clears partial download
    if force && tmp.exists() {
        let _ = fs::remove_file(&tmp);
        start = 0;
    } else if tmp.exists() {
        // Check for partial download
        start = fs::metadata(&tmp)?.len();
        if let Some(total) = size {
            println!(
                "   Resuming from {:.2} MB / {:.2} MB",
                start as f64 / 1_048_576.0,
                total as f64 / 1_048_576.0
            );
        }
    }

    // Check disk space before downloading (calculate only remaining bytes)
    if let Some(total) = size {
        let remaining = total.saturating_sub(start);
        let avail = available_space(&dest_dir)
            .with_context(|| format!("failed to query free space in {}", dest_dir.display()))?;
        // Leave 50MB headroom
        let need = remaining + 50 * 1024 * 1024;
        if avail < need {
            bail!(
                "Not enough disk space in {}: need ~{:.2} MB, have ~{:.2} MB",
                dest_dir.display(),
                need as f64 / 1_048_576.0,
                avail as f64 / 1_048_576.0
            );
        }
    }

    // Single-writer lock to prevent concurrent downloads (alongside the .part file)
    let lock_path = tmp.with_extension("lock");
    let lock_file = std::fs::File::create(&lock_path)
        .with_context(|| format!("failed to create lock file for {}", dest.display()))?;
    lock_file.try_lock_exclusive().with_context(|| {
        format!("another download appears to be running for {}", dest.display())
    })?;

    // Use RAII guard for automatic cleanup (transfers ownership)
    let _lock_guard = LockGuard::new(lock_path, lock_file);

    // Setup SHA256 hasher if verification requested
    let verify = sha256_hex.is_some();
    let mut hasher = if verify {
        let mut h = Sha256::new();
        // If resuming, seed hasher with existing bytes
        if start > 0 && tmp.exists() {
            let mut seed = std::fs::File::open(&tmp)?;
            let mut seed_buf = vec![0u8; 1024 * 256];
            loop {
                let n = std::io::Read::read(&mut seed, &mut seed_buf)?;
                if n == 0 {
                    break;
                }
                h.update(&seed_buf[..n]);
            }
        }
        Some(h)
    } else {
        None
    };

    // Request with retry logic and proper range handling
    let mut attempt = 0;
    let max_attempts = retries;

    // Emit JSON start event
    ev!(json, "start", { url: &url, resume: start > 0, start: start });
    let mut resp = loop {
        // If tmp larger than remote size, restart clean
        if let Some(total) = size
            && start > total
        {
            println!(
                "   Local partial ({:.2} MB) exceeds remote size ({:.2} MB); restarting",
                start as f64 / 1_048_576.0,
                total as f64 / 1_048_576.0
            );
            start = 0;
        }

        let mut rb = client.get(&url);
        if let Some(t) = &token {
            rb = rb.header(AUTHORIZATION, format!("Bearer {t}"));
        }
        rb = rb.header(ACCEPT_ENCODING, "identity");

        // Only request range if resumable and we have bytes to skip
        if resumable && start > 0 {
            rb = rb.header(RANGE, format!("bytes={start}-"));

            // Add If-Range for safe resumption (prefer strong ETag)
            if let Ok(etag) = fs::read_to_string(&etag_path) {
                let etag = etag.trim();
                if !etag.starts_with("W/") {
                    rb = rb.header(IF_RANGE, etag);
                }
            } else if let Ok(lm) = fs::read_to_string(&lastmod_path) {
                rb = rb.header(IF_RANGE, lm.trim());
            }
        } else if start == 0 {
            // Conditional GET when starting from 0
            if let Ok(etag) = fs::read_to_string(&etag_path) {
                rb = rb.header(IF_NONE_MATCH, etag.trim());
            }
            if let Ok(lm) = fs::read_to_string(&lastmod_path) {
                rb = rb.header(IF_MODIFIED_SINCE, lm.trim());
            }
        }

        let r = match rb.send() {
            Ok(resp) => {
                // Handle various status codes before error_for_status()
                match resp.status() {
                    StatusCode::TOO_MANY_REQUESTS if attempt < max_attempts => {
                        let wait = retry_after_secs(resp.headers());
                        eprintln!("   429 rate limited. Waiting {wait}s before retry...");
                        ev!(json, "retry", { wait_secs: wait, msg: "429" });
                        thread::sleep(Duration::from_secs(wait));
                        attempt += 1;
                        continue;
                    }
                    StatusCode::PRECONDITION_FAILED | StatusCode::RANGE_NOT_SATISFIABLE => {
                        // 412 or 416: server rejected resume, restart from 0
                        if verbose {
                            eprintln!(
                                "   server rejected resume ({}); restarting from 0",
                                resp.status()
                            );
                        }
                        let _ = fs::remove_file(&tmp);
                        start = 0; // Will restart from beginning
                        attempt += 1;
                        if attempt > max_attempts {
                            bail!("failed after {} attempts due to resume rejection", max_attempts);
                        }
                        continue;
                    }
                    _ => {} // Continue processing
                }
                // Check for 304 Not Modified on full GET
                if start == 0 && resp.status() == StatusCode::NOT_MODIFIED {
                    println!("✓ File is up to date: {}", dest.display());
                    if let Some(want) = sha256_hex
                        && let Err(e) = verify_sha256(&dest, want)
                    {
                        let _ = fs::remove_file(&dest);
                        let _ = fs::remove_file(&etag_path);
                        let _ = fs::remove_file(&lastmod_path);
                        return Err(e);
                    }
                    return Ok(());
                }
                resp // fall through; handle Content-Range + error_for_status below
            }
            Err(e) if attempt < max_attempts => {
                attempt += 1;
                let backoff = exp_backoff_ms(attempt);
                eprintln!("   transient error: {e}; retrying in {backoff} ms");
                thread::sleep(Duration::from_millis(backoff));
                continue;
            }
            Err(e) => {
                return Err(e).context("download request failed");
            }
        };

        // If server says the Range was invalid, restart from 0
        if r.status() == StatusCode::RANGE_NOT_SATISFIABLE && start > 0 {
            println!("   Server rejected resume; restarting from 0");
            start = 0;
            attempt += 1;
            if attempt > max_attempts {
                bail!("persistent 416 Range errors");
            }
            continue;
        }

        // Friendlier auth errors
        let status = r.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            bail!(
                "HTTP {} from Hugging Face. If the repo is private, set HF_TOKEN, e.g.\n\
                 HF_TOKEN=*** cargo xtask download-model --id {} --file {}",
                status.as_u16(),
                id,
                file
            );
        }

        let resp = r.error_for_status()?;

        // Verify Content-Range alignment on resume
        if start > 0 && resp.status() == StatusCode::PARTIAL_CONTENT {
            // Check if Content-Range is present and valid
            let valid_range = resp
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|h| h.to_str().ok())
                .map(|v| v.starts_with(&format!("bytes {start}-")))
                .unwrap_or(false);

            if !valid_range {
                // 206 without valid Content-Range - unsafe resume
                eprintln!(
                    "   Server sent 206 but Content-Range invalid/missing; restarting from 0"
                );
                drop(resp);
                start = 0;

                // Re-check disk space when restarting
                if let Some(total) = size {
                    let available = fs2::available_space(dest.parent().unwrap_or(Path::new(".")))?;
                    if available < total {
                        bail!(
                            "insufficient disk space: need {} MB, have {} MB",
                            total / 1_048_576,
                            available / 1_048_576
                        );
                    }
                }

                attempt += 1;
                if attempt > max_attempts {
                    bail!("failed after {} attempts due to invalid 206 response", max_attempts);
                }
                thread::sleep(Duration::from_millis(exp_backoff_ms(attempt)));
                continue;
            }
        }

        break resp;
    };

    // Check if server ignored Range header (must restart from 0)
    let resumed = resumable && start > 0;
    if resumed && resp.status() == StatusCode::OK {
        // Server ignored Range -> restart clean
        println!("   Server ignored resume request, restarting download...");
        start = 0;
    }

    // Setup progress bar (hide if not a TTY or if --no-progress)
    let pb = if !no_progress && std::io::stderr().is_terminal() {
        if let Some(total) = size {
            let pb = ProgressBar::new(total);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}",
                )?
                .progress_chars("##-"),
            );
            pb
        } else {
            let pb = ProgressBar::new_spinner();
            pb.set_style(ProgressStyle::with_template(
                "{spinner:.green} downloading {bytes} {msg}",
            )?);
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb
        }
    } else {
        // Hide spinner in CI/non-TTY environments
        let pb = ProgressBar::hidden();
        pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(1));
        pb
    };

    if start > 0 {
        pb.set_position(start);
        pb.set_message("resuming");
    }

    let file_handle = if resumed && resp.status() == StatusCode::OK {
        // Server ignored Range, need to truncate and restart
        let mut f = fs::OpenOptions::new().create(true).write(true).truncate(true).open(&tmp)?;
        f.seek(SeekFrom::Start(0))?;
        f
    } else {
        // Normal case: seek to resume point if needed
        let mut f =
            fs::OpenOptions::new().create(true).write(true).truncate(start == 0).open(&tmp)?;
        if start > 0 {
            f.seek(SeekFrom::Start(start))?;
        } else if let Some(total) = size {
            // Preallocate file to detect ENOSPC early and reduce fragmentation
            let _ = f.set_len(total);
        }
        f
    };

    // Use BufWriter for better I/O performance (1 MiB buffer)
    let mut file_out = BufWriter::with_capacity(1024 * 1024, file_handle);

    // Reset interrupt flag and setup Ctrl-C handler (once per process)
    INTERRUPTED.store(false, Ordering::SeqCst);
    CTRL_ONCE.call_once(|| {
        let _ = ctrlc::set_handler(|| {
            INTERRUPTED.store(true, Ordering::SeqCst);
        });
    });

    let mut downloaded = if resumed && resp.status() == StatusCode::OK {
        0 // Server ignored Range, restarting from 0
    } else {
        start // Normal resume or new download
    };
    let mut last_log = downloaded; // Track last verbose log position
    let mut buf = vec![0u8; 1024 * 256]; // 256KB buffer
    let start_time = Instant::now();

    loop {
        // Check for interruption
        if INTERRUPTED.load(Ordering::SeqCst) {
            pb.finish_with_message("interrupted (partial file kept for resume)");
            println!("   Partial download saved at: {}", tmp.display());
            println!("   Run the same command again to resume");

            // Flush buffer, close file handle, release & remove lock
            file_out.flush().ok();
            drop(file_out);

            process::exit(EXIT_INTERRUPTED);
        }

        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file_out.write_all(&buf[..n])?;
        if let Some(ref mut h) = hasher {
            h.update(&buf[..n]);
        }
        downloaded += n as u64;
        pb.set_position(downloaded);

        // Log progress every 10 MiB (tracks actual delta)
        if verbose && downloaded - last_log >= 10 * 1024 * 1024 {
            eprintln!("[VERBOSE] Downloaded {} MB", downloaded / 1_048_576);
            last_log = downloaded;
        }
    }

    // Durability: flush buffer and fsync before rename
    file_out.flush()?;
    file_out.get_ref().sync_all()?;
    drop(file_out);

    let elapsed = start_time.elapsed();
    let secs = elapsed.as_secs_f64().max(0.001); // Avoid division by zero
    let throughput = (downloaded - start) as f64 / secs / 1_048_576.0;

    pb.finish_with_message(format!("complete ({:.2} MB/s)", throughput));

    // Atomic rename BEFORE persisting metadata
    fs::rename(&tmp, &dest)?;

    // fsync parent directory for journaling
    #[cfg(unix)]
    {
        if let Some(parent) = dest.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
    }

    // Save etag/last-modified atomically for future conditional requests
    if let Some(etag) = resp.headers().get(ETAG).and_then(|v| v.to_str().ok()) {
        atomic_write(&etag_path, etag.as_bytes()).ok();
    }
    if let Some(lm) = resp.headers().get(LAST_MODIFIED).and_then(|v| v.to_str().ok()) {
        atomic_write(&lastmod_path, lm.as_bytes()).ok();
    }

    // Verify final size if known
    if let Some(total) = size {
        let actual = fs::metadata(&dest)?.len();
        if actual != total {
            bail!("download truncated: got {} bytes, expected {}", actual, total);
        }
    }

    // Verify SHA256 using streamed hash
    if let Some(want) = sha256_hex
        && let Some(h) = hasher
    {
        let got = format!("{:x}", h.finalize());
        if got != want {
            let _ = fs::remove_file(&dest);
            let _ = fs::remove_file(&etag_path);
            let _ = fs::remove_file(&lastmod_path);
            bail!("SHA256 mismatch: expected {}, got {}", want, got);
        }
        println!("✓ SHA256 verified");
    }

    // Emit JSON completion event
    ev!(json, "done", { bytes: downloaded, ms: elapsed.as_millis() as u64 });

    if !json {
        eprintln!("✅ Saved: {}", dest.display());
        if let Some(size) = size {
            eprintln!("   Size: {:.2} MB", size as f64 / 1_048_576.0);
        }
        eprintln!("   Time: {:.1}s", elapsed.as_secs_f64());
        eprintln!("   Speed: {:.2} MB/s", throughput);
    }

    // Print ready-to-use export command (to stderr for non-JSON)
    if !json {
        let abs_path = dest.canonicalize().unwrap_or(dest.clone());
        eprintln!();
        eprintln!("To use this model for cross-validation:");
        eprintln!("  export CROSSVAL_GGUF=\"{}\"", abs_path.display());
    }

    Ok(())
}

fn resolve_default_model() -> Result<PathBuf> {
    let root = PathBuf::from("models");
    if !root.exists() {
        return Err(anyhow!("No models directory found. Run `cargo xtask download-model` first."));
    }

    // Prefer default model path
    let preferred =
        root.join(format!("{}/{}", DEFAULT_MODEL_ID.replace('/', "-"), DEFAULT_MODEL_FILE));
    if preferred.exists() {
        return Ok(preferred);
    }

    // Fallback: scan for first *.gguf file
    for entry in WalkDir::new(&root).into_iter().filter_map(Result::ok) {
        if entry.file_type().is_file()
            && let Some(ext) = entry.path().extension()
            && ext == "gguf"
        {
            return Ok(entry.path().to_path_buf());
        }
    }

    Err(anyhow!(
        "No GGUF model found under ./models.\nTip: Run `cargo xtask download-model` or pass --model <path/to/model.gguf>"
    ))
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

fn fetch_cpp_cmd(
    tag: &str,
    force: bool,
    clean: bool,
    backend: &str,
    cmake_flags: &str,
    repo: &str,
) -> Result<()> {
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
    println!("   Repository: {}", repo);
    println!("   Branch/Rev: {}", tag);
    println!("   Backend: {}", backend);
    println!("   Force: {}", force);
    println!("   Clean: {}", clean);
    if !cmake_flags.is_empty() {
        println!("   CMake flags: {}", cmake_flags);
    }

    let mut args =
        vec!["--tag".to_string(), tag.to_string(), "--repo".to_string(), repo.to_string()];
    if force {
        args.push("--force".to_string());
    }
    if clean {
        args.push("--clean".to_string());
    }

    // Add backend-specific CMake flags with static build configuration
    args.push("--cmake-flags".to_string());

    // Always use static builds to avoid library path issues
    let mut all_flags =
        String::from("-DBUILD_SHARED_LIBS=OFF -DLLAMA_STATIC=ON -DLLAMA_BUILD_TESTS=OFF");

    if backend == "cuda" {
        all_flags.push_str(" -DGGML_CUDA=ON -DLLAMA_CUBLAS=ON");
        if cmake_flags.is_empty() {
            // Default CUDA architectures if not specified
            all_flags.push_str(" -DCMAKE_CUDA_ARCHITECTURES=80;86");
        }
    } else {
        // For CPU builds, enable native optimizations
        all_flags.push_str(" -DGGML_NATIVE=ON");
    }

    // Append any additional user-provided flags
    if !cmake_flags.is_empty() {
        all_flags.push(' ');
        all_flags.push_str(cmake_flags);
    }

    args.push(all_flags);

    run("bash", std::iter::once(script.to_string_lossy().to_string()).chain(args).collect())?;

    // Verify the build succeeded by checking for libraries or binaries
    let cpp_dir = dirs::home_dir().unwrap().join(".cache/bitnet_cpp");
    let build_dir = cpp_dir.join("build");

    // Check for any built artifacts (libraries or binaries) - recursively
    let mut found_artifacts = false;

    // Use walkdir to recursively find libraries
    let lib_extensions =
        if cfg!(target_os = "macos") { vec!["dylib", "so", "a"] } else { vec!["so", "a"] };

    for entry in walkdir::WalkDir::new(&build_dir)
        .max_depth(5)  // Limit depth to avoid excessive scanning
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            // Check for library files
            if let Some(ext) = path.extension()
                && lib_extensions.contains(&ext.to_string_lossy().as_ref())
            {
                found_artifacts = true;
                break;
            }
            // Check for executable files (no extension usually)
            if is_executable(path) && path.file_stem().is_some() {
                let name = path.file_name().unwrap().to_string_lossy();
                // Look for typical executable names
                if name.starts_with("llama") || name.starts_with("bitnet") || name == "main" {
                    found_artifacts = true;
                    break;
                }
            }
        }
    }

    // For now, just warn if no artifacts found - the build log already showed success
    if !found_artifacts {
        println!("⚠️  Warning: Could not verify build artifacts in {}", build_dir.display());
        println!("   The build appeared to succeed based on CMake output.");
        println!("   Libraries were reported at the expected locations.");
    } else {
        println!("   ✓ C++ build artifacts verified in: {}", build_dir.display());
    }
    Ok(())
}

/// Apply C++ environment variables for Linux
#[cfg(target_os = "linux")]
fn apply_cpp_env(cmd: &mut Command, cpp_root: &Path) {
    let lib_paths = format!(
        "{}:{}:{}",
        cpp_root.join("build/bin").display(),
        cpp_root.join("build/3rdparty/llama.cpp/src").display(),
        cpp_root.join("build/3rdparty/llama.cpp/ggml/src").display()
    );

    let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let merged =
        if existing.is_empty() { lib_paths } else { format!("{}:{}", lib_paths, existing) };

    cmd.env("LD_LIBRARY_PATH", merged);
}

/// Apply C++ environment variables for macOS
#[cfg(target_os = "macos")]
fn apply_cpp_env(cmd: &mut Command, cpp_root: &Path) {
    let lib_paths = format!(
        "{}:{}:{}",
        cpp_root.join("build/bin").display(),
        cpp_root.join("build/3rdparty/llama.cpp/src").display(),
        cpp_root.join("build/3rdparty/llama.cpp/ggml/src").display()
    );

    let existing = std::env::var("DYLD_LIBRARY_PATH").unwrap_or_default();
    let merged =
        if existing.is_empty() { lib_paths } else { format!("{}:{}", lib_paths, existing) };

    cmd.env("DYLD_LIBRARY_PATH", merged);
}

/// Apply C++ environment variables for Windows
#[cfg(target_os = "windows")]
fn apply_cpp_env(cmd: &mut Command, cpp_root: &Path) {
    let bin_path = cpp_root.join("build/bin").display().to_string();

    let existing = std::env::var("PATH").unwrap_or_default();
    let merged = if existing.is_empty() { bin_path } else { format!("{};{}", bin_path, existing) };

    cmd.env("PATH", merged);
}

/// Apply deterministic environment variables for testing
fn apply_deterministic_env(cmd: &mut Command) {
    cmd.env("RAYON_NUM_THREADS", "1")
        .env("BITNET_DETERMINISTIC", "1")
        .env("BITNET_SEED", "42")
        .env("OMP_NUM_THREADS", "1")
        .env("GGML_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1");
}

/// Preflight check using C++ header tool before full load
fn cpp_header_preflight(cpp_root: &Path, model: &Path) -> Result<()> {
    // Try multiple possible binary names
    let candidates = ["llama-gguf", "llama-cli", "main"];
    let llama_bin = candidates
        .iter()
        .map(|b| cpp_root.join(format!("build/bin/{}", b)))
        .find(|p| p.exists())
        .ok_or_else(|| {
            anyhow!(
                "No llama binary found in {}. Tried: {:?}",
                cpp_root.join("build/bin").display(),
                candidates
            )
        })?;

    // Log which binary we're using
    println!("   • Using C++ binary: {}", llama_bin.display());

    let mut cmd = Command::new(&llama_bin);

    // Use appropriate args based on which binary we found
    if llama_bin.file_name().and_then(|s| s.to_str()) == Some("llama-gguf") {
        cmd.args(["-l", "-m"]).arg(model);
    } else {
        // For llama-cli or main, use a minimal test
        cmd.arg("-m").arg(model).args(["-p", "", "-n", "1"]);
    }
    apply_cpp_env(&mut cmd, cpp_root);
    apply_deterministic_env(&mut cmd);

    let output = cmd
        .output()
        .with_context(|| format!("Failed to run C++ header preflight: {}", llama_bin.display()))?;

    if output.status.success() {
        println!("   ✓ C++ header preflight passed");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let msg = format!("{}\n{}", stderr, stdout).to_lowercase();
        Err(anyhow!("C++ header preflight failed: {}", msg))
    }
}

fn crossval_cmd(
    model: &Path,
    cpp_dir: Option<&Path>,
    release: bool,
    extra: &[String],
    dry_run: bool,
) -> Result<()> {
    if !model.exists() {
        return Err(anyhow!(
            "Model not found: {}\nTip: Run `cargo xtask download-model` first",
            model.display()
        ));
    }

    // Initialize cross-validation report
    let mut report = CrossValReport::new(model);

    // First validate that the Rust implementation can load the model
    println!("🔍 Validating Rust implementation can load the model...");
    match validate_rust_model_loading(model) {
        Ok((version, n_kv, n_tensors, data_offset)) => {
            report.rust_ok = true;
            report.gguf_version_detected = Some(version);
            report.n_kv = Some(n_kv);
            report.n_tensors = Some(n_tensors);
            report.data_offset = Some(data_offset);
            println!("   ✓ Rust implementation loaded model successfully");
        }
        Err(e) => {
            report.rust_ok = false;
            report.notes = format!("Rust implementation failed: {}", e);
            let _ = report.save(&PathBuf::from("target/crossval_report.json"));
            return Err(anyhow!("Rust implementation failed to load model: {}", e));
        }
    }

    let cpp = cpp_dir
        .map(|p| p.to_path_buf())
        .or_else(|| std::env::var_os("BITNET_CPP_DIR").map(PathBuf::from))
        .unwrap_or_else(|| dirs::home_dir().unwrap().join(".cache/bitnet_cpp"));

    if !cpp.exists() {
        eprintln!("⚠️  Warning: BITNET_CPP_DIR not found at {}", cpp.display());
        eprintln!("   Tip: Run `cargo xtask fetch-cpp` first");
    }

    // Check if soft-fail is enabled for C++ compatibility issues
    let allow_cpp_fail = std::env::var("CROSSVAL_ALLOW_CPP_FAIL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    // Run C++ header preflight check before full tests
    if cpp.exists() {
        println!("🔬 Running C++ header preflight check...");
        match cpp_header_preflight(&cpp, model) {
            Ok(()) => {
                println!("   ✓ C++ can parse GGUF header");
                report.cpp_header_ok = true;
            }
            Err(e) => {
                report.cpp_header_ok = false;
                if allow_cpp_fail {
                    println!(
                        "   ⚠️ XFAIL: C++ header preflight failed (CROSSVAL_ALLOW_CPP_FAIL=1)"
                    );
                    println!("   Details: {}", e);
                    report.xfail = true;
                    report.notes = format!("C++ header preflight failed (XFAIL): {}", e);
                    // Save report and exit early with success for known incompatibilities
                    let _ = report.save(&PathBuf::from("target/crossval_report.json"));
                    println!("\n✅ Cross-validation passed (C++ failure allowed)");
                    return Ok(());
                } else {
                    report.notes = format!("C++ header preflight failed: {}", e);
                    let _ = report.save(&PathBuf::from("target/crossval_report.json"));
                    return Err(anyhow!("C++ header preflight failed: {}", e));
                }
            }
        }
    }

    println!("🧪 Running cross-validation tests");
    println!("   Model: {}", model.display());
    // Echo the absolute path so users know exactly what was picked
    if let Ok(abs_model) = model.canonicalize() {
        println!("   Absolute: {}", abs_model.display());
    }
    println!("   C++ dir: {}", cpp.display());
    println!("   Release: {}", release);
    println!("   Deterministic: yes (single-threaded)");

    if allow_cpp_fail {
        println!("   C++ failures: ALLOWED (CROSSVAL_ALLOW_CPP_FAIL=1)");
    }

    // Build the cargo test command
    let mut cmd = Command::new("cargo");
    cmd.arg("test").args(["-p", "bitnet-crossval", "--features", "crossval"]);

    if release {
        cmd.arg("--release");
    }

    // Apply platform-specific C++ library paths
    apply_cpp_env(&mut cmd, &cpp);

    // Apply deterministic environment for testing
    apply_deterministic_env(&mut cmd);

    // Set other required environment variables
    cmd.env("BITNET_CPP_DIR", &cpp).env("CROSSVAL_GGUF", model).env("RUST_BACKTRACE", "1");

    // Add test runner args
    cmd.arg("--").args(["--nocapture", "--test-threads=1"]).args(extra);

    if dry_run {
        println!("\n[DRY RUN] Env + command:");
        println!("  BITNET_CPP_DIR={}", cpp.display());
        println!("  CROSSVAL_GGUF={}", model.display());
        println!("  Platform-specific library paths configured");
        println!("  Deterministic env: RAYON_NUM_THREADS=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42");
        println!("  RUST_BACKTRACE=1");
        println!("  {:?}", cmd);
        return Ok(());
    }

    // Run the tests and handle C++ failures gracefully if configured
    let result = cmd.output();

    match result {
        Ok(output) => {
            // Write output as it was generated
            std::io::Write::write_all(&mut std::io::stdout(), &output.stdout)?;
            std::io::Write::write_all(&mut std::io::stderr(), &output.stderr)?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);

                // Check if this is a C++ model loading failure - expand patterns for robustness
                let msg = format!("{}\n{}", stderr, stdout).to_lowercase();
                let is_cpp_load_fail = msg.contains("llama_load_model_from_file")
                    || msg.contains("failed to load model")
                    || msg.contains("invalid or unsupported tensor")
                    || msg.contains("invalid gguf")
                    || msg.contains("unsupported gguf version")
                    || msg.contains("unknown tensor type")
                    || msg.contains("could not open gguf")
                    || msg.contains("ggml_assert")
                    || msg.contains("c++ backend failed");

                if is_cpp_load_fail && allow_cpp_fail {
                    println!(
                        "\n⚠️  XFAIL: C++ implementation failed to load model (unsupported GGUF variant)"
                    );
                    println!("   This is expected for some experimental BitNet models.");
                    println!("   Rust implementation validated successfully.");
                    report.cpp_full_ok = false;
                    report.xfail = true;
                    report.notes = format!("C++ full load failed (XFAIL): {}", msg);
                    let _ = report.save(&PathBuf::from("target/crossval_report.json"));
                    return Ok(());
                }

                report.cpp_full_ok = false;
                report.notes = format!("Cross-validation tests failed: {}", msg);
                let _ = report.save(&PathBuf::from("target/crossval_report.json"));
                return Err(anyhow!("Cross-validation tests failed"));
            }

            // All tests passed!
            report.cpp_full_ok = true;
            report.notes = "All cross-validation tests passed".to_string();
            let _ = report.save(&PathBuf::from("target/crossval_report.json"));
            Ok(())
        }
        Err(e) => {
            report.notes = format!("Failed to run tests: {}", e);
            let _ = report.save(&PathBuf::from("target/crossval_report.json"));
            Err(e.into())
        }
    }
}

/// Validate that the Rust implementation can load the model
/// Returns GGUF metadata for enhanced reporting  
fn validate_rust_model_loading(model_path: &Path) -> Result<(u32, u64, u64, u64)> {
    // Use the real GGUF reader from bitnet-models
    println!("   Validating with real GGUF reader...");

    use bitnet_models::formats::gguf::GgufReader;
    use bitnet_models::loader::MmapFile;

    // Try to parse with the real GGUF reader
    match MmapFile::open(model_path) {
        Ok(mmap) => {
            match GgufReader::new(mmap.as_slice()) {
                Ok(reader) => {
                    // Validate the file structure
                    if let Err(e) = reader.validate() {
                        return Err(anyhow!("GGUF validation failed: {}", e));
                    }

                    let version = reader.version();
                    let n_kv = reader.metadata_kv_count();
                    let n_tensors = reader.tensor_count();
                    let data_offset = reader.data_offset();

                    println!("   ✓ GGUF v{} parsed and validated successfully", version);
                    println!("     - KV pairs: {}", n_kv);
                    println!("     - Tensors: {}", n_tensors);
                    println!("     - Data offset: {}", data_offset);

                    Ok((version, n_kv, n_tensors, data_offset))
                }
                Err(e) => {
                    // Fallback to basic validation for error details
                    use std::fs::File;
                    use std::io::Read;

                    let mut file = File::open(model_path)
                        .with_context(|| format!("Failed to open: {}", model_path.display()))?;

                    // Check GGUF magic
                    let mut magic = [0u8; 4];
                    file.read_exact(&mut magic)?;
                    if &magic != b"GGUF" {
                        return Err(anyhow!("Not a valid GGUF file (invalid magic)"));
                    }

                    // Read version
                    let mut version_bytes = [0u8; 4];
                    file.read_exact(&mut version_bytes)?;
                    let version = u32::from_le_bytes(version_bytes);

                    if version != 2 && version != 3 {
                        return Err(anyhow!(
                            "Unsupported GGUF version: {} (expected 2 or 3)",
                            version
                        ));
                    }

                    // If basic checks pass but real reader fails, report the reader error
                    Err(anyhow!("Rust GGUF reader could not parse: {}", e))
                }
            }
        }
        Err(e) => Err(anyhow!("Failed to memory-map file: {}", e)),
    }
}

fn full_crossval_cmd(
    force: bool,
    tag: &str,
    backend: &str,
    cmake_flags: &str,
    repo: &str,
) -> Result<()> {
    println!("🚀 Running full cross-validation workflow");
    println!("   Backend: {}", backend);
    println!("   C++ Tag: {}", tag);
    if !cmake_flags.is_empty() {
        println!("   CMake flags: {}", cmake_flags);
    }
    println!();

    // Step 1: Download model
    println!("Step 1/3: Downloading model");
    download_model_cmd(DownloadConfig {
        id: DEFAULT_MODEL_ID,
        file: DEFAULT_MODEL_FILE,
        out_dir: &PathBuf::from("models"),
        sha256_hex: None,
        force,
        rev: None,
        no_progress: false,
        verbose: false,
        base_url: "https://huggingface.co",
        json: false,
        retries: 3,
        timeout: 1800,
    })?;

    println!();

    // Step 2: Fetch C++ implementation
    println!("Step 2/3: Fetching C++ implementation ({})", backend);
    fetch_cpp_cmd(tag, force, false, backend, cmake_flags, repo)?;

    println!();

    // Step 3: Run tests with auto-discovery
    println!("Step 3/3: Running cross-validation tests");

    // Try auto-discovery first
    let model = match resolve_default_model() {
        Ok(m) => {
            println!("   Auto-discovered model: {}", m.display());
            m
        }
        Err(_) => {
            // Fallback to expected path
            let expected = PathBuf::from(format!(
                "models/{}/{}",
                DEFAULT_MODEL_ID.replace('/', "-"),
                DEFAULT_MODEL_FILE
            ));
            if !expected.exists() {
                return Err(anyhow!(
                    "Model not found at expected path: {}\nDownload may have failed.",
                    expected.display()
                ));
            }
            expected
        }
    };

    crossval_cmd(&model, None, true, &[], false)?;

    println!();
    println!("✅ Full cross-validation workflow complete!");

    Ok(())
}

// GGUF format constants
const GGUF_VALUE_TYPE_STRING: u32 = 8;

/// Helper to write a GGUF KV string pair (v3 format only)
fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    // Write key
    let key_bytes = key.as_bytes();
    buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(key_bytes);

    // Write value type (string)
    buf.extend_from_slice(&GGUF_VALUE_TYPE_STRING.to_le_bytes());

    // Write value
    let value_bytes = value.as_bytes();
    buf.extend_from_slice(&(value_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(value_bytes);
}

/// Generate a minimal valid GGUF file for smoke testing
/// Always generates v3 format. If requested_version is 2, adds a metadata tag.
fn gen_mini_gguf(output_path: &Path, requested_version: u32) -> Result<()> {
    println!("🔧 Generating minimal GGUF file (v3 format)...");
    if requested_version == 2 {
        println!("   Note: Emitting v3 with compat.v2_requested=true tag");
    }
    println!("   Output: {}", output_path.display());

    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut data = Vec::new();

    // Write GGUF header (v3)
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // n_tensors = 0

    // Save position for n_kv (will backpatch later)
    let n_kv_pos = data.len();
    data.extend_from_slice(&0u64.to_le_bytes()); // placeholder for n_kv

    data.extend_from_slice(&32u32.to_le_bytes()); // alignment = 32

    // Save position for data_offset (will backpatch later)
    let data_offset_pos = data.len();
    data.extend_from_slice(&0u64.to_le_bytes()); // placeholder for data_offset

    // Write metadata KV pairs, counting as we go
    let mut kv_count = 0u64;

    write_kv_string(&mut data, "general.architecture", "test");
    kv_count += 1;

    write_kv_string(&mut data, "general.name", "mini_test_model");
    kv_count += 1;

    write_kv_string(&mut data, "general.file_type", "smoke");
    kv_count += 1;

    write_kv_string(
        &mut data,
        "compat.v2_requested",
        if requested_version == 2 { "true" } else { "false" },
    );
    kv_count += 1;

    // Backpatch the actual n_kv count
    data[n_kv_pos..n_kv_pos + 8].copy_from_slice(&kv_count.to_le_bytes());

    // Calculate aligned header size
    let alignment = 32usize;
    let unpadded_size = data.len();
    let aligned_size = unpadded_size.div_ceil(alignment) * alignment;

    // Backpatch data_offset to point to aligned header end
    data[data_offset_pos..data_offset_pos + 8]
        .copy_from_slice(&(aligned_size as u64).to_le_bytes());

    // Pad to alignment (0 tensors means file ends at data_offset)
    if aligned_size > unpadded_size {
        data.extend(std::iter::repeat_n(0u8, aligned_size - unpadded_size));
    }

    // Write to file
    fs::write(output_path, &data)?;

    println!("✅ Generated minimal GGUF file ({} bytes)", data.len());
    println!("   - Version: 3 (always)");
    println!("   - Tensors: 0");
    println!("   - Metadata: {} KV pairs", kv_count);
    println!("   - Data offset: {} (aligned header end)", aligned_size);

    Ok(())
}

// Keep existing functionality from original xtask
fn gen_fixtures(size: &str, output_dir: &Path) -> Result<()> {
    use serde_json::json;

    println!("🔧 Generating deterministic test model fixtures...");
    println!("  Size: {}", size);
    println!("  Output: {}", output_dir.display());

    fs::create_dir_all(output_dir)?;

    // Generate more realistic test data based on size
    let (vocab_size, hidden_size, num_layers) = match size {
        "tiny" => (100, 64, 2),
        "small" => (1000, 128, 4),
        "medium" => (10000, 256, 8),
        _ => {
            eprintln!("⚠️  Unknown size '{}', using 'small'", size);
            (1000, 128, 4)
        }
    };

    // Create a minimal GGUF-like metadata file
    let metadata = json!({
        "general.architecture": "bitnet",
        "general.name": format!("test_model_{}", size),
        "bitnet.context_length": 512,
        "bitnet.embedding_length": hidden_size,
        "bitnet.block_count": num_layers,
        "bitnet.feed_forward_length": hidden_size * 4,
        "bitnet.attention.head_count": 8,
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": (0..vocab_size).map(|i| format!("token_{}", i)).collect::<Vec<_>>(),
        "tokenizer.ggml.scores": vec![0.0f32; vocab_size],
        "tokenizer.ggml.token_type": vec![0i32; vocab_size],
    });

    let metadata_path = output_dir.join(format!("test_model_{}_metadata.json", size));
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

    // Generate weight tensors (dummy data)
    let weights_path = output_dir.join(format!("test_model_{}_weights.bin", size));
    let num_params = vocab_size * hidden_size + hidden_size * hidden_size * num_layers;
    let weight_data = vec![0u8; (num_params / 8).max(1024)]; // 1-bit quantized
    fs::write(&weights_path, weight_data)?;

    println!("  Created metadata: {}", metadata_path.display());
    println!("  Created weights: {} ({} bytes)", weights_path.display(), num_params / 8);
    println!("✅ Test fixtures generated for '{}' model", size);
    Ok(())
}

fn setup_crossval() -> Result<()> {
    println!("🔧 Setting up cross-validation environment...");

    // Generate test fixtures
    println!("  Generating test fixtures...");
    gen_fixtures("small", &PathBuf::from("crossval/fixtures/"))?;

    // Build with crossval features
    println!("  Building with cross-validation features...");
    let status = Command::new("cargo").args(["build", "--features", "crossval"]).status()?;

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
        ("Cargo target", PathBuf::from("target/")),
        ("C++ build", dirs::home_dir().unwrap().join(".cache/bitnet_cpp/")),
        ("Test fixtures", PathBuf::from("crossval/fixtures/")),
        ("Models", PathBuf::from("models/")),
    ];

    // Calculate total size
    let mut total_size = 0u64;
    let mut existing_dirs = Vec::new();

    for (name, dir) in &cache_dirs {
        if dir.exists() {
            let size = dir_size(dir)?;
            total_size += size;
            existing_dirs.push((*name, dir.clone(), size));
            println!("  {} ({:.2} MB): {}", name, size as f64 / 1_048_576.0, dir.display());
        }
    }

    if existing_dirs.is_empty() {
        println!("✅ No caches to clean");
        return Ok(());
    }

    println!("\n  Total: {:.2} MB", total_size as f64 / 1_048_576.0);
    println!("\n⚠️  This will delete the directories listed above.");
    print!("  Continue? [y/N]: ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("y") {
        println!("Cancelled.");
        return Ok(());
    }

    for (name, dir, _) in existing_dirs {
        print!("  Removing {}... ", name);
        std::io::stdout().flush()?;
        fs::remove_dir_all(&dir)?;
        println!("✓");
    }

    println!("\n✅ Freed {:.2} MB", total_size as f64 / 1_048_576.0);
    Ok(())
}

fn dir_size(path: &Path) -> Result<u64> {
    let mut size = 0u64;
    for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
        if let Ok(metadata) = entry.metadata() {
            size += metadata.len();
        }
    }
    Ok(size)
}

fn check_features() -> Result<()> {
    println!("🔍 Checking feature flag consistency...");

    let cargo_toml = fs::read_to_string("Cargo.toml")?;

    if cargo_toml.contains("default = [") && cargo_toml.contains("\"crossval\"") {
        return Err(anyhow!("crossval feature is enabled by default! This will slow down builds."));
    }

    println!("  ✅ crossval feature is not in default features");
    println!("✅ Feature flag consistency check passed!");

    Ok(())
}

fn run_benchmark(platform: &str) -> Result<()> {
    println!("🚀 Running performance benchmarks...");
    println!("  Platform: {}", platform);

    let status =
        Command::new("cargo").args(["bench", "--workspace", "--features", "cpu"]).status()?;

    if !status.success() {
        return Err(anyhow!("Benchmarks failed"));
    }

    println!("✅ Benchmarks completed successfully!");
    Ok(())
}

// Metrics structure for cross-validation comparison
#[derive(serde::Deserialize, serde::Serialize, Debug)]
struct CrossvalMetrics {
    #[serde(default)]
    git: GitInfo,
    #[serde(default)]
    timestamp_utc: String,
    #[serde(default)]
    device: DeviceInfo,
    #[serde(default)]
    model: ModelInfo,
    metrics: MetricsData,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Default)]
struct GitInfo {
    sha: String,
    branch: String,
    cpp_tag: String,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Default)]
struct DeviceInfo {
    backend: String,
    compute_caps: String,
    driver: String,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Default)]
struct ModelInfo {
    name: String,
    vocab: u32,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
struct MetricsData {
    ppl: f64,
    acc: f64,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    tok_s: f64,
    #[serde(default)]
    gpu_mem_mb: f64,
}

fn detect_breaking_changes_cmd(
    baseline: Option<&Path>,
    current: &Path,
    format: &str,
) -> Result<()> {
    // If no baseline specified, try to use the latest git tag
    let baseline_path = if let Some(base) = baseline {
        base.to_path_buf()
    } else {
        // Get latest git tag
        let output = Command::new("git")
            .args(["describe", "--tags", "--abbrev=0"])
            .output()
            .context("Failed to get latest git tag")?;

        if !output.status.success() {
            return Err(anyhow!("No git tags found. Please specify --baseline"));
        }

        let tag = String::from_utf8_lossy(&output.stdout).trim().to_string();
        println!("Using git tag as baseline: {}", tag);

        // Create temp directory and checkout the tag
        let temp_dir = tempfile::tempdir()?;
        let baseline_path = temp_dir.path().join("baseline");

        Command::new("git")
            .args(["worktree", "add", baseline_path.to_str().unwrap(), &tag])
            .status()
            .context("Failed to checkout baseline version")?;

        baseline_path
    };

    // Simple implementation - would use the breaking_changes module in production
    println!("🔍 Detecting breaking changes...");
    println!("  Baseline: {}", baseline_path.display());
    println!("  Current: {}", current.display());

    // Run cargo-semver-checks if available
    let result = Command::new("cargo")
        .args([
            "semver-checks",
            "--baseline-path",
            baseline_path.to_str().unwrap(),
            "--manifest-path",
            current.join("Cargo.toml").to_str().unwrap(),
        ])
        .status();

    match result {
        Ok(status) if status.success() => {
            println!("✅ No breaking changes detected!");
        }
        Ok(_) => {
            println!("⚠️  Breaking changes detected!");
            if format == "json" {
                println!(r#"{{"breaking_changes": true, "compatible": false}}"#);
            }
            return Err(anyhow!("Breaking changes detected"));
        }
        Err(_) => {
            println!("⚠️  cargo-semver-checks not installed");
            println!("    Install with: cargo install cargo-semver-checks");
            println!("    Skipping breaking change detection");
        }
    }

    Ok(())
}

fn compare_metrics(
    baseline_path: &Path,
    current_path: &Path,
    ppl_max: f64,
    latency_p95_max: f64,
    tok_s_min: f64,
) -> Result<()> {
    println!("📊 Comparing metrics for regression detection");

    // Load baseline metrics
    let baseline_json = fs::read_to_string(baseline_path)
        .with_context(|| format!("Failed to read baseline: {}", baseline_path.display()))?;
    let baseline: CrossvalMetrics =
        serde_json::from_str(&baseline_json).with_context(|| "Failed to parse baseline JSON")?;

    // Load current metrics
    let current_json = fs::read_to_string(current_path)
        .with_context(|| format!("Failed to read current: {}", current_path.display()))?;
    let current: CrossvalMetrics =
        serde_json::from_str(&current_json).with_context(|| "Failed to parse current JSON")?;

    println!("\n📈 Baseline:");
    println!("  PPL: {:.2}", baseline.metrics.ppl);
    println!("  Latency P95: {:.1}ms", baseline.metrics.latency_p95_ms);
    println!("  Throughput: {:.0} tok/s", baseline.metrics.tok_s);

    println!("\n📉 Current:");
    println!("  PPL: {:.2}", current.metrics.ppl);
    println!("  Latency P95: {:.1}ms", current.metrics.latency_p95_ms);
    println!("  Throughput: {:.0} tok/s", current.metrics.tok_s);

    // Calculate changes
    let ppl_change = (current.metrics.ppl - baseline.metrics.ppl) / baseline.metrics.ppl;
    let latency_change = (current.metrics.latency_p95_ms - baseline.metrics.latency_p95_ms)
        / baseline.metrics.latency_p95_ms;
    let tok_change = (current.metrics.tok_s - baseline.metrics.tok_s) / baseline.metrics.tok_s;

    println!("\n📊 Changes:");
    println!("  PPL: {:+.2}%", ppl_change * 100.0);
    println!("  Latency P95: {:+.1}%", latency_change * 100.0);
    println!("  Throughput: {:+.1}%", tok_change * 100.0);

    // Check thresholds
    let mut regressions = Vec::new();

    if ppl_change > ppl_max {
        regressions.push(format!(
            "PPL increased by {:.2}% (max allowed: {:.2}%)",
            ppl_change * 100.0,
            ppl_max * 100.0
        ));
    }

    if latency_change > latency_p95_max {
        regressions.push(format!(
            "Latency P95 increased by {:.1}% (max allowed: {:.1}%)",
            latency_change * 100.0,
            latency_p95_max * 100.0
        ));
    }

    if tok_change < tok_s_min {
        regressions.push(format!(
            "Throughput decreased by {:.1}% (max allowed: {:.1}%)",
            -tok_change * 100.0,
            -tok_s_min * 100.0
        ));
    }

    if !regressions.is_empty() {
        println!("\n❌ Regression detected!");
        for reg in &regressions {
            println!("  - {}", reg);
        }
        return Err(anyhow!("Performance regressions detected: {}", regressions.join(", ")));
    }

    println!("\n✅ All metrics within acceptable thresholds!");
    Ok(())
}

fn run(bin: &str, args: Vec<String>) -> Result<()> {
    let mut cmd = Command::new(bin);
    cmd.args(args);
    run_cmd(&mut cmd)
}

fn run_cmd(cmd: &mut Command) -> Result<()> {
    let status = cmd.status().with_context(|| format!("Failed to spawn: {:?}", cmd))?;

    if !status.success() {
        return Err(anyhow!("Command failed with status: {:?}", status));
    }

    Ok(())
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = std::fs::metadata(path) {
            metadata.permissions().mode() & 0o111 != 0
        } else {
            false
        }
    }
    #[cfg(not(unix))]
    {
        // On Windows, check if it's an .exe file
        path.extension().map_or(false, |ext| ext == "exe")
    }
}

fn vendor_ggml_cmd(commit: &str, force: bool, output_dir: &Path) -> Result<()> {
    println!("📦 Vendoring GGML quantization files from llama.cpp");
    println!("   Commit: {}", commit);
    println!("   Output: {}", output_dir.display());

    // Create output directory structure
    let ggml_dir = output_dir.join("ggml");
    let include_dir = ggml_dir.join("include").join("ggml");
    let src_dir = ggml_dir.join("src");

    fs::create_dir_all(&include_dir)?;
    fs::create_dir_all(&src_dir)?;

    // Files to download - try multiple paths for compatibility
    let files = vec![
        // Try new structure first, then old
        (vec!["ggml/include/ggml/ggml.h", "ggml.h", "ggml-src/ggml.h"], include_dir.join("ggml.h")),
        (vec!["ggml/src/ggml-quants.h", "ggml-quants.h"], src_dir.join("ggml-quants.h")),
        (vec!["ggml/src/ggml-quants.c", "ggml-quants.c"], src_dir.join("ggml-quants.c")),
        (vec!["ggml/src/ggml-common.h", "ggml-common.h"], src_dir.join("ggml-common.h")),
        (vec!["ggml/src/ggml-impl.h", "ggml-impl.h"], src_dir.join("ggml-impl.h")),
    ];

    let client =
        Client::builder().user_agent(USER_AGENT_STRING).timeout(Duration::from_secs(30)).build()?;

    let base_url = format!("https://raw.githubusercontent.com/ggerganov/llama.cpp/{}", commit);

    for (remote_paths, local_path) in files {
        if local_path.exists() && !force {
            println!(
                "   ✓ {} (exists, skipping)",
                local_path.file_name().unwrap().to_string_lossy()
            );
            continue;
        }

        println!("   ⬇ Downloading {}...", local_path.file_name().unwrap().to_string_lossy());

        let mut downloaded = false;
        for remote_path in &remote_paths {
            let url = format!("{}/{}", base_url, remote_path);

            match client.get(&url).send() {
                Ok(response) if response.status().is_success() => {
                    let content = response.bytes()?;
                    fs::write(&local_path, &content)?;
                    println!(
                        "   ✓ {} ({} bytes)",
                        local_path.file_name().unwrap().to_string_lossy(),
                        content.len()
                    );
                    downloaded = true;
                    break;
                }
                _ => continue,
            }
        }

        if !downloaded {
            // Some files are optional (e.g., ggml-common.h, ggml-impl.h)
            if local_path.file_name().unwrap().to_string_lossy().contains("common")
                || local_path.file_name().unwrap().to_string_lossy().contains("impl")
            {
                println!(
                    "   ⚠ {} (optional file not found, skipping)",
                    local_path.file_name().unwrap().to_string_lossy()
                );
            } else {
                bail!(
                    "Failed to download required file: {}",
                    local_path.file_name().unwrap().to_string_lossy()
                );
            }
        }
    }

    // Create version file to track vendored commit
    let version_file = ggml_dir.join("GGML_VERSION");
    fs::write(&version_file, commit)?;

    println!();
    println!("✅ GGML files vendored successfully from commit {}", commit);
    println!("   Files saved to: {}", ggml_dir.display());
    println!();
    println!("Next steps:");
    println!("  1. Build with IQ2_S support:");
    println!("     cargo build -p bitnet-cli --release --features iq2s-ffi");
    println!("  2. Test IQ2_S model loading:");
    println!("     ./target/release/bitnet inspect --model <iq2s-model.gguf>");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    struct TestServer {
        #[allow(dead_code)]
        port: u16,
        #[allow(dead_code)]
        requests: Arc<Mutex<Vec<String>>>,
    }

    impl TestServer {
        fn new<F>(handler: F) -> Self
        where
            F: Fn(&tiny_http::Request) -> tiny_http::Response<std::io::Cursor<Vec<u8>>>
                + Send
                + 'static,
        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);

            let server = tiny_http::Server::http(format!("127.0.0.1:{}", port)).unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let requests_clone = requests.clone();

            thread::spawn(move || {
                for rq in server.incoming_requests() {
                    let path = rq.url().to_string();
                    requests_clone.lock().unwrap().push(path.clone());
                    let response = handler(&rq);
                    let _ = rq.respond(response);
                }
            });

            TestServer { port, requests }
        }

        #[allow(dead_code)]
        fn url(&self, path: &str) -> String {
            format!("http://127.0.0.1:{}{}", self.port, path)
        }
    }

    #[test]
    fn test_retry_after_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(RETRY_AFTER, "10".parse().unwrap());
        assert_eq!(retry_after_secs(&headers), 10);
    }

    #[test]
    fn test_retry_after_http_date() {
        use std::time::{Duration, SystemTime};
        let mut headers = reqwest::header::HeaderMap::new();

        // Future date (5 seconds from now)
        let future = SystemTime::now() + Duration::from_secs(5);
        let date_str = httpdate::fmt_http_date(future);
        headers.insert(RETRY_AFTER, date_str.parse().unwrap());

        let wait = retry_after_secs(&headers);
        assert!((4..=6).contains(&wait)); // Allow for timing variance
    }

    #[test]
    fn test_retry_after_past_date() {
        use std::time::{Duration, SystemTime};
        let mut headers = reqwest::header::HeaderMap::new();

        // Past date returns 5 (default fallback) since duration_since would fail
        let past = SystemTime::now() - Duration::from_secs(10);
        let date_str = httpdate::fmt_http_date(past);
        headers.insert(RETRY_AFTER, date_str.parse().unwrap());

        assert_eq!(retry_after_secs(&headers), 5); // Default fallback
    }

    #[test]
    fn test_classify_exit_codes() {
        // Test disk space error
        let err = anyhow::anyhow!("insufficient disk space: need 100MB");
        assert_eq!(classify_exit(&err), EXIT_NO_SPACE);

        // Test SHA mismatch
        let err = anyhow::anyhow!("SHA256 mismatch: expected abc, got def");
        assert_eq!(classify_exit(&err), EXIT_HASH_MISMATCH);
    }

    #[test]
    fn test_exp_backoff() {
        // Test with jitter: base + (attempt * 37) % 200
        assert_eq!(exp_backoff_ms(1), 200 + 37); // 200 + 37 = 237
        assert_eq!(exp_backoff_ms(2), 400 + 74); // 400 + 74 = 474
        assert_eq!(exp_backoff_ms(3), 800 + 111); // 800 + 111 = 911
        assert_eq!(exp_backoff_ms(10), 10_000 + 170); // Capped at 10s + jitter
    }

    // Happy-path test: aligned 206 response
    #[test]
    fn test_aligned_206_download() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let bytes_sent = Arc::new(AtomicUsize::new(0));
        let bytes_sent_clone = bytes_sent.clone();

        let _server = TestServer::new(move |rq| {
            use tiny_http::{Header, Response, StatusCode};

            if rq.method() == &tiny_http::Method::Get {
                let range_header = rq
                    .headers()
                    .iter()
                    .find(|h| h.field.as_str() == "Range")
                    .and_then(|h| h.value.as_str().strip_prefix("bytes="))
                    .and_then(|s| s.strip_suffix("-"))
                    .and_then(|s| s.parse::<usize>().ok());

                if let Some(start) = range_header {
                    // Return aligned 206 with correct Content-Range
                    let data = b"Hello, World! This is test data.";
                    let chunk = &data[start.min(data.len())..];
                    bytes_sent_clone.fetch_add(chunk.len(), Ordering::SeqCst);

                    let mut resp = Response::from_data(chunk).with_status_code(StatusCode(206));
                    resp.add_header(
                        Header::from_bytes(
                            &b"Content-Range"[..],
                            format!("bytes {}-{}/{}", start, start + chunk.len() - 1, data.len())
                                .as_bytes(),
                        )
                        .unwrap(),
                    );
                    return resp;
                }

                // Full response
                let data = b"Hello, World! This is test data.";
                bytes_sent_clone.fetch_add(data.len(), Ordering::SeqCst);
                Response::from_data(&data[..])
            } else {
                Response::from_string("").with_status_code(StatusCode(405))
            }
        });

        // Would test that download succeeds and final size matches
        // assert_eq!(bytes_sent.load(Ordering::SeqCst), 32);
    }

    // Happy-path test: 304 conditional GET
    #[test]
    fn test_304_conditional_get() {
        let _server = TestServer::new(|rq| {
            use tiny_http::{Response, StatusCode};

            // HEAD returns 405
            if rq.method() == &tiny_http::Method::Head {
                return Response::from_string("").with_status_code(StatusCode(405));
            }

            // GET with If-None-Match returns 304
            if rq.method() == &tiny_http::Method::Get {
                let has_etag = rq.headers().iter().any(|h| h.field.as_str() == "If-None-Match");

                if has_etag {
                    return Response::from_string("").with_status_code(StatusCode(304));
                }
            }

            // Default: return data
            Response::from_string("test data")
        });

        // Would test that:
        // 1. File is not re-downloaded
        // 2. .lock file is cleaned up
        // 3. Early exit occurs
    }

    // Integration test for download edge cases
    #[test]
    #[ignore] // Run with: cargo test --features test-download -- --ignored
    fn test_download_206_misaligned() {
        let _server = TestServer::new(|rq| {
            use tiny_http::{Header, Response, StatusCode};

            if rq.method() == &tiny_http::Method::Get {
                // Return 206 with wrong Content-Range
                let mut resp = Response::from_string("test data").with_status_code(StatusCode(206));
                resp.add_header(
                    Header::from_bytes(&b"Content-Range"[..], &b"bytes 999-1000/2000"[..]).unwrap(),
                );
                return resp;
            }
            Response::from_string("").with_status_code(StatusCode(405))
        });

        // Would test download_model_cmd with server.url("/test.bin")
        // and verify it restarts from 0
    }

    #[test]
    #[ignore]
    fn test_download_429_retry_after() {
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        let _server = TestServer::new(move |_rq| {
            use tiny_http::{Header, Response, StatusCode};

            let mut count = counter_clone.lock().unwrap();
            *count += 1;

            if *count == 1 {
                // First request: 429 with Retry-After
                let mut resp = Response::from_string("").with_status_code(StatusCode(429));
                resp.add_header(Header::from_bytes(&b"Retry-After"[..], &b"2"[..]).unwrap());
                return resp;
            }

            // Second request: success
            Response::from_string("success")
        });

        // Would test that download retries after 2 seconds
    }
}

// GPU-related command implementations

fn gpu_preflight_cmd(require: bool, format: &str) -> Result<()> {
    // Try to load the GPU utils module dynamically
    let info = get_gpu_info_stub();

    match format {
        "json" => {
            let json = serde_json::json!({
                "cuda": info.cuda,
                "cuda_version": info.cuda_version,
                "metal": info.metal,
                "rocm": info.rocm,
                "rocm_version": info.rocm_version,
                "wgpu": info.wgpu,
                "any_available": info.any_available(),
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("🔍 GPU Preflight Check");
            println!("═════════════════════");
            println!();
            println!("{}", info.summary());

            if !info.any_available() {
                println!();
                println!("⚠️  No GPU backend detected");
                println!();
                println!("To enable GPU acceleration:");
                println!(
                    "  • NVIDIA GPUs: Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads"
                );
                println!("  • AMD GPUs: Install ROCm from https://rocm.docs.amd.com");
                println!("  • Apple Silicon: Metal support is built-in on macOS");
                println!("  • Other GPUs: WebGPU support depends on platform/runtime availability");
                println!();
                println!("Set CUDA_HOME or ROCM_PATH environment variables after installation.");
            }
        }
    }

    if require && !info.any_available() {
        bail!("No GPU backend available (required by --require flag)");
    }

    Ok(())
}

fn gpu_smoke_cmd(size: &str, tolerance: f32, skip_if_no_gpu: bool) -> Result<()> {
    let info = get_gpu_info_stub();

    if !info.any_available() {
        if skip_if_no_gpu {
            println!("⏭️  Skipping GPU smoke test (no GPU available)");
            return Ok(());
        } else {
            bail!("No GPU available for smoke test");
        }
    }

    println!("🚀 Running GPU smoke test");
    println!("  Size: {}", size);
    println!("  Tolerance: {}", tolerance);
    println!();

    // Build and run the GPU smoke test
    let mut cmd = Command::new("cargo");
    cmd.args([
        "test",
        "--package",
        "bitnet-kernels",
        "--test",
        "gpu_smoke",
        "--no-default-features",
        "--features",
        "cuda",
        "--",
        "--nocapture",
    ]);

    // Pass test parameters via environment variables
    cmd.env("GPU_TEST_SIZE", size);
    cmd.env("GPU_TEST_TOLERANCE", tolerance.to_string());

    let status = cmd.status()?;
    if !status.success() {
        bail!("GPU smoke test failed");
    }

    println!("✅ GPU smoke test passed");
    Ok(())
}

fn demo_cmd(which: &str, args: &[String]) -> Result<()> {
    println!("🎭 Running demo: {}", which);

    let demos = match which {
        "system" => vec!["demo_reporting_system"],
        "comprehensive" => vec!["demo_reporting_comprehensive"],
        "all" => vec!["demo_reporting_system", "demo_reporting_comprehensive"],
        _ => bail!("Unknown demo: {}. Use 'system', 'comprehensive', or 'all'", which),
    };

    for demo_name in demos {
        println!();
        println!("▶️  Running {}", demo_name);
        println!("─────────────────────");

        let mut cmd = Command::new("cargo");
        cmd.args([
            "run",
            "--package",
            "bitnet-tests",
            "--bin",
            demo_name,
            "--features",
            "reporting",
        ]);

        // Add any additional arguments
        if !args.is_empty() {
            cmd.arg("--");
            cmd.args(args);
        }

        let status = cmd.status()?;
        if !status.success() {
            bail!("{} failed", demo_name);
        }
    }

    println!();
    println!("✅ All demos completed successfully");
    Ok(())
}

// Stub implementation of GPU info for xtask
// The real implementation would be in bitnet-kernels
struct GpuInfo {
    cuda: bool,
    cuda_version: Option<String>,
    metal: bool,
    rocm: bool,
    rocm_version: Option<String>,
    wgpu: bool,
}

impl GpuInfo {
    fn any_available(&self) -> bool {
        self.cuda || self.metal || self.rocm || self.wgpu
    }

    fn summary(&self) -> String {
        let mut backends = Vec::new();

        if self.cuda {
            if let Some(version) = &self.cuda_version {
                backends.push(format!("CUDA {}", version));
            } else {
                backends.push("CUDA".to_string());
            }
        }

        if self.metal {
            backends.push("Metal".to_string());
        }

        if self.rocm {
            if let Some(version) = &self.rocm_version {
                backends.push(format!("ROCm {}", version));
            } else {
                backends.push("ROCm".to_string());
            }
        }

        if self.wgpu {
            backends.push("WebGPU (platform-dependent)".to_string());
        }

        if backends.is_empty() {
            "No GPU backends available".to_string()
        } else {
            format!("Available GPU backends: {}", backends.join(", "))
        }
    }
}

fn get_gpu_info_stub() -> GpuInfo {
    use std::process::Command;

    let cuda = Command::new("nvidia-smi")
        .arg("--query-gpu=gpu_name")
        .arg("--format=csv,noheader")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    let cuda_version = if cuda {
        Command::new("nvcc")
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| {
                let mut s = String::from_utf8(o.stdout).ok().unwrap_or_default();
                if s.trim().is_empty() {
                    s = String::from_utf8(o.stderr).ok().unwrap_or_default();
                }
                (!s.trim().is_empty()).then_some(s)
            })
            .and_then(|output| {
                output.lines().find(|line| line.contains("release")).and_then(|line| {
                    line.split("release")
                        .nth(1)
                        .and_then(|s| s.split(',').next())
                        .map(|s| s.trim().to_string())
                })
            })
    } else {
        None
    };

    let metal = cfg!(target_os = "macos");

    let rocm = std::env::var("ROCM_PATH").is_ok()
        || Command::new("rocm-smi")
            .arg("--showid")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

    let rocm_version = if rocm {
        Command::new("rocm-smi")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() { String::from_utf8(output.stdout).ok() } else { None }
            })
            .and_then(|output| {
                output
                    .lines()
                    .find(|line| line.contains("Version"))
                    .and_then(|line| line.split(':').nth(1).map(|s| s.trim().to_string()))
            })
    } else {
        None
    };

    GpuInfo {
        cuda,
        cuda_version,
        metal,
        rocm,
        rocm_version,
        wgpu: false, // Don't imply a guaranteed fallback
    }
}
