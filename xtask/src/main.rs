use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{Once, atomic::{AtomicBool, Ordering}},
    thread,
    time::{Duration, Instant},
};
use anyhow::{anyhow, bail, Context, Result};
use fs2::available_space;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle, ProgressDrawTarget};
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE, RANGE, ETAG, IF_NONE_MATCH, LAST_MODIFIED, IF_MODIFIED_SINCE, IF_RANGE, ACCEPT_RANGES, ACCEPT_ENCODING};
use reqwest::StatusCode;
use sha2::{Digest, Sha256};
use walkdir::WalkDir;
use fs2::FileExt;

// Global interrupt flag and setup
static CTRL_ONCE: Once = Once::new();
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

// Centralized defaults to avoid drift
const DEFAULT_MODEL_ID: &str = "microsoft/bitnet-b1.58-2B-4T-gguf";
const DEFAULT_MODEL_FILE: &str = "ggml-model-i2_s.gguf";
const USER_AGENT_STRING: &str = "bitnet-xtask/0.1 (+https://github.com/microsoft/BitNet-rs)";
const DEFAULT_CPP_TAG: &str = "b1-65-ggml";

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
    },

    /// Fetch & build microsoft/BitNet C++ (pinned tag)
    FetchCpp {
        /// Tag or rev to fetch (default: b1-65-ggml)
        #[arg(long, default_value = DEFAULT_CPP_TAG)]
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
            dry_run,
            extra,
        } => {
            let model_path = match model {
                Some(p) => p,
                None => resolve_default_model()?,
            };
            crossval_cmd(&model_path, cpp_dir.as_deref(), release, &extra, dry_run)
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
    
    // Guard against path traversal
    let safe_file = Path::new(file)
        .file_name()
        .ok_or_else(|| anyhow!("invalid file name: {}", file))?;
    
    let dest_dir = out_dir.join(id.replace('/', "-"));
    fs::create_dir_all(&dest_dir)?;
    let dest = dest_dir.join(safe_file);

    let url = format!("https://huggingface.co/{id}/resolve/main/{file}");
    let token = std::env::var("HF_TOKEN").ok();

    // Build client first (needed for conditional checks)
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(30 * 60)) // 30 min for big models
        .user_agent(USER_AGENT_STRING)
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
                    bail!("HTTP {} from Hugging Face during metadata check. If the repo is private, set HF_TOKEN, e.g.\n\
                           HF_TOKEN=*** cargo xtask download-model --id {} --file {}", 
                          resp.status().as_u16(), id, file);
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
            // Check if server supports range requests
            let resumable = r.headers()
                .get(ACCEPT_RANGES)
                .and_then(|h| h.to_str().ok())
                .map(|v| v.eq_ignore_ascii_case("bytes"))
                .unwrap_or(true);
            
            let sz = r.headers()
                .get(CONTENT_LENGTH)?
                .to_str()
                .ok()?
                .parse::<u64>()
                .ok()?;
            Some((sz, resumable))
        })
        .map(|(sz, res)| (Some(sz), res))
        .or_else(|| {
            // Fallback: try 1-byte GET to extract total from Content-Range (with cache headers)
            let mut probe = client.get(&url);
            if let Some(t) = &token {
                probe = probe.header(AUTHORIZATION, format!("Bearer {t}"));
            }
            probe = probe.header(RANGE, "bytes=0-0")
                  .header(ACCEPT_ENCODING, "identity");
            
            // Add conditional headers for cache checking on fallback
            if dest.exists() && !force {
                if let Ok(etag) = fs::read_to_string(&etag_path) {
                    probe = probe.header(IF_NONE_MATCH, etag.trim());
                }
                if let Ok(lastmod) = fs::read_to_string(&lastmod_path) {
                    probe = probe.header(IF_MODIFIED_SINCE, lastmod.trim());
                }
            }
            
            probe.send().ok().and_then(|r| {
                // Check for 304 on the 1-byte probe - means file is current
                if r.status() == StatusCode::NOT_MODIFIED && dest.exists() && !force {
                    // Can't early return from a closure, will handle after
                    return None;
                }
                let sz = r.headers()
                    .get(CONTENT_RANGE)
                    .and_then(|h| h.to_str().ok())
                    .and_then(|s| s.rsplit('/').next()?.parse::<u64>().ok())?;
                Some(sz)
            })
            .map(|sz| (Some(sz), true))
        })
        .unwrap_or((None, true));
    
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
        if let Ok(r) = probe.send() {
            if r.status() == StatusCode::NOT_MODIFIED {
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
            println!("   Resuming from {:.2} MB / {:.2} MB", 
                start as f64 / 1_048_576.0,
                total as f64 / 1_048_576.0);
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
    
    // Single-writer lock to prevent concurrent downloads
    let lock_path = dest.with_extension("lock");
    let lock_file = std::fs::File::create(&lock_path)
        .with_context(|| format!("failed to create lock file for {}", dest.display()))?;
    lock_file.try_lock_exclusive()
        .with_context(|| format!("another download appears to be running for {}", dest.display()))?;
    
    // Request with retry logic and proper range handling
    let mut attempt = 0;
    let max_attempts = 3;
    let mut resp = loop {
        // If tmp larger than remote size, restart clean
        if let Some(total) = size {
            if start > total {
                println!("   Local partial ({:.2} MB) exceeds remote size ({:.2} MB); restarting",
                         start as f64 / 1_048_576.0, total as f64 / 1_048_576.0);
                start = 0;
            }
        }

        let mut rb = client.get(&url);
        if let Some(t) = &token { rb = rb.header(AUTHORIZATION, format!("Bearer {t}")); }
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
        }

        let r = rb.send();

        let r = match r {
            Ok(r) => r,
            Err(e) if attempt < max_attempts => {
                // Handle 429 rate limiting
                if let Some(status) = e.status() {
                    if status == StatusCode::TOO_MANY_REQUESTS {
                        eprintln!("   Rate limited (429). Waiting 5s before retry...");
                        thread::sleep(Duration::from_secs(5));
                        attempt += 1;
                        continue;
                    }
                }
                
                attempt += 1;
                let backoff = Duration::from_millis(200 * (1 << (attempt - 1)));
                println!("   transient error: {e}; retrying in {} ms", backoff.as_millis());
                thread::sleep(backoff);
                continue;
            }
            Err(e) => {
                drop(lock_file);
                let _ = fs::remove_file(&lock_path);
                return Err(e).context("download request failed");
            }
        };

        // If server says the Range was invalid, restart from 0
        if r.status() == StatusCode::RANGE_NOT_SATISFIABLE && start > 0 {
            println!("   Server rejected resume; restarting from 0");
            start = 0;
            attempt += 1;
            if attempt > max_attempts { bail!("persistent 416 Range errors"); }
            continue;
        }

        // Friendlier auth errors
        let status = r.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            bail!(
                "HTTP {} from Hugging Face. If the repo is private, set HF_TOKEN, e.g.\n\
                 HF_TOKEN=*** cargo xtask download-model --id {} --file {}",
                status.as_u16(), id, file
            );
        }
        
        let resp = r.error_for_status()?;
        
        // Verify Content-Range alignment on resume
        if start > 0 && resp.status() == StatusCode::PARTIAL_CONTENT {
            if let Some(cr) = resp.headers().get(CONTENT_RANGE).and_then(|h| h.to_str().ok()) {
                if let Some(begin) = cr.strip_prefix("bytes ")
                                      .and_then(|s| s.split('-').next())
                                      .and_then(|s| s.parse::<u64>().ok()) {
                    if begin != start {
                        // Misaligned resume point - must restart the request
                        eprintln!("   Server resumed at {} (expected {}); restarting from 0", begin, start);
                        drop(resp);  // Drop current response before retry
                        start = 0;
                        
                        if attempt > max_attempts {
                            drop(lock_file);
                            let _ = fs::remove_file(&lock_path);
                            bail!("failed to download after {} attempts due to misaligned resume", max_attempts);
                        }
                        // Small backoff before retry
                        thread::sleep(Duration::from_millis(200 << (attempt - 1)));
                        continue;  // Continue the outer loop with a fresh GET (no Range)
                    }
                }
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

    // Setup progress bar (hide if not a TTY)
    let pb = if atty::is(atty::Stream::Stderr) {
        if let Some(total) = size {
            let pb = ProgressBar::new(total);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}"
                )?
                .progress_chars("##-")
            );
            pb
        } else {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} downloading {bytes} {msg}"
                )?
            );
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

    let mut file_out = if resumed && resp.status() == StatusCode::OK {
        // Server ignored Range, need to truncate and restart
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?
    } else {
        // Normal case: append if resuming, write if new
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(start > 0)
            .open(&tmp)?
    };
    
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
    let mut buf = vec![0u8; 1024 * 256]; // 256KB buffer
    let start_time = Instant::now();
    
    loop {
        // Check for interruption
        if INTERRUPTED.load(Ordering::SeqCst) {
            pb.finish_with_message("interrupted (partial file kept for resume)");
            println!("   Partial download saved at: {}", tmp.display());
            println!("   Run the same command again to resume");
            return Ok(());
        }
        
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file_out.write_all(&buf[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }
    
    file_out.flush()?;
    file_out.sync_all()?;  // Ensure data is on disk before rename
    drop(file_out);
    
    let elapsed = start_time.elapsed();
    let secs = elapsed.as_secs_f64().max(0.001); // Avoid division by zero
    let throughput = (downloaded - start) as f64 / secs / 1_048_576.0;
    
    pb.finish_with_message(format!("complete ({:.2} MB/s)", throughput));

    // Atomic rename BEFORE persisting metadata
    fs::rename(&tmp, &dest)?;
    
    // Persist ETag and Last-Modified AFTER successful rename
    if let Some(etag) = resp.headers().get(ETAG) {
        if let Ok(etag_str) = etag.to_str() {
            let _ = fs::write(&etag_path, etag_str);
        }
    }
    if let Some(lm) = resp.headers().get(LAST_MODIFIED) {
        if let Ok(lm_str) = lm.to_str() {
            let _ = fs::write(&lastmod_path, lm_str);
        }
    }
    
    // Verify final size if known
    if let Some(total) = size {
        let actual = fs::metadata(&dest)?.len();
        if actual != total {
            bail!("download truncated: got {} bytes, expected {}", actual, total);
        }
    }

    // Clean up lock file
    drop(lock_file);
    let _ = fs::remove_file(&lock_path);
    
    if let Some(want) = sha256_hex {
        print!("🔒 Verifying SHA256... ");
        std::io::stdout().flush()?;
        if let Err(e) = verify_sha256(&dest, want) {
            // Remove bad file and cache files
            let _ = fs::remove_file(&dest);
            let _ = fs::remove_file(&etag_path);
            let _ = fs::remove_file(&lastmod_path);
            return Err(e);
        }
        println!("✓ OK");
    }
    
    println!("✅ Saved: {}", dest.display());
    if let Some(size) = size {
        println!("   Size: {:.2} MB", size as f64 / 1_048_576.0);
    }
    println!("   Time: {:.1}s", elapsed.as_secs_f64());
    println!("   Speed: {:.2} MB/s", throughput);
    
    // Print ready-to-use export command
    let abs_path = dest.canonicalize().unwrap_or(dest.clone());
    println!();
    println!("To use this model for cross-validation:");
    println!("  export CROSSVAL_GGUF=\"{}\"", abs_path.display());
    
    Ok(())
}

fn resolve_default_model() -> Result<PathBuf> {
    let root = PathBuf::from("models");
    if !root.exists() {
        return Err(anyhow!(
            "No models directory found. Run `cargo xtask download-model` first."
        ));
    }
    
    // Prefer default model path
    let preferred = root.join(format!("{}/{}", 
        DEFAULT_MODEL_ID.replace('/', "-"), 
        DEFAULT_MODEL_FILE));
    if preferred.exists() {
        return Ok(preferred);
    }
    
    // Fallback: scan for first *.gguf file
    for entry in WalkDir::new(&root).into_iter().filter_map(Result::ok) {
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension() {
                if ext == "gguf" {
                    return Ok(entry.path().to_path_buf());
                }
            }
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
    dry_run: bool,
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
    // Echo the absolute path so users know exactly what was picked
    if let Ok(abs_model) = model.canonicalize() {
        println!("   Absolute: {}", abs_model.display());
    }
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
        .env("GGML_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("RUST_BACKTRACE", "1");
    
    // Add test runner args
    cmd.arg("--")
        .args(["--nocapture", "--test-threads=1"])
        .args(extra);

    if dry_run {
        println!("\n[DRY RUN] Env + command:");
        println!("  BITNET_CPP_DIR={}", cpp.display());
        println!("  CROSSVAL_GGUF={}", model.display());
        println!("  OMP_NUM_THREADS=1 GGML_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1");
        println!("  RUST_BACKTRACE=1");
        println!("  {:?}", cmd);
        return Ok(());
    }

    run_cmd(&mut cmd)
}

fn full_crossval_cmd(force: bool) -> Result<()> {
    println!("🚀 Running full cross-validation workflow");
    println!();
    
    // Step 1: Download model
    println!("Step 1/3: Downloading model");
    download_model_cmd(
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_FILE,
        &PathBuf::from("models"),
        None, // Add SHA256 if available
        force,
    )?;
    
    println!();
    
    // Step 2: Fetch C++ implementation
    println!("Step 2/3: Fetching C++ implementation");
    fetch_cpp_cmd(DEFAULT_CPP_TAG, force, false)?;
    
    println!();
    
    // Step 3: Run tests
    println!("Step 3/3: Running cross-validation tests");
    let model = PathBuf::from(format!("models/{}/{}",
        DEFAULT_MODEL_ID.replace('/', "-"),
        DEFAULT_MODEL_FILE));
    crossval_cmd(&model, None, true, &[], false)?;
    
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