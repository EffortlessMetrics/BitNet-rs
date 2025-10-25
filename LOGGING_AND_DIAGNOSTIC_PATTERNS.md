# BitNet.rs Logging and Diagnostic Patterns Report

## Executive Summary

BitNet.rs uses a multi-layered logging and diagnostic infrastructure combining `tracing` for structured logging, custom warn-once rate limiting, environment-based debug output, and CLI banner/status formatting via the `console` crate. The system supports multiple verbosity levels through `RUST_LOG`, specialized diagnostic modes via environment variables, and exit codes for error classification.

---

## 1. Logging Infrastructure

### 1.1 Primary Logging Framework: Tracing

The project uses the Rust `tracing` crate for structured, leveled logging across all components.

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/main.rs` (lines 476-634)

**Setup Function**:
```rust
fn setup_logging(config: &CliConfig, log_level_override: Option<&str>) -> Result<()> {
    let level = log_level_override.unwrap_or(&config.logging.level);

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr);

    match config.logging.format.as_str() {
        "json" => {
            subscriber.json().with_timer(tracing_subscriber::fmt::time::uptime()).init();
        }
        "compact" => {
            subscriber.compact().init();
        }
        _ => {
            subscriber.pretty().init();
        }
    }
    Ok(())
}
```

**Key Features**:
- Filters configured via `RUST_LOG` environment variable
- Supports multiple output formats: `json`, `compact`, `pretty` (default)
- Logs written to stderr (keeps stdout clean for output data)
- `with_target(false)` reduces noise by hiding module paths

**Common Log Levels Used**:
- `error!()` - Critical failures and user errors
- `warn!()` - Non-fatal issues (first occurrence)
- `info!()` - General information, status messages
- `debug!()` - Detailed diagnostic information
- `trace!()` - Very detailed low-level debugging (rarely used)

### 1.2 Logging Integration Points

**Model Loading** (`crates/bitnet-models/src/formats/gguf/loader.rs`, lines 13):
```rust
use tracing::{debug, info};
```
- Logs model metadata extraction
- Logs tensor validation results
- Logs architecture detection

**Tokenizer Loading** (`crates/bitnet-cli/src/main.rs`, lines 1026-1028):
```rust
match bitnet_tokenizers::loader::load_tokenizer_from_gguf_reader(&reader) {
    Ok(tok) => {
        println!("Successfully loaded SentencePiece tokenizer from GGUF");
        tok
    }
```

**Inference Pipeline** (`crates/bitnet-cli/src/main.rs`, lines 1098-1101):
```rust
debug!(
    "Template: {} | Stop sequences: {:?} | Stop IDs: {:?}",
    template_type, all_stop_sequences, all_stop_ids
);
```

---

## 2. Warn-Once Rate Limiting Pattern

### 2.1 Purpose and Design

Rate-limited warnings prevent log spam from repeated warnings in hot paths. Each unique warning key is logged at WARN level once, then at DEBUG level for subsequent occurrences.

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-common/src/warn_once.rs`

### 2.2 Implementation

**Thread-Safe Registry**:
```rust
use std::sync::{Mutex, OnceLock};

static WARN_REGISTRY: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

pub fn warn_once_fn(key: &str, message: &str) {
    let registry = get_registry();
    let mut seen = match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    if seen.insert(key.to_string()) {
        // First occurrence - log at WARN level
        tracing::warn!(key = %key, "{}", message);
    } else {
        // Subsequent occurrence - log at DEBUG level
        tracing::debug!(key = %key, "(rate-limited) {}", message);
    }
}
```

**Macro Interface**:
```rust
#[macro_export]
macro_rules! warn_once {
    ($key:expr, $($arg:tt)*) => {
        $crate::warn_once_fn($key, &format!($($arg)*))
    };
}
```

### 2.3 Usage Examples

```rust
use bitnet_common::warn_once;

// Simple message
warn_once!("deprecated_api_v1", "Using deprecated API v1, please migrate to v2");

// Formatted message
warn_once!("model_fallback", "Falling back to CPU for operation: {}", operation_name);
```

**Benefits**:
- Prevents DOS-like log spam in loops
- Preserves first warning (most important for troubleshooting)
- Thread-safe with zero unsafe code
- Easy testing with `clear_registry_for_test()`

---

## 3. Banner and Status Output Patterns

### 3.1 Styled Console Output

The `console` crate provides colored, styled terminal output via the `style()` function.

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/main.rs` (line 15)

```rust
use console::style;
```

### 3.2 Banner Examples

**Interactive Chat Banner** (`crates/bitnet-cli/src/commands/chat.rs`):
```rust
println!("{}", style("BitNet Interactive Chat").bold().cyan());
println!("Chat ready!");
println!("Template: {}", style(format!("{:?}", template_type)).dim());
```

**System Info Banner** (`crates/bitnet-cli/src/main.rs`, lines 1562-1642):
```rust
async fn show_system_info() -> Result<()> {
    println!("{}", style("BitNet System Information").bold().cyan());
    
    println!("{}", style("Version:").bold());
    println!("  BitNet CLI: {}", env!("CARGO_PKG_VERSION"));
    
    println!("{}", style("Features:").bold());
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        println!("  GPU support: {}", style("✓ Enabled").green());
    }
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    {
        println!("  GPU support: {}", style("✗ Disabled").red());
    }
}
```

**Style Options**:
- `.bold()` - Bold text
- `.cyan()`, `.green()`, `.red()`, `.yellow()` - Colors
- `.dim()` - Dimmed/gray text
- Chainable: `style("text").bold().cyan()`

### 3.3 Backend Selection Indicators

**QK256 Performance Warning** (`crates/bitnet-cli/src/main.rs`, lines 809-835):
```rust
if avx2_available {
    eprintln!("{} Using QK256 quantization with AVX2 acceleration", 
              style("ℹ").cyan().bold());
    return Ok(());
}

eprintln!();
eprintln!("{}", style("⚠  WARNING: Using QK256 scalar kernels (~0.1 tok/s)").yellow().bold());
eprintln!("For quick validation, use --max-tokens 4-16");
eprintln!("Performance: ~10 seconds per token (2B models)");
```

**GPU Status Output**:
```rust
match candle_core::Device::cuda_if_available(0).is_ok() {
    true => println!("  CUDA: {}", style("✓ Available").green()),
    false => println!("  CUDA: {}", style("✗ Not available").red()),
}
```

---

## 4. Debug Output Patterns

### 4.1 Environment-Controlled Debug Output

Several special debug modes are available via environment variables:

**Timing Instrumentation** (`crates/bitnet-cli/src/main.rs`, lines 1143-1192):
```rust
let timing_enabled = std::env::var("BITNET_TRACE_TIMING").as_deref() == Ok("1");

for step_idx in 0..max_new_tokens {
    let t0 = if timing_enabled { Some(std::time::Instant::now()) } else { None };
    let x = model.embed(&[last_token])?;
    if let Some(t) = t0 {
        eprintln!("timing: embed_us={}", t.elapsed().as_micros());
    }
}
```

**Logits Debug Mode** (`crates/bitnet-cli/src/main.rs`, lines 1180-1212):
```rust
if std::env::var("BITNET_DEBUG_LOGITS").as_deref() == Ok("1") && step_idx == 0 {
    let h_vec = tensor_to_vec(&last_hidden)?;
    let hidden_rms = compute_rms(&h_vec);
    eprintln!("hidden_rms={:.6}", hidden_rms);
    
    let logits_shape = logits.shape();
    eprintln!("logits_shape=(rows={}, cols={})", 
              logits_shape.first().copied().unwrap_or(1),
              logits_shape.get(1).copied().unwrap_or(logits_vec.len()));
    
    let top = &idx[..idx.len().min(5)];
    eprintln!("top5_idx={:?}", top);
    eprintln!("top5_val={:?}", top.iter().map(|&i| logits_vec[i]).collect::<Vec<_>>());
}
```

**Parity Logging** (`crates/bitnet-cli/src/main.rs`, lines 1259-1277):
```rust
if std::env::var("BITNET_PARITY").as_deref() == Ok("1") {
    let mut logits_with_idx: Vec<(usize, f32)> = 
        logits_vec.iter().copied().enumerate().collect();
    logits_with_idx.sort_by(|a, b| 
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let top_k_logits: Vec<(u32, f32)> = 
        logits_with_idx.iter().take(10)
            .map(|(idx, logit)| (*idx as u32, *logit))
            .collect();
    
    eprintln!("{{\"step\":{},\"token\":{},\"top_k\":{}}}",
              step_idx, next_token, 
              serde_json::to_string(&top_k_logits).unwrap_or_default());
}
```

### 4.2 Debug Output Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `BITNET_TRACE_TIMING=1` | Measure per-step timing (embed, forward, logits, sample) | `embed_us=12345` |
| `BITNET_DEBUG_LOGITS=1` | Dump hidden state RMS and top-5 logits on first step | Diagnostics for divergence |
| `BITNET_PARITY=1` | JSON format with chosen token + top-10 logits | `{"step":0,"token":123,"top_k":[...]}` |

---

## 5. Token ID Debug Output (--dump-ids)

### 5.1 CLI Flag Integration

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/main.rs` (lines 252-254)

```rust
/// Dump token IDs to stdout
#[arg(long, default_value_t = false)]
dump_ids: bool,
```

### 5.2 Output Implementation

```rust
if dump_ids {
    println!("Token IDs: {:?}", generated_tokens);
}
```

**Usage**:
```bash
bitnet run --model model.gguf --prompt "What is 2+2?" --dump-ids --max-tokens 4
# Output:
# Token IDs: [1357, 1358, 1359, 1360]
```

**Use Cases**:
- Debugging tokenization issues
- Verifying prompt encoding
- Comparing token sequences across backends (Rust vs C++)

---

## 6. Logit Dumping (--dump-logit-steps)

### 6.1 CLI Interface

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/main.rs` (lines 288-294)

```rust
/// Dump logit steps during generation (max steps)
#[arg(long)]
dump_logit_steps: Option<usize>,

/// Top-k tokens to include in logit dump
#[arg(long, default_value = "10", value_name = "K")]
logits_topk: usize,

/// Assert greedy argmax invariant when dumping logits
#[arg(long, default_value_t = false)]
assert_greedy: bool,
```

### 6.2 Logit Capture Implementation

```rust
#[derive(Debug, serde::Serialize)]
struct LogitStep {
    step: usize,
    top_logits: Vec<serde_json::Value>,
    chosen_id: Option<u32>,
}

// Capture logits if requested
if dump_logit_steps.is_some_and(|max_steps| step_idx < max_steps) {
    let mut indexed: Vec<(usize, f32)> = 
        logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| match (a.1.is_finite(), b.1.is_finite()) {
        (false, true) => std::cmp::Ordering::Greater,
        (true, false) => std::cmp::Ordering::Less,
        _ => {
            let cmp = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
            if cmp == std::cmp::Ordering::Equal { a.0.cmp(&b.0) } else { cmp }
        }
    });
    
    let top_logits: Vec<(u32, f32)> = 
        indexed.into_iter().take(logits_topk)
            .map(|(i, _)| (i as u32, logits_vec[i]))
            .collect();
    
    let step = LogitStep {
        step: step_idx,
        top_logits: top_logits.iter().map(|&(id, logit)| 
            serde_json::json!({"token_id": id, "logit": logit})
        ).collect(),
        chosen_id: None,
    };
    logits_dump.push(step);
}
```

### 6.3 Output Verification

**Greedy Invariant Check** (`crates/bitnet-cli/src/main.rs`, lines 1279-1294):
```rust
if assert_greedy && greedy && dump_logit_steps.is_some_and(|max_steps| step_idx < max_steps) {
    let (mut best_i, mut best_v) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits_vec.iter().enumerate() {
        if v.is_finite() && v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    if next_token as usize != best_i {
        eprintln!("ERROR: Non-argmax token chosen in --greedy at step {}", step_idx);
        eprintln!("  argmax={} (logit={:.4}) but chosen={}", best_i, best_v, next_token);
        std::process::exit(EXIT_ARGMAX_MISMATCH);
    }
}
```

---

## 7. Error Reporting and Exit Codes

### 7.1 Structured Exit Codes

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/exit.rs`

```rust
pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_STRICT_TOKENIZER: i32 = 5;
pub const EXIT_STRICT_MODE: i32 = 8;
pub const EXIT_ARGMAX_MISMATCH: i32 = 42;
pub const EXIT_INFERENCE_FAILED: i32 = 16;
```

**xtask Exit Codes** (`/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs`, lines 116-126):

```rust
const EXIT_SUCCESS: i32 = 0;
const EXIT_NO_SPACE: i32 = 10;
const EXIT_AUTH: i32 = 11;
const EXIT_RATE_LIMIT: i32 = 12;
const EXIT_HASH_MISMATCH: i32 = 13;
const EXIT_NETWORK: i32 = 14;
const EXIT_VERIFICATION_FAILED: i32 = 15;
const EXIT_INFERENCE_FAILED: i32 = 16;
const EXIT_BENCHMARK_FAILED: i32 = 17;
const EXIT_INTERRUPTED: i32 = 130;
```

### 7.2 Error Chain Logging

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/main.rs` (lines 570-581)

```rust
if let Err(e) = result {
    error!("Command failed: {}", e);

    // Show error chain
    let mut source = e.source();
    while let Some(err) = source {
        error!("  Caused by: {}", err);
        source = err.source();
    }

    std::process::exit(1);
}
```

### 7.3 Detailed Error Messages

**Model Loading Failure** (`crates/bitnet-cli/src/main.rs`, lines 941-953):
```rust
if !allow_mock {
    anyhow::bail!(
        "Failed to load real model: {e}\n\
         To run with mock tensors (for smoke/UX testing only), \
         pass --allow-mock or set BITNET_ALLOW_MOCK=1"
    );
}
```

**Tokenizer Resolution Failure** (`crates/bitnet-cli/src/main.rs`, lines 1039-1053):
```rust
anyhow::bail!(
    "Failed to load tokenizer from GGUF: {e}\n\
     \n\
     No tokenizer found. Solutions:\n\
     1. Download tokenizer:\n\
        cargo run -p xtask -- tokenizer --into {}\n\
     2. Provide explicit tokenizer path:\n\
        --tokenizer /path/to/tokenizer.json\n\
     3. Use mock tokenizer for testing:\n\
        --allow-mock",
    model_dir.display()
);
```

---

## 8. Diagnostic Commands

### 8.1 Preflight Command

**Purpose**: Check GPU compilation and runtime availability

**Testing** (`xtask/tests/preflight.rs`):
```rust
#[test]
fn ac5_preflight_detects_gpu_with_fake_cuda() {
    let output = Command::new("cargo")
        .args(["run", "-p", "xtask", "--", "preflight"])
        .env("BITNET_GPU_FAKE", "cuda")
        .output()
        .expect("Failed to run xtask preflight");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    let indicates_gpu = combined_output.contains("GPU: Available")
        || combined_output.contains("GPU: ✓")
        || combined_output.contains("CUDA");
    
    assert!(indicates_gpu, "Preflight should report GPU present");
}
```

**Output Indicators**:
- GPU available: `"GPU: Available"`, `"GPU: ✓"`, `"CUDA"`
- GPU unavailable: `"GPU: Not available"`, `"GPU: ✗"`

### 8.2 Inspect Command

**Purpose**: Validate LayerNorm gamma statistics and projection weights

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/commands/inspect.rs`

```rust
pub async fn execute(&self) -> Result<()> {
    if self.ln_stats {
        self.check_ln_gamma_stats().await
    } else {
        anyhow::bail!(
            "No inspection mode specified. Use --ln-stats to check LayerNorm gamma statistics."
        );
    }
}
```

**Usage**:
```bash
# Architecture-aware validation
bitnet inspect --ln-stats --gate auto model.gguf

# Custom policy
bitnet inspect --ln-stats --gate policy --policy rules.yml model.gguf

# JSON output
bitnet inspect --ln-stats --json model.gguf
```

### 8.3 System Info Command

**Purpose**: Display compilation features, GPU support, CPU features, quantization types

**Output**:
```
BitNet System Information

Version:
  BitNet CLI: 0.1.0

System:
  OS: linux
  Architecture: x86_64
  CPU cores: 8

Features:
  GPU support: ✓ Enabled

CPU features:
  AVX2: ✓
  AVX-512: ✗

Quantization types:
  I2_S (2-bit signed): ✓
  TL1 (ARM optimized): ✓
  TL2 (x86 optimized): ✓
```

---

## 9. Verbose Mode Implementation Guide

### 9.1 Current State: No Global --verbose Flag

BitNet.rs does **not** currently have a global `--verbose` flag. Instead, it uses:

1. **RUST_LOG environment variable** (standard Rust logging)
   ```bash
   RUST_LOG=debug bitnet run --model model.gguf --prompt "Test"
   RUST_LOG=trace cargo test --workspace
   ```

2. **Specialized debug flags** (environment variables)
   ```bash
   BITNET_TRACE_TIMING=1 bitnet run --model model.gguf --prompt "Test"
   BITNET_DEBUG_LOGITS=1 bitnet run --model model.gguf --prompt "Test"
   ```

3. **Inspection commands** (dedicated diagnostic tools)
   ```bash
   bitnet inspect --ln-stats model.gguf
   bitnet info  # System information
   ```

### 9.2 Adding --verbose Flag (Recommended Pattern)

To add a global `--verbose` flag:

```rust
// In Cli struct
#[derive(Parser)]
struct Cli {
    /// Enable verbose output (repeat for more verbosity: -v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
    
    // ... other fields
}

// In logging setup
fn setup_logging(verbose: u8) -> Result<()> {
    let level = match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    
    let filter = tracing_subscriber::EnvFilter::new(level);
    // ... rest of setup
}
```

### 9.3 Silent Mode Implementation

```rust
// In Cli struct
#[derive(Parser)]
struct Cli {
    /// Suppress output (only errors to stderr)
    #[arg(short, long, global = true)]
    quiet: bool,
    
    // ... other fields
}

// During setup
if cli.quiet {
    // Suppress all but error logs
    let filter = tracing_subscriber::EnvFilter::new("error");
}
```

---

## 10. Banner Output Patterns for Different Scenarios

### 10.1 Chat Mode Banner
```rust
println!("{}", style("BitNet Interactive Chat").bold().cyan());
println!("Chat ready!");
println!("Template: {}", style(format!("{:?}", template_type)).dim());
```

### 10.2 Device Selection Banner (Recommended)
```rust
// Print device banner before loading
match device {
    Device::Cpu => {
        println!("{} Running on CPU", style("✓").cyan());
        println!("  Threads: {} (use RAYON_NUM_THREADS to override)", rayon::current_num_threads());
    }
    Device::Cuda => {
        match bitnet_kernels::device_features::gpu_available_runtime() {
            true => println!("{} GPU available: CUDA", style("✓").green()),
            false => println!("{} GPU requested but not available", style("⚠").yellow()),
        }
    }
}
```

### 10.3 Performance Warning Banner
```rust
eprintln!("{}", style("⚠  WARNING: Using QK256 scalar kernels (~0.1 tok/s)").yellow().bold());
eprintln!("For quick validation, use --max-tokens 4-16");
eprintln!("Performance: ~10 seconds per token (2B models)");
eprintln!();
eprintln!("SIMD optimizations coming in v0.2.0 (≥3× faster)");
```

### 10.4 Status Progress Indicator
Uses `indicatif` crate (not yet visible in sampled code but available):
```rust
use indicatif::{ProgressBar, ProgressStyle};

let pb = ProgressBar::new(total_steps);
pb.set_style(ProgressStyle::default_bar()
    .template("{spinner:.cyan} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
    .progress_chars("#>-"));
```

---

## 11. Best Practices Summary

### 11.1 Logging Do's

**DO**:
- Use `tracing::info!()` for user-visible status
- Use `tracing::debug!()` for diagnostic information
- Use `tracing::error!()` for unrecoverable failures
- Use `warn_once!()` for repeated warnings in hot paths
- Write logs to stderr (keep stdout for output data)
- Use `style()` from `console` crate for banner output
- Include context in error messages (suggestions, alternatives)
- Use structured logging with named fields: `debug!("tensor: {}", name)`

**DON'T**:
- Use `println!()` for diagnostic output (use tracing)
- Call `eprintln!()` directly except for banners
- Mix stdout and stderr for the same logical output
- Log sensitive information (API keys, private model paths)
- Ignore error context (always include `source()` in error chains)

### 11.2 Diagnostic Output Guidelines

- **Timing data** → `eprintln!()` with `BITNET_TRACE_TIMING=1` guard
- **Intermediate values** → `eprintln!()` with `BITNET_DEBUG_*` guards
- **User warnings** → `warn_once!()` for first-time issues
- **Status/progress** → `println!()` with `style()` formatting
- **Errors** → Use `anyhow::bail!()` or `tracing::error!()`

### 11.3 Exit Code Guidelines

Use structured exit codes:
- `0` - Success
- `1` - General error
- `5` - Strict tokenizer failure
- `8` - Strict mode violation
- `10-17` - Specific operation failures (network, auth, etc.)

---

## 12. Example: Adding Verbose Support to Custom Command

```rust
use tracing::{debug, info};
use console::style;

#[derive(Args)]
pub struct CustomCommand {
    /// Enable verbose output
    #[arg(long, action = clap::ArgAction::Count)]
    verbose: u8,
    
    #[arg(short, long)]
    model: PathBuf,
}

impl CustomCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.verbose > 0 {
            println!("{}", style("Verbose mode enabled").dim());
        }
        
        if self.verbose > 1 {
            debug!("Loading model from: {}", self.model.display());
        }
        
        let model = load_model(&self.model)?;
        info!("Model loaded successfully");
        
        if self.verbose > 0 {
            println!("  Layers: {}", model.num_layers());
            println!("  Parameters: {}", model.num_parameters());
        }
        
        Ok(())
    }
}
```

---

## Key Files Reference

| File | Purpose | Key Patterns |
|------|---------|--------------|
| `/crates/bitnet-cli/src/main.rs` | CLI setup, logging config | `setup_logging()`, performance warnings |
| `/crates/bitnet-cli/src/commands/inspect.rs` | Diagnostics | Validation gates, error reporting |
| `/crates/bitnet-cli/src/commands/chat.rs` | Interactive mode | Banner output, styled formatting |
| `/crates/bitnet-common/src/warn_once.rs` | Rate-limited warnings | `warn_once!()` macro, thread-safe registry |
| `/docs/environment-variables.md` | Environment config | All debug flags documented |

---

## Conclusion

BitNet.rs implements a sophisticated, multi-layered logging and diagnostic system that:

1. **Separates concerns**: Structured logging via tracing, unstructured debug output via eprintln!, banners via styled println!
2. **Prevents log spam**: warn_once!() rate-limiting for hot paths
3. **Enables detailed diagnostics**: Environment variables for specialized debug modes
4. **Maintains clean output**: Logs to stderr, data to stdout
5. **Provides rich error context**: Error chains with suggestions and alternatives
6. **Supports production and development**: Strict mode for production, flexible diagnostics for development

The system follows Rust logging best practices while maintaining flexibility for BitNet-specific diagnostics (timing, logit dumps, parity checking).

