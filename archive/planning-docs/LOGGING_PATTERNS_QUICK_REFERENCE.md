# BitNet.rs Logging and Diagnostics - Quick Reference

## Logging Macros (tracing crate)

```rust
use tracing::{debug, info, warn, error};

tracing::info!("General status")
tracing::debug!("Diagnostic detail")
tracing::warn!("Non-fatal issue")
tracing::error!("Critical failure")
```

**Control via**: `RUST_LOG=debug cargo ...` or `RUST_LOG=info cargo ...`

## Rate-Limited Warnings

```rust
use bitnet_common::warn_once;

// Log once at WARN level, then DEBUG for repeats
warn_once!("key_name", "Message");
warn_once!("key_name", "Formatted: {}", value);
```

## Banner/Status Output (console crate)

```rust
use console::style;

println!("{}", style("Title").bold().cyan());
println!("  Item: {}", style("✓ Success").green());
println!("  Warn: {}", style("⚠ Warning").yellow());
eprintln!("  Error: {}", style("✗ Failed").red());
```

## Debug Output Patterns

| Purpose | Pattern | Enable With |
|---------|---------|-------------|
| Timing | `eprintln!("timing: name_us={}", elapsed.as_micros())` | `BITNET_TRACE_TIMING=1` |
| Logits | `eprintln!("hidden_rms={:.6}", value)` | `BITNET_DEBUG_LOGITS=1` |
| Parity | `eprintln!("{{\"step\":{},\"token\":{}}}", step, token)` | `BITNET_PARITY=1` |

## CLI Debug Flags

```bash
# Timing per inference step
BITNET_TRACE_TIMING=1 bitnet run --model model.gguf --prompt "Test"

# Logit debugging
BITNET_DEBUG_LOGITS=1 bitnet run --model model.gguf --prompt "Test"

# Parity checking (JSON output with top-10 logits)
BITNET_PARITY=1 bitnet run --model model.gguf --prompt "Test"
```

## CLI Debug Flags

```bash
# Dump generated token IDs
bitnet run --model model.gguf --prompt "Test" --dump-ids

# Dump logit steps (first N steps, top-10 tokens)
bitnet run --model model.gguf --prompt "Test" --dump-logit-steps 3

# Validate greedy invariant
bitnet run --model model.gguf --prompt "Test" --greedy --assert-greedy
```

## Diagnostic Commands

```bash
# System information (features, CPU capabilities, GPU status)
bitnet info

# Model validation (LayerNorm statistics)
bitnet inspect --ln-stats model.gguf

# GPU/CPU capability check
cargo run -p xtask -- preflight
```

## Error Handling

```rust
use anyhow::{bail, Context};

// Actionable error with suggestions
bail!("Failed to load: {}\n\nSolutions:\n1. ...\n2. ...", error)

// Error context chain
result.context("Failed to load model")?

// Log error chain (automatic in main)
if let Err(e) = result {
    error!("Failed: {}", e);
    let mut source = e.source();
    while let Some(err) = source {
        error!("  Caused by: {}", err);
        source = err.source();
    }
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 5 | Strict tokenizer failure |
| 8 | Strict mode violation |
| 42 | Greedy argmax mismatch |
| 10-17 | Specific failures (network, auth, etc.) |
| 130 | Interrupted (Ctrl+C) |

## Key Environment Variables

```bash
# Logging
RUST_LOG=debug              # Set log level
RUST_LOG=info,bitnet=debug  # Per-module control

# Timing & Debug
BITNET_TRACE_TIMING=1       # Per-step timing
BITNET_DEBUG_LOGITS=1       # Logit diagnostics
BITNET_PARITY=1             # JSON parity data

# Testing/Validation
BITNET_STRICT_MODE=1        # Require real inference (no mocks)
BITNET_DETERMINISTIC=1      # Reproducible runs
BITNET_SEED=42              # Fixed random seed
BITNET_GPU_FAKE=none        # Fake GPU detection

# Performance
RAYON_NUM_THREADS=4         # CPU thread control
```

## Adding --verbose Support (Code Pattern)

```rust
#[derive(Parser)]
struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

// In setup:
let level = match verbose {
    0 => "info",
    1 => "debug",
    _ => "trace",
};
let filter = tracing_subscriber::EnvFilter::new(level);
```

## File Reference

- **Main CLI**: `/crates/bitnet-cli/src/main.rs` (logging setup, output formatting)
- **Warnings**: `/crates/bitnet-common/src/warn_once.rs` (rate-limiting)
- **Diagnostics**: `/crates/bitnet-cli/src/commands/inspect.rs`
- **Env Variables**: `/docs/environment-variables.md` (complete reference)

## Best Practices

- Use `tracing::*!()` for all diagnostic output
- Use `println!()` + `style()` only for banners/status
- Write logs to stderr, output to stdout
- Use `warn_once!()` for repeated warnings
- Include actionable error suggestions
- Guard debug output with env var checks: `if std::env::var("VAR").is_ok()`

