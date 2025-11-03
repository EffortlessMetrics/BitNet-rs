# Logging and Diagnostic Patterns - Documentation Index

This directory contains comprehensive documentation about the logging, diagnostic, and error reporting patterns used throughout the BitNet.rs codebase.

## Quick Start

Start with the **Quick Reference** for copy-paste patterns:
- **File**: `LOGGING_PATTERNS_QUICK_REFERENCE.md`
- **Read Time**: 5 minutes
- **Purpose**: Get common patterns for your code

For in-depth understanding, read the **Full Guide**:
- **File**: `LOGGING_AND_DIAGNOSTIC_PATTERNS.md`
- **Read Time**: 20-30 minutes
- **Purpose**: Understand all infrastructure, design decisions, best practices

## Documents Overview

### 1. LOGGING_PATTERNS_QUICK_REFERENCE.md (4.4 KB)

One-page cheat sheet for developers. Contains:

- Logging macros (tracing framework)
- Rate-limited warnings (warn_once!)
- Banner/status output (console::style)
- Debug output patterns
- CLI debug flags
- Diagnostic commands
- Error handling patterns
- Exit codes table
- Environment variables reference
- Code pattern templates

**Use when**: You need to quickly copy a pattern or lookup a specific technique.

### 2. LOGGING_AND_DIAGNOSTIC_PATTERNS.md (23 KB)

Comprehensive technical reference. Contains:

1. **Logging Infrastructure** - tracing framework setup
2. **Warn-Once Rate Limiting** - Thread-safe warning deduplication
3. **Banner and Status Output** - Styled console formatting
4. **Debug Output Patterns** - Timing, logits, parity diagnostics
5. **Token ID Debug Output** - --dump-ids CLI flag
6. **Logit Dumping** - --dump-logit-steps with verification
7. **Error Reporting and Exit Codes** - Structured error handling
8. **Diagnostic Commands** - preflight, inspect, info
9. **Verbose Mode Implementation Guide** - How to add --verbose support
10. **Banner Output Patterns** - Different scenarios
11. **Best Practices Summary** - Do's and don'ts
12. **Example: Adding Verbose Support** - Complete code example

Each section includes:
- File locations with line numbers
- Code snippets
- Usage examples
- Design rationale

**Use when**: You need deep understanding of how something works or why it was designed that way.

## Key Patterns Documented

### Logging (Structured with tracing)
```rust
use tracing::{info, debug, error};
info!("User-visible status");
debug!("Diagnostic detail");
// Control via: RUST_LOG=debug cargo ...
```

### Rate-Limited Warnings
```rust
use bitnet_common::warn_once;
warn_once!("unique_key", "Warning message");
// First occurrence: WARN level, then DEBUG (rate-limited)
```

### Banner Output (Colored)
```rust
use console::style;
println!("{}", style("Title").bold().cyan());
println!("  Status: {}", style("✓").green());
```

### Debug Output (Environment-Controlled)
```rust
if std::env::var("BITNET_DEBUG_LOGITS").is_ok() {
    eprintln!("debug_data: {:?}", value);
}
// Control via: BITNET_DEBUG_LOGITS=1 cargo ...
```

### Error Messages (Actionable)
```rust
anyhow::bail!(
    "Main error\n\nSolutions:\n1. Option A\n2. Option B\n3. Option C"
)
```

### Diagnostic Commands
```bash
bitnet info                    # System capabilities
bitnet inspect --ln-stats      # Model validation
cargo run -p xtask -- preflight  # GPU/CPU detection
```

## Important Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `RUST_LOG` | Set log level | `RUST_LOG=debug cargo test` |
| `BITNET_TRACE_TIMING=1` | Per-step timing | Measure embed, forward, logits, sample |
| `BITNET_DEBUG_LOGITS=1` | Logit diagnostics | Dump hidden state RMS and top-5 |
| `BITNET_PARITY=1` | Parity checking | JSON with chosen token + top-10 |
| `BITNET_STRICT_MODE=1` | Require real inference | No mock fallbacks |
| `RAYON_NUM_THREADS=N` | CPU thread control | Override parallelism |

## Important Files in Codebase

| File | Purpose | Key Sections |
|------|---------|--------------|
| `/crates/bitnet-cli/src/main.rs` | CLI setup and inference | setup_logging(), check_and_warn_qk256_performance(), run_simple_generation() |
| `/crates/bitnet-common/src/warn_once.rs` | Rate-limited warnings | warn_once_fn(), warn_once! macro |
| `/crates/bitnet-cli/src/commands/inspect.rs` | Model validation | LayerNorm gamma checking, architecture-aware rules |
| `/crates/bitnet-cli/src/commands/chat.rs` | Interactive mode | Banner output, styling |
| `/docs/environment-variables.md` | Environment config | Complete reference |

## How to Add Logging to New Code

1. **For general diagnostic messages**:
   ```rust
   use tracing::debug;
   debug!("Loading model from: {}", path.display());
   ```

2. **For repeated warnings in hot paths**:
   ```rust
   use bitnet_common::warn_once;
   warn_once!("model_fallback", "Falling back to CPU");
   ```

3. **For debug output (development only)**:
   ```rust
   if std::env::var("BITNET_DEBUG_CUSTOM").is_ok() {
       eprintln!("custom_debug: {:?}", value);
   }
   ```

4. **For colored status output**:
   ```rust
   use console::style;
   println!("{} Device: {}", style("✓").green(), device_name);
   ```

5. **For errors with solutions**:
   ```rust
   anyhow::bail!(
       "Failed to load: {}\n\n\
        Solutions:\n\
        1. Provide --model flag\n\
        2. Set BITNET_GGUF environment variable\n\
        3. Use default model path",
       error
   )
   ```

## How to Add Verbose Support

See section 9.2 in LOGGING_AND_DIAGNOSTIC_PATTERNS.md for complete pattern:

```rust
#[derive(Parser)]
struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

// Then:
let level = match verbose {
    0 => "info",
    1 => "debug",
    _ => "trace",
};
```

## Testing Logging Code

Use serial_test for environment-variable tests:

```rust
#[test]
#[serial(bitnet_env)]  // Ensures serial execution
fn test_with_env() {
    let _guard = EnvGuard::new("BITNET_DEBUG_CUSTOM", "1");
    // Test code here - env automatically restored on drop
}
```

## Best Practices Summary

### DO:
- Use `tracing::*!()` for diagnostic output
- Use `println!()` + `style()` only for banners
- Write logs to stderr, output to stdout
- Use `warn_once!()` for repeated warnings
- Include 3+ actionable solutions in error messages
- Guard debug output with environment variables

### DON'T:
- Use `println!()` for diagnostic output (use tracing)
- Call `eprintln!()` except for banners and debug
- Mix stdout/stderr for the same logical output
- Log sensitive information (keys, private paths)
- Ignore error context

## Questions or Issues?

Refer to the specific sections in the comprehensive guide:
- **How do I add logging?** → Section 1 (Logging Infrastructure)
- **How do I prevent log spam?** → Section 2 (Warn-Once Pattern)
- **How do I make output look nice?** → Section 3 (Banner Output)
- **How do I debug timing issues?** → Section 4 (Debug Output Patterns)
- **How do I compare token IDs?** → Section 5 (Token ID Debug)
- **How do I validate greedy decoding?** → Section 6 (Logit Dumping)
- **How do I report errors well?** → Section 7 (Error Reporting)
- **How do I write a diagnostic command?** → Section 8 (Diagnostic Commands)
- **How do I add --verbose support?** → Section 9 (Verbose Mode)
- **How do I pick exit codes?** → Section 7.1 (Exit Codes)

---

**Last Updated**: October 25, 2025
**Coverage**: Complete logging, diagnostic, and error reporting infrastructure
**Status**: Ready for production use
