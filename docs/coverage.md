# Coverage Collection and Reporting

This document describes the coverage system for BitNet-rs, using
[cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) for
source-based code coverage via LLVM instrumentation.

## Quick Start

### Install

```bash
cargo install cargo-llvm-cov --locked
```

### Collect Coverage

```bash
# JSON report (machine-readable)
cargo cov

# HTML report (interactive, line-by-line)
cargo cov-html

# Open the HTML report
open target/llvm-cov/html/index.html
```

### Cargo Aliases

Defined in `.cargo/config.toml`:

| Alias | Command | Output |
|-------|---------|--------|
| `cargo cov` | `cargo llvm-cov ... --json --output-path coverage.json` | `coverage.json` |
| `cargo cov-all` | `cargo llvm-cov ... --html --output-dir target/llvm-cov/html` | HTML report |
| `cargo cov-html` | `cargo llvm-cov ... --html --output-dir target/llvm-cov/html` | HTML report |

All aliases use `--workspace --no-default-features --features cpu`.

## CI Workflow

The coverage workflow (`.github/workflows/coverage.yml`) runs a single
instrumented build and generates HTML, JSON, and text reports.

### Triggers

| Trigger | Behavior |
|---------|----------|
| Push to `main` | Full run with 70% threshold enforcement |
| PR with `coverage` label | Coverage collected, no threshold gate |
| Manual dispatch | Full run with 70% threshold enforcement |

### Threshold

The CI enforces a **70%** line-coverage minimum on `main` and manual
dispatch runs. PRs with the `coverage` label collect coverage without
enforcing the threshold.

### Artifacts

A single `coverage-report` artifact is uploaded on every run containing:

- `coverage.json` — machine-readable JSON report
- `coverage.txt` — text summary
- `target/llvm-cov/html/**` — interactive HTML report

Retention: **7 days**.

## Output Paths

| File | Description |
|------|-------------|
| `coverage.json` | JSON report (workspace root) |
| `coverage.txt` | Text summary (workspace root) |
| `target/llvm-cov/html/index.html` | Interactive HTML report |

## Platform Support

`cargo-llvm-cov` uses LLVM instrumentation (not ptrace), so it works on
all platforms:

- **Linux** — full support
- **macOS** — full support
- **Windows** — full support

## Resources

- [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) — the
  coverage tool used by this project
- [LLVM Source-Based Code Coverage](https://llvm.org/docs/CoverageMapping.html)
