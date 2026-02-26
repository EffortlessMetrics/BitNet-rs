# Nix Flake Usage Guide

## Overview

BitNet-rs uses **Nix as the canonical build and development spine**. The flake provides:

- ✅ **Reproducible dev environments** (pinned Rust + all deps)
- ✅ **Buildable packages** (`bitnet-server`, `bitnet-cli`, `bitnet-st2gguf`)
- ✅ **Runnable apps** (`nix run .#bitnet-cli`)
- ✅ **Local CI validation** (hermetic, identical to future CI)

This ensures toolchain consistency across machines and provides one-command workflows for development, building, and validation.

---

## Quick Start

### Build & Run Binaries

```bash
# Build packages (reproducible, pinned toolchain)
nix build .#bitnet-server
nix build .#bitnet-cli
nix build .#bitnet-st2gguf

# Run directly without building
nix run .#bitnet-cli -- --help
nix run .#bitnet-server -- --help
nix run .#bitnet-st2gguf -- --help

# Default package (bitnet-server)
nix build
./result/bin/bitnet-server --version
```

### Enter Development Shell

```bash
# Standard dev shell (stable Rust + all dependencies)
nix develop

# MSRV debugging shell (Rust 1.89.0)
nix develop .#msrv
```

### Run CI Checks

```bash
# Run all checks (workspace + receipts validation)
nix flake check

# Run specific check
nix flake check .#workspace
nix flake check .#bitnet-server-receipts

# Or via nix build
nix build .#checks.workspace
nix build .#checks.bitnet-server-receipts
```

---

## What's Included

### Packages (Buildable Binaries)

#### `packages.bitnet-server` (default)
**Production-ready inference server**
- Features: `cpu` (CPU-only for reproducibility)
- Binary: `result/bin/bitnet-server`

#### `packages.bitnet-cli`
**CLI for model inspection and inference**
- Features: `cpu,full-cli` (all subcommands)
- Binary: `result/bin/bitnet-cli`

#### `packages.bitnet-st2gguf`
**SafeTensors → GGUF converter**
- Binary: `result/bin/bitnet-st2gguf`

**Usage:**
```bash
# Build and run
nix build .#bitnet-cli
./result/bin/bitnet-cli --help

# Run directly (no local artifact)
nix run .#bitnet-cli -- --help
nix run .#bitnet-server -- --version
```

### Apps (Direct Execution)

Convenient wrappers for `nix run`:
- `apps.bitnet-cli` - Run CLI directly
- `apps.bitnet-server` - Run server directly
- `apps.bitnet-st2gguf` - Run converter directly

### Development Shells

#### `devShells.default`
**Standard development environment** with:
- ✅ Rust stable (latest) + 1.89.0 MSRV
- ✅ Native dependencies (openssl, pkg-config, cmake, etc.)
- ✅ C/C++ toolchain (clang, libclang for bindgen)
- ✅ Utilities (git, jq, python3)

**Environment variables:**
- `RUSTUP_TOOLCHAIN=stable`
- `RUSTC_WRAPPER=""` (disables sccache)
- `CARGO_INCREMENTAL=0` (faster clean builds)
- `LIBCLANG_PATH` (set for bindgen)

#### `devShells.msrv`
**MSRV-only shell** (1.89.0) for compatibility testing.

### CI Checks

#### `checks.workspace`
**Full workspace validation** (mirrors `./scripts/ci-local.sh workspace`):
1. Clean build
2. Build & test (strict `-D warnings`)
3. Clippy (strict)
4. Format check
5. Documentation
6. MSRV check (1.89.0)

#### `checks.bitnet-server-receipts`
**Focused receipts validation** (mirrors `./scripts/ci-local.sh bitnet-server-receipts`):
1. Baseline CPU check
2. Clippy (CPU only)
3. Format check
4. Documentation
5. MSRV (1.89.0)
6. Feature combo: `cpu,receipts`
7. Feature combo: `cpu,receipts,tuning`
8. Test: receipts happy path
9. Test: receipts guard path

**Log artifacts:** `result/ci-receipts.log`

#### `checks.nextest`
**Fast test runner with timeout protection**:
- Uses `cargo-nextest` with CI profile (`.config/nextest.toml`)
- 5-minute global timeout prevents test hangs
- CPU features only (`--no-default-features --features cpu`)
- JUnit XML report generation (`target/nextest/ci/junit.xml`)
- Faster than `cargo test` with per-test isolation

**Log artifacts:** `result/ci-nextest.log`, `result/junit.xml`

**Why nextest?**
- ✅ Timeout protection (prevents infinite loops in tests)
- ✅ Clean output (success-output = "never")
- ✅ No retries (retries = 0 ensures consistent passes)
- ✅ JUnit reports (for future CI integration)

---

## Typical Workflows

### Development Workflow (Nix-First)

```bash
# Enter dev shell (pinned toolchain + deps)
nix develop

# Work on code...
cargo build --no-default-features --features cpu
cargo test -p bitnet-server --no-default-features --features cpu,receipts

# Quick local validation (2-4 min)
./scripts/ci-local.sh bitnet-server-receipts

# Or run hermetic checks (slower, reproducible)
nix flake check .#bitnet-server-receipts
```

### Building Artifacts (Nix-First)

```bash
# Build for distribution (reproducible)
nix build .#bitnet-server
nix build .#bitnet-cli
nix build .#bitnet-st2gguf

# Copy artifacts elsewhere
cp -L result/bin/bitnet-server /path/to/deploy/

# Or run directly without local artifact
nix run .#bitnet-cli -- run --model model.gguf --prompt "Test"
```

### Pre-PR Validation

```bash
# Run all checks via Nix (guaranteed reproducible)
nix flake check

# Or specific check
nix flake check .#bitnet-server-receipts
nix flake check .#workspace
```

### CI Replacement (GitHub Actions Offline)

```bash
# One-command full validation (hermetic)
nix flake check

# Or use the scripts directly in dev shell (faster iteration)
nix develop
./scripts/ci-local.sh workspace
./scripts/ci-local.sh bitnet-server-receipts
```

---

## Benefits of the Flake

### 1. Reproducibility
- ✅ Pinned Rust toolchains (stable + MSRV)
- ✅ Pinned native dependencies (nixpkgs-24.05)
- ✅ Same environment across machines

### 2. CI/Local Parity
- ✅ Same checks run locally and in CI
- ✅ No "works on my machine" issues
- ✅ Easy to add new checks

### 3. Onboarding
- ✅ New contributors run `nix develop` → ready to work
- ✅ No manual toolchain setup
- ✅ All dependencies included

### 4. Extensibility
- ✅ Add new crate-specific checks to `flake.nix`
- ✅ Wire them to `scripts/ci-local.sh` modes
- ✅ Automatic dependency management

---

## Advanced Usage

### Custom Checks

To add a new check (e.g., for `bitnet-cli`):

1. Add mode to `scripts/ci-local.sh`:
   ```bash
   elif [[ "${MODE}" == "bitnet-cli-validation" ]]; then
     # validation steps...
   fi
   ```

2. Add check to `flake.nix`:
   ```nix
   checks.bitnet-cli =
     pkgs.stdenv.mkDerivation {
       name = "ci-bitnet-cli";
       src = ./.;
       nativeBuildInputs = [ rustStable rustMsrv ] ++ nativeDeps;

       buildPhase = ''
         export HOME=$(mktemp -d)
         export RUSTUP_TOOLCHAIN=stable
         export RUSTC_WRAPPER=""
         export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"

         chmod +x scripts/ci-local.sh
         ./scripts/ci-local.sh bitnet-cli-validation
       '';

       installPhase = ''
         mkdir -p $out
         echo "bitnet-cli checks passed" > $out/result
       '';
     };
   ```

3. Run: `nix flake check .#bitnet-cli`

### Updating Pinned Inputs

```bash
# Update all inputs (nixpkgs, rust-overlay)
nix flake update

# Update specific input
nix flake lock --update-input nixpkgs
nix flake lock --update-input rust-overlay

# Check what changed
git diff flake.lock
```

### Formatting the Flake

```bash
# Format flake.nix with nixpkgs-fmt
nix fmt
```

---

## Troubleshooting

### "unknown setting 'lazy-trees'"
- **Cause:** Warning from newer Nix experimental features
- **Solution:** Ignore (harmless) or update `nix.conf`

### "Git tree is dirty"
- **Cause:** Uncommitted changes in repo
- **Solution:** Commit or stash changes, or use `--impure` flag:
  ```bash
  nix develop --impure
  nix flake check --impure
  ```

### Build fails in Nix sandbox
- **Cause:** Check tries to access network or user home
- **Solution:** Ensure checks are hermetic (no network, no `~/.cargo`)
- **Workaround:** Run in dev shell instead:
  ```bash
  nix develop
  ./scripts/ci-local.sh bitnet-server-receipts
  ```

### MSRV check fails
- **Cause:** Code uses features from Rust > 1.89.0
- **Solution:** Fix code or update MSRV in `Cargo.toml` and `flake.nix`

---

## Nix as the Primary Workflow

The flake is **the canonical build and development spine**:

| Command | Use Case | Speed | Reproducibility |
|---------|----------|-------|----------------|
| `nix build .#bitnet-server` | Build production artifact | Slow (first), fast (cached) | ✅ Guaranteed |
| `nix run .#bitnet-cli` | Run CLI directly | Fast (cached) | ✅ Guaranteed |
| `nix develop` | Enter dev environment | Instant | ✅ Pinned deps |
| `nix flake check` | Full validation (hermetic) | Slow (~4-8 min) | ✅ Guaranteed |
| `./scripts/ci-local.sh <mode>` | Quick local validation | Fast (2-4 min) | ⚠️ Depends on local env |
| `cargo build/test` | Fast iteration in dev shell | Fastest | ⚠️ Depends on local env |

**Recommended workflow:**
1. **Setup:** `nix develop` → reproducible environment
2. **Development:** Use `cargo` directly for fast iteration
3. **Local validation:** `./scripts/ci-local.sh <mode>` for quick feedback
4. **Pre-PR:** `nix flake check` for reproducibility guarantee
5. **Distribution:** `nix build .#<package>` for reproducible artifacts

**Key principle:** Nix provides the **source of truth** for builds and validation. Scripts and cargo are **convenience layers** for iteration speed.

---

## See Also

- `docs/kv-pool/LOCAL_VALIDATION_WORKFLOW.md` - Detailed local CI guide
- `scripts/ci-local.sh` - Local CI script implementation
- `flake.nix` - Flake configuration
- `flake.lock` - Pinned dependency versions
