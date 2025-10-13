# BitNet.rs Development Workflows
# Run 'just' to see available commands

# Default recipe - show available commands
default:
    @just --list

# Build with CPU support (default)
cpu:
    cargo build --workspace --no-default-features --features cpu
    cargo test --workspace --no-default-features --features cpu --exclude bitnet-py

# Build with GPU/CUDA support
cuda:
    cargo build --workspace --no-default-features --features cuda
    cargo test -p bitnet-kernels --no-default-features --features cuda
    cargo run -p bitnet-kernels --example cuda_smoke --no-default-features --features cuda

# Alias for backward compatibility
gpu: cuda

# Build with FFI support (CPU)
ffi TAG="main":
    cargo run -p xtask -- fetch-cpp --tag {{TAG}}
    cargo build --workspace --no-default-features --features "cpu,ffi"
    cargo test -p bitnet-crossval --features "crossval,ffi"

# Build with FFI support (CUDA)
ffi-cuda TAG="main" ARCHS="80;86":
    cargo run -p xtask -- fetch-cpp --tag {{TAG}} --backend cuda --cmake-flags "-DCMAKE_CUDA_ARCHITECTURES={{ARCHS}}"
    cargo build --workspace --no-default-features --features "cuda,ffi"
    cargo test -p bitnet-crossval --features "crossval,ffi"

# Run full cross-validation suite (CPU)
crossval TAG="main":
    cargo run -p xtask -- full-crossval --tag {{TAG}}

# Run full cross-validation suite (CUDA)
crossval-cuda TAG="main" ARCHS="80;86":
    cargo run -p xtask -- full-crossval --tag {{TAG}} --backend cuda --cmake-flags "-DCMAKE_CUDA_ARCHITECTURES={{ARCHS}}"

# Compare metrics for regression detection
compare-metrics BASELINE="baselines/cpu-main.json" CURRENT="crossval/results/last_run.json":
    cargo run -p xtask -- compare-metrics --baseline {{BASELINE}} --current {{CURRENT}}

# Check for API breaking changes
check-breaking BASELINE="origin/main":
    cargo run -p xtask -- detect-breaking --baseline {{BASELINE}} --current .

# Run API snapshot tests
test-api-snapshots:
    cargo test --test api_snapshots
    @echo "✅ API snapshots verified"

# Update API snapshots (use when intentionally changing API)
update-api-snapshots:
    cargo test --test api_snapshots -- --accept
    @echo "📸 API snapshots updated - remember to commit the changes!"

# Update API baselines (when making intentional API changes)
update-api-baselines:
    @echo "Updating Rust API baselines..."
    cargo public-api -p bitnet-common > api/rust/bitnet-common.public-api.txt || true
    cargo public-api -p bitnet-kernels > api/rust/bitnet-kernels.public-api.txt || true
    cargo public-api -p bitnet-inference > api/rust/bitnet-inference.public-api.txt || true
    cargo public-api -p bitnet-ffi > api/rust/bitnet-ffi.public-api.txt || true
    cargo public-api -p bitnet-cli > api/rust/bitnet-cli.public-api.txt || true
    @echo "Updating FFI header..."
    cbindgen crates/bitnet-ffi --config api/ffi/cbindgen.toml -o api/ffi/bitnet_ffi.h || true
    @echo "Updating FFI symbols..."
    cargo build -p bitnet-ffi --release --no-default-features --features ffi
    nm -D --defined-only target/release/libbitnet_ffi.so | awk '{print $3}' | sort > api/ffi/ffi.symbols.txt || true
    @echo "Updating CLI help..."
    cargo run -p bitnet-cli -- --help > api/cli/help.txt || true
    @echo "📝 API baselines updated - review changes and update API_CHANGES.md!"

# Full API compatibility check
api-check: check-breaking test-api-snapshots
    @echo "✅ API compatibility verified"

# Verify API baselines match current code
verify-api-baselines:
    @echo "Verifying API baselines..."
    @for crate in bitnet-common bitnet-kernels bitnet-inference bitnet-ffi bitnet-cli; do \
        echo "Checking $$crate..."; \
        cargo public-api -p $$crate > /tmp/$$crate.current.txt || true; \
        diff -q api/rust/$$crate.public-api.txt /tmp/$$crate.current.txt || echo "⚠️  $$crate baseline differs"; \
    done
    @echo "✅ API baseline verification complete"

# Format all code
fmt:
    cargo fmt --all

# Run clippy lints
lint:
    cargo clippy --workspace --no-default-features --features cpu --exclude bitnet-py

# Run all quality checks
quality: fmt lint
    cargo test --workspace --no-default-features --features cpu --exclude bitnet-py

# Clean all build artifacts and caches
clean:
    cargo run -p xtask -- clean-cache
    cargo clean

# Download model for testing
download-model:
    cargo run -p xtask -- download-model

# Quick test of critical functionality
test-quick:
    cargo test -p bitnet-kernels --no-default-features --features cpu --lib
    cargo test -p bitnet-common --lib
    cargo test -p bitnet-quantization --lib

# Full test suite
test-all:
    cargo test --workspace --no-default-features --features cpu --exclude bitnet-py
    cargo test --workspace --no-default-features --features "cpu,ffi" --exclude bitnet-py

# Build release binaries
release:
    cargo build --release --no-default-features --features cpu

# Generate code coverage
coverage:
    cargo llvm-cov --workspace --features cpu --html

# Run benchmarks
bench:
    cargo bench --workspace --no-default-features --features cpu

# Check for security vulnerabilities
audit:
    cargo audit

# Check for outdated dependencies
outdated:
    cargo outdated

# CI simulation - run all checks locally (CPU)
ci-cpu:
    cargo fmt --all -- --check
    cargo clippy --workspace --no-default-features --features cpu --exclude bitnet-py -- -D warnings
    cargo build --workspace --no-default-features --features cpu
    cargo test --workspace --no-default-features --features cpu --lib --exclude bitnet-py
    @echo "✅ All CPU CI checks passed!"

# CI simulation - run all checks locally (CUDA)
ci-cuda TAG="main" ARCHS="80;86":
    cargo run -p xtask -- fetch-cpp --tag {{TAG}} --backend cuda --cmake-flags "-DCMAKE_CUDA_ARCHITECTURES={{ARCHS}}"
    cargo build --workspace --no-default-features --features "cuda,ffi"
    cargo test -p bitnet-kernels --features cuda --tests
    cargo run -p xtask -- full-crossval --tag {{TAG}} --backend cuda --cmake-flags "-DCMAKE_CUDA_ARCHITECTURES={{ARCHS}}"
    @echo "✅ All CUDA CI checks passed!"

# Quick CI check (default)
ci: ci-cpu

# Run comprehensive verification test suite
verify:
    bash scripts/verify-tests.sh

# ============================================================================
# Model Operations
# ============================================================================

# Export a clean F16 GGUF model (LayerNorm preserved in float)
model-export model_dir tokenizer out_dir="models/clean":
    @echo "Exporting clean GGUF from {{model_dir}}..."
    ./scripts/export_clean_gguf.sh "{{model_dir}}" "{{tokenizer}}" "{{out_dir}}"

# Validate a GGUF model (strict mode, no policy corrections)
model-validate gguf tokenizer:
    @echo "Validating GGUF: {{gguf}}..."
    ./scripts/validate_gguf.sh "{{gguf}}" "{{tokenizer}}"

# Export and validate in one command
model-clean model_dir tokenizer out_dir="models/clean": (model-export model_dir tokenizer out_dir)
    @echo "Validating exported model..."
    just model-validate "{{out_dir}}/clean-f16.gguf" "{{tokenizer}}"

# Quantize F16 GGUF to I2_S (LayerNorm excluded)
model-quantize-i2s f16_gguf out_dir:
    @echo "Quantizing {{f16_gguf}} to I2_S..."
    ./scripts/quantize_i2s_clean.sh "{{f16_gguf}}" "{{out_dir}}"

# Inspect LayerNorm statistics for a GGUF model
model-inspect-ln gguf:
    @echo "Inspecting LayerNorm statistics..."
    cargo run -q -p bitnet-cli --no-default-features --features cpu -- \
        inspect --ln-stats "{{gguf}}"

# Run inference probe on a model (deterministic, greedy)
model-probe gguf tokenizer prompt="The capital of France is" max_tokens="8":
    @echo "Running inference probe..."
    BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 \
    cargo run -q -p bitnet-cli --no-default-features --features cpu -- \
        run --model "{{gguf}}" --tokenizer "{{tokenizer}}" \
        --prompt "{{prompt}}" --max-new-tokens "{{max_tokens}}" --temperature 0.0

# Generate baseline report for a model
model-baseline gguf tokenizer out_file="docs/baselines/model-baseline.md":
    @echo "Generating baseline report..."
    @echo "# Model Baseline Report" > "{{out_file}}"
    @echo "" >> "{{out_file}}"
    @echo "**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "{{out_file}}"
    @echo "**Model:** {{gguf}}" >> "{{out_file}}"
    @echo "**Tokenizer:** {{tokenizer}}" >> "{{out_file}}"
    @echo "" >> "{{out_file}}"
    @echo "## Fingerprint" >> "{{out_file}}"
    @echo "" >> "{{out_file}}"
    @if [ -f "{{gguf}}.fingerprint" ] || [ -f "$(dirname {{gguf}})/$(basename {{gguf}} .gguf).fingerprint" ]; then \
        echo '```' >> "{{out_file}}"; \
        cat "{{gguf}}.fingerprint" 2>/dev/null || cat "$(dirname {{gguf}})/$(basename {{gguf}} .gguf).fingerprint" >> "{{out_file}}"; \
        echo '```' >> "{{out_file}}"; \
    else \
        echo "Fingerprint not found - run model-export to generate" >> "{{out_file}}"; \
    fi
    @echo "" >> "{{out_file}}"
    @echo "## LayerNorm Statistics" >> "{{out_file}}"
    @echo "" >> "{{out_file}}"
    @echo '```' >> "{{out_file}}"
    @just model-inspect-ln "{{gguf}}" >> "{{out_file}}" 2>&1 || echo "Failed to inspect LN" >> "{{out_file}}"
    @echo '```' >> "{{out_file}}"
    @echo "" >> "{{out_file}}"
    @echo "## Probe Outputs" >> "{{out_file}}"
    @echo "" >> "{{out_file}}"
    @for prompt in "The capital of France is" "Once upon a time" "def factorial(n):"; do \
        echo "### Prompt: $$prompt" >> "{{out_file}}"; \
        echo "" >> "{{out_file}}"; \
        echo '```' >> "{{out_file}}"; \
        just model-probe "{{gguf}}" "{{tokenizer}}" "$$prompt" 8 >> "{{out_file}}" 2>&1 || echo "Failed" >> "{{out_file}}"; \
        echo '```' >> "{{out_file}}"; \
        echo "" >> "{{out_file}}"; \
    done
    @echo "Baseline report written to: {{out_file}}"

# Quick model check: export, validate, and generate baseline
model-check model_dir tokenizer:
    @echo "Running full model check pipeline..."
    just model-clean "{{model_dir}}" "{{tokenizer}}"
    just model-baseline "models/clean/clean-f16.gguf" "{{tokenizer}}"

# ============================================================================
# SafeTensors to GGUF Converter (st2gguf)
# ============================================================================

# Convert SafeTensors to clean F16 GGUF using Rust converter
st2gguf-convert input output config="" tokenizer="" strict="":
    #!/usr/bin/env bash
    set -euo pipefail
    args="--input {{input}} --output {{output}}"
    [[ -n "{{config}}" ]] && args="$args --config {{config}}"
    [[ -n "{{tokenizer}}" ]] && args="$args --tokenizer {{tokenizer}}"
    [[ -n "{{strict}}" ]] && args="$args --strict"
    cargo run -q --release -p bitnet-st2gguf -- $args

# Build st2gguf binary in release mode
st2gguf-build:
    @echo "Building st2gguf converter..."
    cargo build --release -p bitnet-st2gguf
    @echo "✅ Built: target/release/st2gguf"

# Run st2gguf tests
st2gguf-test:
    @echo "Running st2gguf tests..."
    cargo test --release -p bitnet-st2gguf
    @echo "✅ All st2gguf tests passed"

# Install st2gguf binary to cargo bin
st2gguf-install:
    @echo "Installing st2gguf to cargo bin..."
    cargo install --path crates/bitnet-st2gguf --force
    @echo "✅ Installed: st2gguf (run with 'st2gguf --help')"

# ============================================================================
# SafeTensors Tools (bitnet-st-tools)
# ============================================================================

# Merge SafeTensors shards with LayerNorm preservation (F16)
st-merge-ln-f16 input output:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔗 Merging SafeTensors shards with LayerNorm preservation..."
    cargo run --release -p bitnet-st-tools --bin st-merge-ln-f16 -- \
        "{{input}}" "{{output}}"

# Inspect LayerNorm tensors in SafeTensors file
st-ln-inspect safetensors_file:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔍 Inspecting LayerNorm tensors in SafeTensors..."
    cargo run --release -p bitnet-st-tools --bin st-ln-inspect -- \
        "{{safetensors_file}}"

# Build st-tools binaries in release mode
st-tools-build:
    @echo "Building st-tools..."
    cargo build --release -p bitnet-st-tools
    @echo "✅ Built: target/release/st-ln-inspect"
    @echo "✅ Built: target/release/st-merge-ln-f16"

# ============================================================================
# Advanced Inspection (with policy and gate control)
# ============================================================================

# Run inspect with auto gate detection (architecture-aware rules)
inspect-auto model_gguf:
    #!/usr/bin/env bash
    set -euo pipefail
    # Resolve model path
    MODEL="{{model_gguf}}"
    if [[ ! -f "$MODEL" ]]; then
        if [[ -f "models/{{model_gguf}}" ]]; then
            MODEL="models/{{model_gguf}}"
        fi
    fi
    cargo run --release -p bitnet-cli --no-default-features --features cpu,full-cli -- \
        inspect --ln-stats --gate auto "$MODEL"

# Run inspect with custom policy file
inspect-policy model_gguf policy_yml policy_key:
    #!/usr/bin/env bash
    set -euo pipefail
    # Resolve model path
    MODEL="{{model_gguf}}"
    if [[ ! -f "$MODEL" ]]; then
        if [[ -f "models/{{model_gguf}}" ]]; then
            MODEL="models/{{model_gguf}}"
        fi
    fi
    cargo run --release -p bitnet-cli --no-default-features --features cpu,full-cli -- \
        inspect --ln-stats --gate policy \
        --policy "{{policy_yml}}" \
        --policy-key "{{policy_key}}" \
        "$MODEL"

# Quick validation gate check with auto ruleset (strict mode)
gates-auto model_gguf:
    #!/usr/bin/env bash
    set -euo pipefail
    # Resolve model path
    MODEL="{{model_gguf}}"
    if [[ ! -f "$MODEL" ]]; then
        if [[ -f "models/{{model_gguf}}" ]]; then
            MODEL="models/{{model_gguf}}"
        fi
    fi
    BITNET_STRICT_MODE=1 \
    ./target/release/bitnet inspect --ln-stats --gate auto "$MODEL"

# Quick validation gate check with policy file (strict mode)
gates-policy model_gguf policy_yml policy_key:
    #!/usr/bin/env bash
    set -euo pipefail
    # Resolve model path
    MODEL="{{model_gguf}}"
    if [[ ! -f "$MODEL" ]]; then
        if [[ -f "models/{{model_gguf}}" ]]; then
            MODEL="models/{{model_gguf}}"
        fi
    fi
    BITNET_STRICT_MODE=1 \
    ./target/release/bitnet inspect --ln-stats \
        --gate policy --policy "{{policy_yml}}" --policy-key "{{policy_key}}" "$MODEL"

# ============================================================================
# Build Recipes
# ============================================================================

# Build all components (CLI, st2gguf, st-tools)
build-all: build-cli st2gguf-build st-tools-build
    @echo "✅ All components built"

# Build bitnet-cli with full features (CPU inference + all subcommands)
build-cli:
    @echo "🔨 Building bitnet-cli with full features..."
    cargo build --release -p bitnet-cli --no-default-features --features cpu,full-cli
    @echo "✅ Built: target/release/bitnet"

# Build bitnet-cli with GPU support
build-cli-gpu:
    @echo "🔨 Building bitnet-cli with GPU support..."
    cargo build --release -p bitnet-cli --no-default-features --features gpu,full-cli
    @echo "✅ Built: target/release/bitnet (GPU)"

# ============================================================================
# Utility Recipes
# ============================================================================

# Show bitnet-cli version and features
version:
    cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- --version

# Show system information
info:
    cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- info

# Check GGUF file compatibility
compat-check model_gguf:
    #!/usr/bin/env bash
    set -euo pipefail
    # Resolve model path
    MODEL="{{model_gguf}}"
    if [[ ! -f "$MODEL" ]]; then
        if [[ -f "models/{{model_gguf}}" ]]; then
            MODEL="models/{{model_gguf}}"
        fi
    fi
    cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
        compat-check "$MODEL" --show-kv --kv-limit 20