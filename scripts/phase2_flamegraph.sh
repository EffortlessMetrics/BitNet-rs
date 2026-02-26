#!/usr/bin/env bash
set -euo pipefail

# Enforce determinism for reproducible flamegraphs
export BITNET_DETERMINISTIC=1
export RAYON_NUM_THREADS=1

# =============================================================================
# BitNet-rs Phase 2 Flamegraph Generation Script
# =============================================================================
# Purpose: Generate CPU flamegraphs for performance profiling and hotspot analysis
#
# This script:
# 1. Checks for flamegraph tools (cargo-flamegraph or samply)
# 2. Builds optimized release binary with debug symbols
# 3. Generates flamegraphs for 1-token and 10-token inference runs
# 4. Creates metadata markdown with system fingerprint and top hotspots
# 5. Outputs SVGs to docs/baselines/perf/flamegraphs/
#
# Usage:
#   ./scripts/phase2_flamegraph.sh [MODEL] [TOKENIZER] [OUTPUT_DIR]
#
# Arguments:
#   MODEL       - Path to GGUF model (default: auto-discover from models/)
#   TOKENIZER   - Path to tokenizer.json (default: auto-discover from models/)
#   OUTPUT_DIR  - Output directory (default: docs/baselines/perf/flamegraphs)
#
# Environment Variables:
#   BITNET_FLAMEGRAPH_TOOL - Force specific tool: 'cargo-flamegraph' or 'samply'
#   BITNET_SKIP_BUILD      - Skip build step if binary already exists
#   BITNET_DETERMINISTIC   - Enable deterministic inference (default: 1)
#   BITNET_SEED            - Random seed for deterministic runs (default: 42)
#
# Examples:
#   # Basic usage (auto-discovers model)
#   ./scripts/phase2_flamegraph.sh
#
#   # Custom model and tokenizer
#   ./scripts/phase2_flamegraph.sh models/custom.gguf models/tokenizer.json
#
#   # Force samply instead of cargo-flamegraph
#   BITNET_FLAMEGRAPH_TOOL=samply ./scripts/phase2_flamegraph.sh
#
# Requirements:
#   - cargo-flamegraph (preferred) OR samply
#   - perf (Linux) or DTrace (macOS)
#   - Elevated privileges for perf (Linux only)
#
# Output:
#   - docs/baselines/perf/flamegraphs/phase2_1tok.svg
#   - docs/baselines/perf/flamegraphs/phase2_1tok.md
#   - docs/baselines/perf/flamegraphs/phase2_10tok.svg
#   - docs/baselines/perf/flamegraphs/phase2_10tok.md
#   - docs/baselines/perf/flamegraphs/README.md
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL="${1:-}"
TOKENIZER="${2:-}"
OUTPUT_DIR="${3:-docs/baselines/perf/flamegraphs}"
FLAMEGRAPH_TOOL="${BITNET_FLAMEGRAPH_TOOL:-auto}"
SKIP_BUILD="${BITNET_SKIP_BUILD:-0}"
DETERMINISTIC="${BITNET_DETERMINISTIC:-1}"
SEED="${BITNET_SEED:-42}"

# Timestamps and metadata
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

show_usage() {
    cat << 'EOF'
Usage: ./scripts/phase2_flamegraph.sh [MODEL] [TOKENIZER] [OUTPUT_DIR]

Generate CPU flamegraphs for BitNet-rs inference profiling.

Arguments:
  MODEL       - Path to GGUF model (default: auto-discover)
  TOKENIZER   - Path to tokenizer.json (default: auto-discover)
  OUTPUT_DIR  - Output directory (default: docs/baselines/perf/flamegraphs)

Environment Variables:
  BITNET_FLAMEGRAPH_TOOL - Force 'cargo-flamegraph' or 'samply'
  BITNET_SKIP_BUILD      - Skip build if binary exists (0/1)
  BITNET_DETERMINISTIC   - Enable deterministic inference (default: 1)
  BITNET_SEED            - Random seed (default: 42)

Examples:
  # Auto-discover model and generate flamegraphs
  ./scripts/phase2_flamegraph.sh

  # Custom model and tokenizer
  ./scripts/phase2_flamegraph.sh models/custom.gguf models/tokenizer.json

  # Use samply instead of cargo-flamegraph
  BITNET_FLAMEGRAPH_TOOL=samply ./scripts/phase2_flamegraph.sh

Requirements:
  - cargo-flamegraph OR samply
  - perf (Linux) or DTrace (macOS)
  - sudo/perf_event_paranoid access (Linux)

For more info, see: docs/performance-tuning.md#profiling-and-monitoring
EOF
}

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    show_usage
    exit 0
fi

log_info() {
    echo "[INFO] $*" >&2
}

log_warn() {
    echo "[WARN] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

# Auto-discover model and tokenizer if not provided
auto_discover_files() {
    if [[ -z "$MODEL" ]] || [[ -z "$TOKENIZER" ]]; then
        log_info "Auto-discovering model and tokenizer..."

        # Search common locations
        local model_dirs=(
            "models/microsoft-bitnet-b1.58-2B-4T-gguf"
            "models"
            "."
        )

        for dir in "${model_dirs[@]}"; do
            if [[ -z "$MODEL" ]] && [[ -f "$dir/ggml-model-i2_s.gguf" ]]; then
                MODEL="$dir/ggml-model-i2_s.gguf"
                log_info "Found model: $MODEL"
            fi
            if [[ -z "$TOKENIZER" ]] && [[ -f "$dir/tokenizer.json" ]]; then
                TOKENIZER="$dir/tokenizer.json"
                log_info "Found tokenizer: $TOKENIZER"
            fi

            if [[ -n "$MODEL" ]] && [[ -n "$TOKENIZER" ]]; then
                break
            fi
        done

        if [[ -z "$MODEL" ]]; then
            log_error "Could not auto-discover model. Please provide MODEL argument."
            log_error "Hint: Download model with: cargo run -p xtask -- download-model"
            exit 1
        fi

        if [[ -z "$TOKENIZER" ]]; then
            log_error "Could not auto-discover tokenizer. Please provide TOKENIZER argument."
            exit 1
        fi
    fi

    # Validate files exist
    if [[ ! -f "$MODEL" ]]; then
        log_error "Model not found: $MODEL"
        exit 1
    fi

    if [[ ! -f "$TOKENIZER" ]]; then
        log_error "Tokenizer not found: $TOKENIZER"
        exit 1
    fi
}

# Detect available flamegraph tool
detect_flamegraph_tool() {
    if [[ "$FLAMEGRAPH_TOOL" != "auto" ]]; then
        log_info "Using forced flamegraph tool: $FLAMEGRAPH_TOOL"
        return 0
    fi

    log_info "Detecting available flamegraph tools..."

    if command -v flamegraph &> /dev/null; then
        FLAMEGRAPH_TOOL="cargo-flamegraph"
        log_info "Found cargo-flamegraph"
        return 0
    fi

    if command -v samply &> /dev/null; then
        FLAMEGRAPH_TOOL="samply"
        log_info "Found samply"
        return 0
    fi

    log_warn "No flamegraph tool found. Attempting to install cargo-flamegraph..."

    if cargo install --list | grep -q "flamegraph"; then
        log_info "cargo-flamegraph already installed"
        FLAMEGRAPH_TOOL="cargo-flamegraph"
        return 0
    fi

    log_info "Installing cargo-flamegraph..."
    if cargo install --locked flamegraph; then
        FLAMEGRAPH_TOOL="cargo-flamegraph"
        log_info "Successfully installed cargo-flamegraph"
        return 0
    else
        log_error "Failed to install cargo-flamegraph"
        log_error "Please install manually: cargo install --locked flamegraph"
        log_error "Or install samply: cargo install samply"
        exit 1
    fi
}

# Check for perf/dtrace availability
check_profiling_capability() {
    local os_type=$(uname -s)

    case "$os_type" in
        Linux)
            if ! command -v perf &> /dev/null; then
                log_error "perf not found. Please install: sudo apt install linux-tools-generic"
                exit 1
            fi

            # Check perf_event_paranoid setting
            if [[ -f /proc/sys/kernel/perf_event_paranoid ]]; then
                local paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
                if [[ "$paranoid" -gt 1 ]]; then
                    log_warn "perf_event_paranoid is $paranoid (>1), may need elevated privileges"
                    log_warn "To fix: echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
                    log_warn "Or run this script with sudo"
                fi
            fi

            # Test perf record capability
            if ! perf record -o /tmp/bitnet-perf-test.$$ -- true 2>/dev/null; then
                log_error "Cannot run perf record. You may need elevated privileges."
                log_error "Try: sudo ./scripts/phase2_flamegraph.sh"
                log_error "Or: echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
                rm -f /tmp/bitnet-perf-test.$$ 2>/dev/null
                exit 1
            fi
            rm -f /tmp/bitnet-perf-test.$$ 2>/dev/null
            ;;

        Darwin)
            log_info "macOS detected - using DTrace backend"
            # DTrace typically works without elevated privileges on modern macOS
            ;;

        *)
            log_warn "Unknown OS: $os_type. Profiling may not work correctly."
            ;;
    esac

    log_info "Profiling capability check: OK"
}

# Get system fingerprint for metadata
get_system_fingerprint() {
    local os_type=$(uname -s)
    local cpu_info=""

    case "$os_type" in
        Linux)
            if command -v lscpu &> /dev/null; then
                cpu_info=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
            else
                cpu_info=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
            fi
            ;;
        Darwin)
            cpu_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown CPU")
            ;;
        *)
            cpu_info="Unknown CPU"
            ;;
    esac

    echo "$cpu_info"
}

# Build optimized release binary
build_release_binary() {
    if [[ "$SKIP_BUILD" == "1" ]]; then
        log_info "Skipping build (BITNET_SKIP_BUILD=1)"
        if [[ ! -f "target/release/bitnet" ]]; then
            log_error "Binary not found: target/release/bitnet"
            log_error "Either build it first or unset BITNET_SKIP_BUILD"
            exit 1
        fi
        return 0
    fi

    log_info "Building optimized release binary with debug symbols..."

    # Use target-cpu=native for optimal performance but keep debug info for profiling
    RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
        cargo build --release --no-default-features --features cpu,full-cli

    if [[ ! -f "target/release/bitnet" ]]; then
        log_error "Build failed - binary not found: target/release/bitnet"
        exit 1
    fi

    log_info "Build complete"
}

# Generate flamegraph using cargo-flamegraph
generate_flamegraph_cargo() {
    local output_svg="$1"
    local prompt="$2"
    local max_tokens="$3"
    local label="$4"

    log_info "Generating flamegraph ($label) with cargo-flamegraph..."

    # Set determinism env vars
    local env_vars=""
    if [[ "$DETERMINISTIC" == "1" ]]; then
        env_vars="BITNET_DETERMINISTIC=1 BITNET_SEED=$SEED RAYON_NUM_THREADS=1 RUST_LOG=warn"
    else
        env_vars="RUST_LOG=warn"
    fi

    # Run cargo flamegraph (wraps perf record + flamegraph generation)
    eval "$env_vars" cargo flamegraph \
        --bin bitnet \
        --output "$output_svg" \
        -- run \
            --model "$MODEL" \
            --tokenizer "$TOKENIZER" \
            --prompt "$prompt" \
            --max-tokens "$max_tokens" \
            --greedy \
            --temperature 0.0 \
        2>&1 | grep -v "Compiling\|Finished\|Running" || true

    if [[ ! -f "$output_svg" ]]; then
        log_error "Flamegraph generation failed: $output_svg"
        return 1
    fi

    log_info "Flamegraph generated: $output_svg"
    return 0
}

# Generate flamegraph using samply
generate_flamegraph_samply() {
    local output_svg="$1"
    local prompt="$2"
    local max_tokens="$3"
    local label="$4"

    log_info "Generating flamegraph ($label) with samply..."

    # Set determinism env vars
    local env_vars=""
    if [[ "$DETERMINISTIC" == "1" ]]; then
        env_vars="BITNET_DETERMINISTIC=1 BITNET_SEED=$SEED RAYON_NUM_THREADS=1 RUST_LOG=warn"
    else
        env_vars="RUST_LOG=warn"
    fi

    # samply outputs to profile.json by default - we need to convert
    local temp_json="/tmp/bitnet-samply-$$.json"

    eval "$env_vars" samply record \
        --save-only \
        --output "$temp_json" \
        target/release/bitnet run \
            --model "$MODEL" \
            --tokenizer "$TOKENIZER" \
            --prompt "$prompt" \
            --max-tokens "$max_tokens" \
            --greedy \
            --temperature 0.0 \
        2>&1 | grep -v "Recording\|Saved" || true

    if [[ ! -f "$temp_json" ]]; then
        log_error "Samply recording failed"
        return 1
    fi

    # Convert to flamegraph SVG (requires inferno or flamegraph tool)
    if command -v inferno-flamegraph &> /dev/null; then
        inferno-flamegraph "$temp_json" > "$output_svg"
    elif command -v flamegraph &> /dev/null; then
        flamegraph "$temp_json" > "$output_svg"
    else
        log_warn "No flamegraph converter found. Saved profile data to: $temp_json"
        log_warn "Install inferno: cargo install inferno"
        cp "$temp_json" "$output_svg.json"
        rm -f "$temp_json"
        return 1
    fi

    rm -f "$temp_json"
    log_info "Flamegraph generated: $output_svg"
    return 0
}

# Generate metadata markdown for flamegraph
generate_metadata() {
    local svg_file="$1"
    local md_file="$2"
    local prompt="$3"
    local max_tokens="$4"
    local label="$5"

    local cpu_info=$(get_system_fingerprint)
    local rust_version=$(rustc --version)
    local svg_size=$(du -h "$svg_file" | cut -f1)

    cat > "$md_file" << EOF
# Phase 2 Flamegraph - $label

**Generated**: $TIMESTAMP
**Git Commit**: $GIT_COMMIT ($GIT_BRANCH)
**Model**: $(basename "$MODEL")
**Tokenizer**: $(basename "$TOKENIZER")
**Workload**: $max_tokens token(s), greedy decoding
**Prompt**: "$prompt"
**Hardware**: $cpu_info
**Rust Version**: $rust_version
**Flamegraph Tool**: $FLAMEGRAPH_TOOL
**SVG Size**: $svg_size

---

## Flamegraph Location

**SVG**: \`$(basename "$svg_file")\`

## How to View

1. **Open in browser**: \`firefox $svg_file\` or \`open $svg_file\`
2. **Click** any stack frame to zoom into that subtree
3. **Search** with Ctrl+F (or Cmd+F) to find specific functions
4. **Reset zoom** by clicking the top-level frame or refreshing

## How to Interpret

- **Width** = Time spent in that function (wider = more time)
- **Height** = Call stack depth (taller = deeper call chains)
- **Color** = Function category (varies by tool - typically green=userspace, yellow=kernel)
- **X-axis** = Not time! Stacks are ordered alphabetically for consistency

## Top Hotspots

> **Note**: Fill this section by analyzing the generated SVG.
> Look for the widest stack frames at the top level.

### Expected Hotspots (for BitNet-rs inference)

1. **Forward Pass** (~90-95% of total time)
   - \`bitnet_inference::forward\` or \`bitnet_models::transformer::forward\`
   - Sub-calls:
     - \`matmul_i2s\` (matrix multiplication - largest contributor)
     - \`RMSNorm\` or \`LayerNorm\` (normalization layers)
     - \`RoPE\` (rotary position embeddings)
     - \`Attention\` (self-attention mechanism)

2. **Logits Computation** (~3-5% of total time)
   - \`compute_logits\` or \`lm_head\` projection
   - \`softmax\` or temperature scaling

3. **Sampling** (<1% of total time)
   - Greedy argmax or nucleus sampling

4. **Miscellaneous** (<1% of total time)
   - Embedding lookup
   - Memory allocation
   - Tokenization overhead

### Performance Insights

> **TODO**: After analyzing the flamegraph, document:
> - Which kernel is the primary bottleneck?
> - Are there unexpected hotspots (e.g., memory allocation overhead)?
> - Is the forward pass CPU-bound or memory-bound? (check for cache misses)
> - Are SIMD optimizations being utilized? (look for AVX2/AVX-512 symbols)

### Actionable Optimization Opportunities

> **TODO**: Based on flamegraph analysis:
> - SIMD optimization candidates (e.g., scalar loops in matmul)
> - Memory layout improvements (e.g., cache-unfriendly access patterns)
> - Algorithmic optimizations (e.g., redundant computations)

## Build Configuration

\`\`\`bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \\
  cargo build --release --no-default-features --features cpu,full-cli
\`\`\`

## Determinism Settings

- **BITNET_DETERMINISTIC**: $DETERMINISTIC
- **BITNET_SEED**: $SEED
- **RAYON_NUM_THREADS**: ${RAYON_NUM_THREADS:-auto}

## Comparison to Phase 2 Timing

See: \`docs/baselines/perf/phase2_timing_i2s.md\` for timing breakdown comparison.

Expected correlations:
- Flamegraph \`forward\` width should match \`forward_us\` from timing receipt
- Logits computation width should match \`logits_us\`
- Sampling width should match \`sample_us\`

## How to Regenerate

\`\`\`bash
./scripts/phase2_flamegraph.sh \\
  "$MODEL" \\
  "$TOKENIZER" \\
  "$OUTPUT_DIR"
\`\`\`

Or with default auto-discovery:

\`\`\`bash
./scripts/phase2_flamegraph.sh
\`\`\`

## Related Documentation

- **Phase 2 Timing**: \`docs/baselines/perf/phase2_timing_i2s.md\`
- **Performance Tuning Guide**: \`docs/performance-tuning.md\`
- **Receipt Validation**: \`docs/how-to/receipt-verification.md\`
- **Kernel Benchmarks**: Run \`cargo bench --bench kernel_benchmarks --features cpu\`

---

*Generated by: \`scripts/phase2_flamegraph.sh\`*
*Last updated: $TIMESTAMP*
EOF

    log_info "Metadata generated: $md_file"
}

# Generate README for flamegraphs directory
generate_flamegraph_readme() {
    local readme_file="$OUTPUT_DIR/README.md"

    cat > "$readme_file" << 'EOF'
# BitNet-rs Flamegraph Baselines

This directory contains CPU flamegraphs for BitNet-rs inference profiling and hotspot analysis.

## Current Baselines

- **phase2_1tok.svg** - 1-token inference flamegraph (single decode step)
  - Metadata: `phase2_1tok.md`
  - Use case: Analyze per-token decode latency and kernel hotspots

- **phase2_10tok.svg** - 10-token inference flamegraph (short sequence)
  - Metadata: `phase2_10tok.md`
  - Use case: Analyze multi-token generation overhead and caching behavior

## How to Regenerate

```bash
# Auto-discover model and tokenizer
./scripts/phase2_flamegraph.sh

# Custom model and tokenizer
./scripts/phase2_flamegraph.sh models/custom.gguf models/tokenizer.json

# Use samply instead of cargo-flamegraph
BITNET_FLAMEGRAPH_TOOL=samply ./scripts/phase2_flamegraph.sh
```

## How to View Flamegraphs

1. **Open in browser**: Double-click the SVG or run `firefox phase2_1tok.svg`
2. **Click** any stack frame to zoom into that subtree
3. **Search** functions with Ctrl+F (e.g., search "matmul" to find matrix operations)
4. **Reset zoom** by clicking the top-level frame or refreshing

## Interpreting Flamegraphs

### Visual Legend

- **Width** = Time spent in function (wider = more CPU time consumed)
- **Height** = Call stack depth (taller = deeper function calls)
- **Color** = Function type (varies by tool; typically green=userspace, yellow=kernel)
- **X-axis** = NOT chronological time! Stacks are ordered alphabetically

### Key Functions to Look For

1. **Forward Pass Kernels** (should dominate flamegraph width)
   - `bitnet_inference::forward` or `bitnet_models::transformer::forward`
   - `matmul_*` functions (matrix multiplication - typically 80-90% of forward pass)
   - `RMSNorm` or `LayerNorm` (normalization - ~5-10%)
   - `RoPE` (rotary embeddings - ~2-5%)
   - `Attention` (self-attention - includes QKV projections)

2. **Quantization Operations**
   - `i2s_dequantize` or `i2s_matmul` (I2S-specific kernels)
   - `qk256_*` functions (QK256 quantization format)
   - Look for SIMD symbols: `avx2_*`, `avx512_*`, `neon_*`

3. **Memory Operations**
   - `alloc::*` (heap allocations - should be minimal for inference)
   - `memcpy` or `memmove` (data movement - ideally minimal due to zero-copy design)

4. **Unexpected Hotspots** (potential optimization opportunities)
   - Large allocator overhead (suggests excessive allocations)
   - Repeated identical stack frames (suggests redundant computation)
   - Kernel-space functions (yellow - suggests syscall overhead)

## Performance Analysis Workflow

1. **Generate baseline flamegraphs** (this script)
2. **Identify primary hotspot** (widest stack frame in forward pass)
3. **Cross-reference with timing data** (`docs/baselines/perf/phase2_timing_i2s.md`)
4. **Validate optimization opportunities** with kernel benchmarks:
   ```bash
   cargo bench --bench kernel_benchmarks --features cpu
   ```
5. **Implement optimization** (e.g., SIMD kernels, cache-friendly layouts)
6. **Re-generate flamegraphs** to validate improvement
7. **Compare before/after** using flamegraph diff tools

## Flamegraph Diff Analysis

To compare two flamegraphs:

```bash
# Generate new flamegraph after optimization
./scripts/phase2_flamegraph.sh models/new.gguf models/tokenizer.json

# Visual diff (requires flamegraph-diff tool)
# cargo install flamegraph-diff
flamegraph-diff phase2_1tok.svg phase2_1tok_new.svg
```

## Historical Archives

For regression tracking, consider archiving flamegraphs by date/commit:

```bash
mkdir -p archives/
cp phase2_1tok.svg archives/phase2_1tok_$(date +%Y%m%d)_${GIT_COMMIT}.svg
```

## Troubleshooting

### "Permission denied" when profiling

**Linux**: Adjust perf_event_paranoid or use sudo:
```bash
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
# Or run with sudo
sudo ./scripts/phase2_flamegraph.sh
```

**macOS**: DTrace typically works without elevated privileges.

### "flamegraph tool not found"

Install cargo-flamegraph or samply:
```bash
cargo install --locked flamegraph  # Preferred
# Or
cargo install samply              # Alternative
```

### Empty or incomplete flamegraphs

- Check that binary has debug symbols (default in this script)
- Increase max-tokens if flamegraph is too narrow
- Ensure profiling ran successfully (check for .perf files)

## Related Documentation

- **Performance Tuning**: `docs/performance-tuning.md`
- **Kernel Benchmarks**: `crates/bitnet-kernels/benches/kernel_benchmarks.rs`
- **Receipt Validation**: `docs/how-to/receipt-verification.md`
- **Phase 2 Timing**: `docs/baselines/perf/phase2_timing_i2s.md`

---

*Flamegraphs generated by: `scripts/phase2_flamegraph.sh`*
*For questions, see: docs/performance-tuning.md#profiling-and-monitoring*
EOF

    log_info "README generated: $readme_file"
}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

main() {
    echo "==================================================================="
    echo "BitNet-rs Phase 2 Flamegraph Generation"
    echo "==================================================================="
    echo ""

    # Step 1: Pre-flight checks
    log_info "Step 1/7: Pre-flight checks"
    auto_discover_files
    detect_flamegraph_tool
    check_profiling_capability
    echo ""

    # Step 2: Create output directory
    log_info "Step 2/7: Creating output directory"
    mkdir -p "$OUTPUT_DIR"
    log_info "Output directory: $OUTPUT_DIR"
    echo ""

    # Step 3: Build release binary
    log_info "Step 3/7: Building optimized release binary"
    build_release_binary
    echo ""

    # Step 4: Generate 1-token flamegraph
    log_info "Step 4/7: Generating 1-token flamegraph"
    local svg_1tok="$OUTPUT_DIR/phase2_1tok.svg"
    local md_1tok="$OUTPUT_DIR/phase2_1tok.md"

    case "$FLAMEGRAPH_TOOL" in
        cargo-flamegraph)
            generate_flamegraph_cargo "$svg_1tok" "2+2=" 1 "1-token decode"
            ;;
        samply)
            generate_flamegraph_samply "$svg_1tok" "2+2=" 1 "1-token decode"
            ;;
        *)
            log_error "Unknown flamegraph tool: $FLAMEGRAPH_TOOL"
            exit 1
            ;;
    esac

    generate_metadata "$svg_1tok" "$md_1tok" "2+2=" 1 "1-Token Decode"
    echo ""

    # Step 5: Generate 10-token flamegraph
    log_info "Step 5/7: Generating 10-token flamegraph"
    local svg_10tok="$OUTPUT_DIR/phase2_10tok.svg"
    local md_10tok="$OUTPUT_DIR/phase2_10tok.md"

    case "$FLAMEGRAPH_TOOL" in
        cargo-flamegraph)
            generate_flamegraph_cargo "$svg_10tok" "What is 2+2?" 10 "10-token sequence"
            ;;
        samply)
            generate_flamegraph_samply "$svg_10tok" "What is 2+2?" 10 "10-token sequence"
            ;;
        *)
            log_error "Unknown flamegraph tool: $FLAMEGRAPH_TOOL"
            exit 1
            ;;
    esac

    generate_metadata "$svg_10tok" "$md_10tok" "What is 2+2?" 10 "10-Token Sequence"
    echo ""

    # Step 6: Generate README
    log_info "Step 6/7: Generating flamegraphs README"
    generate_flamegraph_readme
    echo ""

    # Step 7: Summary
    log_info "Step 7/7: Summary"
    echo "==================================================================="
    echo "Flamegraph Generation Complete"
    echo "==================================================================="
    echo ""
    echo "Output Directory: $OUTPUT_DIR"
    echo ""
    echo "Generated Files:"
    echo "  - $svg_1tok"
    echo "  - $md_1tok"
    echo "  - $svg_10tok"
    echo "  - $md_10tok"
    echo "  - $OUTPUT_DIR/README.md"
    echo ""
    echo "Next Steps:"
    echo "  1. Open flamegraphs in browser:"
    echo "     firefox $svg_1tok"
    echo "     firefox $svg_10tok"
    echo ""
    echo "  2. Analyze hotspots and update metadata files with insights"
    echo ""
    echo "  3. Compare with timing data:"
    echo "     cat docs/baselines/perf/phase2_timing_i2s.md"
    echo ""
    echo "  4. Run kernel benchmarks for detailed performance metrics:"
    echo "     cargo bench --bench kernel_benchmarks --features cpu"
    echo ""
    echo "For more information, see: docs/performance-tuning.md"
    echo "==================================================================="
}

# Run main function
main
