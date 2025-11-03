#!/usr/bin/env bash
# Common utilities for BitNet.rs validation scripts

set -euo pipefail

# Binary auto-detection
find_bitnet_binary() {
    if [[ -n "${BITNET_BIN:-}" ]]; then
        echo "$BITNET_BIN"
    elif command -v bitnet >/dev/null 2>&1; then
        command -v bitnet
    elif [[ -x "target/release/bitnet" ]]; then
        echo "target/release/bitnet"
    else
        echo "ERROR: Could not find bitnet binary" >&2
        echo "Set BITNET_BIN or build with: cargo build --release --no-default-features --features cpu" >&2
        exit 1
    fi
}

# Platform detection and stamping
get_platform_name() {
    local platform="$(uname -s)-$(uname -m)"

    # Detect WSL2
    if grep -qi microsoft /proc/version 2>/dev/null; then
        platform="${platform}-WSL2"
    fi

    echo "$platform"
}

# WSL2 detection and warning
detect_wsl2() {
    if grep -qi microsoft /proc/version 2>/dev/null; then
        echo -e "\033[1;33mNOTE: Running under WSL2 - results reflect a guest VM environment\033[0m"
        return 0
    fi
    return 1
}

get_platform_info() {
    local platform_json="/tmp/platform_info.json"

    # Basic system info
    local os_name=$(uname -s)
    local os_version=$(uname -r)
    local arch=$(uname -m)
    local hostname=$(hostname)

    # Detect WSL
    local is_wsl="false"
    local wsl_note=""
    if [[ -f /proc/version ]] && grep -qi microsoft /proc/version; then
        is_wsl="true"
        wsl_note="Running under WSL2 (Windows Subsystem for Linux)"
    fi

    # CPU info
    local cpu_model="unknown"
    local cpu_cores="unknown"
    if [[ -f /proc/cpuinfo ]]; then
        cpu_model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs || echo "unknown")
        cpu_cores=$(grep -c "processor" /proc/cpuinfo || echo "unknown")
    elif command -v sysctl >/dev/null 2>&1; then
        # macOS
        cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
        cpu_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
    fi

    # Memory info
    local mem_total="unknown"
    if command -v free >/dev/null 2>&1; then
        mem_total=$(free -m | awk '/^Mem:/ {print $2}')
    elif command -v sysctl >/dev/null 2>&1; then
        # macOS
        mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        mem_total=$((mem_bytes / 1024 / 1024))
    fi

    # Rust toolchain
    local rust_version="unknown"
    if command -v rustc >/dev/null 2>&1; then
        rust_version=$(rustc --version | cut -d' ' -f2)
    fi

    # Create JSON
    cat > "$platform_json" <<EOF
{
  "os": {
    "name": "$os_name",
    "version": "$os_version",
    "arch": "$arch",
    "hostname": "$hostname",
    "is_wsl": $is_wsl,
    "wsl_note": "$wsl_note"
  },
  "cpu": {
    "model": "$cpu_model",
    "cores": "$cpu_cores"
  },
  "memory": {
    "total_mb": "$mem_total"
  },
  "toolchain": {
    "rust": "$rust_version"
  },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    echo "$platform_json"
}

# Print platform banner
print_platform_banner() {
    echo "========================================"
    echo "Platform Information"
    echo "========================================"

    local platform_json=$(get_platform_info)

    # Parse JSON (using jq if available, otherwise raw)
    if command -v jq >/dev/null 2>&1; then
        local os_name=$(jq -r '.os.name' "$platform_json")
        local os_version=$(jq -r '.os.version' "$platform_json")
        local cpu_model=$(jq -r '.cpu.model' "$platform_json")
        local cpu_cores=$(jq -r '.cpu.cores' "$platform_json")
        local mem_total=$(jq -r '.memory.total_mb' "$platform_json")
        local rust_version=$(jq -r '.toolchain.rust' "$platform_json")
        local is_wsl=$(jq -r '.os.is_wsl' "$platform_json")
        local wsl_note=$(jq -r '.os.wsl_note' "$platform_json")
    else
        # Fallback to grep
        local os_name=$(grep '"name":' "$platform_json" | head -1 | cut -d'"' -f4)
        local cpu_model=$(grep '"model":' "$platform_json" | cut -d'"' -f4)
        local is_wsl=$(grep '"is_wsl":' "$platform_json" | cut -d':' -f2 | tr -d ' ,')
    fi

    echo "OS: $os_name $os_version"
    echo "CPU: $cpu_model ($cpu_cores cores)"
    echo "Memory: ${mem_total}MB"
    echo "Rust: $rust_version"

    if [[ "$is_wsl" == "true" ]]; then
        echo ""
        echo "⚠️  NOTE: Running under WSL2 (Windows Subsystem for Linux)"
        echo "   Results reflect guest VM environment, not native Windows performance"
    fi

    echo "========================================"
    echo ""
}

# Setup deterministic environment
setup_deterministic_env() {
    echo "Setting up deterministic environment..."

    # Core determinism variables
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42

    # Thread control
    export RAYON_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export BLIS_NUM_THREADS=1

    # CUDA determinism (if applicable)
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export CUDA_LAUNCH_BLOCKING=1

    echo "Environment configured:"
    echo "  BITNET_DETERMINISTIC=1"
    echo "  BITNET_SEED=42"
    echo "  All threading limited to 1"
    echo ""
}

# Create methods & environment box for docs
create_methods_box() {
    local output_file="${1:-methods_env.md}"
    local bitnet_bin=$(find_bitnet_binary)
    local bitnet_version="unknown"

    if [[ -x "$bitnet_bin" ]]; then
        bitnet_version=$("$bitnet_bin" --version 2>/dev/null | cut -d' ' -f2 || echo "unknown")
    fi

    # Get platform info
    local platform_json=$(get_platform_info)

    # Parse info
    if command -v jq >/dev/null 2>&1; then
        local os_info=$(jq -r '"\(.os.name) \(.os.version)"' "$platform_json")
        local cpu_info=$(jq -r '"\(.cpu.model) @ \(.cpu.cores) cores"' "$platform_json")
        local is_wsl=$(jq -r '.os.is_wsl' "$platform_json")
    else
        local os_info="$(uname -s) $(uname -r)"
        local cpu_info="$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs) @ $(nproc) cores"
        local is_wsl="false"
    fi

    # Python/ML versions
    local python_version="unknown"
    local torch_version="unknown"
    local transformers_version="unknown"

    if command -v python3 >/dev/null 2>&1; then
        python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        torch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
        transformers_version=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
    fi

    # Create the box
    cat > "$output_file" <<EOF
## Methods & Environment

\`\`\`
Platform: $os_info, $cpu_info
BitNet CLI: v$bitnet_version | Rust: $(rustc --version | cut -d' ' -f2) | Python: $python_version
ML Stack: PyTorch $torch_version | Transformers $transformers_version
Determinism: BITNET_DETERMINISTIC=1 RAYON/OMP/MKL/BLAS=1
Validation: 3 prompts, max_new_tokens=128, warmup=1, median of 5 runs
\`\`\`
EOF

    if [[ "$is_wsl" == "true" ]]; then
        echo "" >> "$output_file"
        echo "> **Note:** Tests run under WSL2. Performance reflects virtualized environment." >> "$output_file"
    fi

    echo "Methods box written to: $output_file"
}

# Ensure output directory exists
ensure_output_dir() {
    local dir="${1:-validation_results}"
    mkdir -p "$dir"
    echo "$dir"
}

# Log with timestamp
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Check if running in CI
is_ci() {
    [[ -n "${CI:-}" ]] || [[ -n "${GITHUB_ACTIONS:-}" ]] || [[ -n "${GITLAB_CI:-}" ]]
}

# Export functions for use in other scripts
export -f find_bitnet_binary
export -f get_platform_name
export -f detect_wsl2
export -f get_platform_info
export -f print_platform_banner
export -f setup_deterministic_env
export -f create_methods_box
export -f ensure_output_dir
export -f log_info
export -f is_ci
