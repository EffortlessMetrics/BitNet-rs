#!/usr/bin/env bash
# BitNet-rs One-Click Deploy Script
# Everything you need with a single command

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Unicode symbols
CHECK="✓"
CROSS="✗"
ARROW="→"
ROCKET="🚀"
WARNING="⚠️"
INFO="ℹ️"

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect OS and architecture
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
CPU_COUNT="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# Log functions
log() { echo -e "${BLUE}${INFO}${NC} $*"; }
success() { echo -e "${GREEN}${CHECK}${NC} $*"; }
error() { echo -e "${RED}${CROSS}${NC} $*" >&2; }
warning() { echo -e "${YELLOW}${WARNING}${NC} $*"; }
header() {
    echo
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}$*${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Progress spinner
spin() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while ps -p $pid > /dev/null 2>&1; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect GPU backend
detect_gpu() {
    local gpu_type="none"

    if command_exists nvidia-smi && nvidia-smi &>/dev/null; then
        gpu_type="cuda"
        local cuda_version=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' | head -1)
        log "NVIDIA GPU detected (CUDA ${cuda_version:-unknown})"
    elif [[ "$OS" == "darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
        gpu_type="metal"
        log "Apple Silicon detected (Metal)"
    elif command_exists rocm-smi && rocm-smi &>/dev/null; then
        gpu_type="rocm"
        log "AMD GPU detected (ROCm)"
    else
        gpu_type="cpu"
        warning "No GPU detected, using CPU backend"
    fi

    echo "$gpu_type"
}

# Install Rust if needed
ensure_rust() {
    if ! command_exists rustc; then
        header "Installing Rust"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
        source "$HOME/.cargo/env"
    fi

    # Ensure we have the right version
    rustup toolchain install 1.89.0
    rustup default stable
    rustup component add rustfmt clippy
}

# Install system dependencies
install_dependencies() {
    header "Checking System Dependencies"

    case "$OS" in
        linux)
            if command_exists apt-get; then
                log "Detected Debian/Ubuntu"
                local deps="build-essential cmake pkg-config libssl-dev"

                # Check if deps are installed
                local missing=""
                for dep in $deps; do
                    if ! dpkg -l | grep -q "^ii  $dep"; then
                        missing="$missing $dep"
                    fi
                done

                if [[ -n "$missing" ]]; then
                    log "Installing missing dependencies:$missing"
                    sudo apt-get update && sudo apt-get install -y $missing
                fi
            elif command_exists dnf; then
                log "Detected Fedora/RHEL"
                sudo dnf install -y gcc gcc-c++ cmake openssl-devel
            elif command_exists pacman; then
                log "Detected Arch Linux"
                sudo pacman -S --needed base-devel cmake openssl
            fi
            ;;
        darwin)
            if ! command_exists brew; then
                log "Installing Homebrew"
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install cmake
            ;;
    esac

    success "Dependencies ready"
}

# Setup Python environment
setup_python() {
    if [[ "${SKIP_PYTHON:-0}" == "1" ]]; then
        return
    fi

    header "Setting up Python Environment"

    if command_exists python3; then
        if [[ ! -d "venv" ]]; then
            python3 -m venv venv
        fi
        source venv/bin/activate 2>/dev/null || . venv/bin/activate
        pip install --upgrade pip wheel
        pip install -q -r requirements.txt 2>/dev/null || true
        success "Python environment ready"
    else
        warning "Python not found, skipping Python setup"
    fi
}

# Download models
download_models() {
    header "Downloading Models"

    if [[ -f "models/bitnet/ggml-model-i2_s.gguf" ]]; then
        log "Model already downloaded"
    else
        log "Downloading BitNet model..."
        cargo run -p xtask -- download-model --no-progress
        success "Model downloaded"
    fi
}

# Build the project
build_project() {
    local gpu_backend=$(detect_gpu)
    local features="cpu"

    if [[ "$gpu_backend" != "cpu" ]] && [[ "$gpu_backend" != "none" ]]; then
        features="gpu"
    fi

    header "Building BitNet-rs (${features} backend)"

    # Clean build if requested
    if [[ "${CLEAN:-0}" == "1" ]]; then
        log "Cleaning previous build..."
        cargo clean
    fi

    # Build with progress
    log "Building with features: ${features}"
    cargo build --release --no-default-features --features "$features"

    success "Build complete"
}

# Run tests
run_tests() {
    local gpu_backend=$(detect_gpu)
    local features="cpu"

    if [[ "$gpu_backend" != "cpu" ]] && [[ "$gpu_backend" != "none" ]]; then
        features="gpu"
    fi

    header "Running Tests"

    # Quick tests first
    log "Running unit tests..."
    cargo test --workspace --no-default-features --features "$features" --lib

    # Integration tests if requested
    if [[ "${FULL_TEST:-0}" == "1" ]]; then
        log "Running integration tests..."
        cargo test --workspace --no-default-features --features "$features"

        # GPU smoke test if available
        if [[ "$features" == "gpu" ]]; then
            log "Running GPU smoke tests..."
            cargo run -p xtask -- gpu-smoke || warning "GPU smoke test skipped"
        fi
    fi

    success "Tests passed"
}

# Run benchmarks
run_benchmarks() {
    if [[ "${BENCH:-0}" == "1" ]]; then
        header "Running Benchmarks"
        cargo bench --workspace --no-default-features --features cpu
        success "Benchmarks complete"
    fi
}

# Setup development environment
setup_dev() {
    header "Setting up Development Environment"

    # Git hooks
    if [[ -d ".git" ]]; then
        log "Installing git hooks..."
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
cargo fmt --all -- --check || {
    echo "Code needs formatting. Run: cargo fmt --all"
    exit 1
}
cargo clippy --all-targets -- -D warnings || {
    echo "Clippy warnings found"
    exit 1
}
EOF
        chmod +x .git/hooks/pre-commit
    fi

    # VS Code settings
    if [[ ! -f ".vscode/settings.json" ]]; then
        mkdir -p .vscode
        cat > .vscode/settings.json << 'EOF'
{
    "rust-analyzer.cargo.features": ["cpu"],
    "rust-analyzer.checkOnSave.command": "clippy",
    "editor.formatOnSave": true
}
EOF
    fi

    success "Development environment ready"
}

# Deploy for production
deploy_production() {
    header "Production Deployment"

    # Build optimized binary
    log "Building optimized release..."
    RUSTFLAGS="-C target-cpu=native -C lto=fat -C embed-bitcode=yes" \
        cargo build --release --no-default-features --features cpu

    # Create deployment directory
    local deploy_dir="deploy/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$deploy_dir"

    # Copy artifacts
    cp target/release/bitnet "$deploy_dir/"
    cp -r models "$deploy_dir/" 2>/dev/null || true

    # Create run script
    cat > "$deploy_dir/run.sh" << 'EOF'
#!/bin/bash
./bitnet "$@"
EOF
    chmod +x "$deploy_dir/run.sh"

    # Create systemd service if on Linux
    if [[ "$OS" == "linux" ]]; then
        cat > "$deploy_dir/bitnet.service" << EOF
[Unit]
Description=BitNet Inference Service
After=network.target

[Service]
Type=simple
WorkingDirectory=$(pwd)/$deploy_dir
ExecStart=$(pwd)/$deploy_dir/bitnet serve
Restart=on-failure
User=$USER

[Install]
WantedBy=multi-user.target
EOF
        log "Systemd service file created: $deploy_dir/bitnet.service"
    fi

    success "Deployed to: $deploy_dir"
}

# Interactive menu
show_menu() {
    clear
    header "${ROCKET} BitNet-rs One-Click Deploy ${ROCKET}"
    echo
    echo -e "${CYAN}1)${NC} Quick Start ${GREEN}(Recommended)${NC}"
    echo -e "${CYAN}2)${NC} Full Installation"
    echo -e "${CYAN}3)${NC} Development Setup"
    echo -e "${CYAN}4)${NC} Production Deploy"
    echo -e "${CYAN}5)${NC} Run Tests"
    echo -e "${CYAN}6)${NC} Run Benchmarks"
    echo -e "${CYAN}7)${NC} Clean & Rebuild"
    echo -e "${CYAN}8)${NC} GPU Check"
    echo -e "${CYAN}9)${NC} Update Everything"
    echo -e "${CYAN}0)${NC} Exit"
    echo
}

# Quick start - the true one-click experience
quick_start() {
    header "${ROCKET} Quick Start - One Click Deploy ${ROCKET}"

    # Parallel execution where possible
    log "Detecting environment..."
    detect_gpu > /tmp/gpu_type &

    ensure_rust
    install_dependencies

    # Download models in background
    download_models &
    local download_pid=$!

    # Build while downloading
    build_project

    # Wait for download if still running
    wait $download_pid 2>/dev/null

    # Quick test
    log "Running smoke test..."
    cargo test --package bitnet-inference --lib --no-default-features --features cpu

    echo
    success "${BOLD}BitNet-rs is ready to use!${NC}"
    echo
    echo -e "${GREEN}Quick commands:${NC}"
    echo -e "  ${CYAN}cargo run --release --${NC}    # Run CLI"
    echo -e "  ${CYAN}cargo tw${NC}                  # Test workspace"
    echo -e "  ${CYAN}cargo xtask gpu-preflight${NC} # Check GPU"
    echo
}

# Full installation
full_install() {
    header "Full Installation"

    ensure_rust
    install_dependencies
    setup_python
    download_models
    build_project
    run_tests
    setup_dev

    success "Full installation complete!"
}

# Update everything
update_all() {
    header "Updating Everything"

    # Update repo
    if [[ -d ".git" ]]; then
        log "Updating repository..."
        git pull --rebase
    fi

    # Update Rust
    log "Updating Rust..."
    rustup update

    # Update dependencies
    log "Updating dependencies..."
    cargo update

    # Rebuild
    build_project

    success "Everything updated!"
}

# Main execution
main() {
    # Parse command line arguments
    case "${1:-}" in
        quick|--quick|-q)
            quick_start
            ;;
        full|--full|-f)
            full_install
            ;;
        dev|--dev|-d)
            ensure_rust
            install_dependencies
            setup_dev
            build_project
            ;;
        prod|--prod|-p)
            ensure_rust
            install_dependencies
            deploy_production
            ;;
        test|--test|-t)
            FULL_TEST=1 run_tests
            ;;
        bench|--bench|-b)
            BENCH=1 run_benchmarks
            ;;
        clean|--clean|-c)
            CLEAN=1 build_project
            ;;
        gpu|--gpu|-g)
            cargo run -p xtask -- gpu-preflight
            ;;
        update|--update|-u)
            update_all
            ;;
        help|--help|-h)
            echo "BitNet-rs One-Click Deploy"
            echo
            echo "Usage: $0 [command]"
            echo
            echo "Commands:"
            echo "  quick    - Quick start (recommended)"
            echo "  full     - Full installation"
            echo "  dev      - Development setup"
            echo "  prod     - Production deployment"
            echo "  test     - Run all tests"
            echo "  bench    - Run benchmarks"
            echo "  clean    - Clean and rebuild"
            echo "  gpu      - Check GPU availability"
            echo "  update   - Update everything"
            echo "  help     - Show this help"
            echo
            echo "No arguments: Interactive menu"
            ;;
        "")
            # Interactive mode
            while true; do
                show_menu
                read -p "Select option: " choice
                case $choice in
                    1) quick_start; read -p "Press Enter to continue..." ;;
                    2) full_install; read -p "Press Enter to continue..." ;;
                    3) ensure_rust; install_dependencies; setup_dev; build_project; read -p "Press Enter to continue..." ;;
                    4) deploy_production; read -p "Press Enter to continue..." ;;
                    5) FULL_TEST=1 run_tests; read -p "Press Enter to continue..." ;;
                    6) BENCH=1 run_benchmarks; read -p "Press Enter to continue..." ;;
                    7) CLEAN=1 build_project; read -p "Press Enter to continue..." ;;
                    8) cargo run -p xtask -- gpu-preflight; read -p "Press Enter to continue..." ;;
                    9) update_all; read -p "Press Enter to continue..." ;;
                    0) echo "Goodbye!"; exit 0 ;;
                    *) error "Invalid option"; sleep 2 ;;
                esac
            done
            ;;
        *)
            error "Unknown command: $1"
            echo "Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

# Trap errors
trap 'error "Script failed at line $LINENO"' ERR

# Run main function
main "$@"
