#!/bin/bash
# BitNet-rs Development Environment Setup
# This script sets up a complete development environment for BitNet-rs

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
    ____  _ _   _   _      _
   |  _ \(_) | | \ | |    | |
   | |_) |_| |_|  \| | ___| |_ _ __ ___
   |  _ <| | __| . ` |/ _ \ __| '__/ __|
   | |_) | | |_| |\  |  __/ |_| |  \__ \
   |____/|_|\__|_| \_|\___|\__|_|  |___/

   Production-Ready Rust Implementation
   Development Environment Setup
EOF
    echo -e "${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_system() {
    log_info "Checking system requirements..."

    # Check OS
    case "$(uname -s)" in
        Linux*)     OS="Linux";;
        Darwin*)    OS="macOS";;
        CYGWIN*|MINGW*|MSYS*) OS="Windows";;
        *)          OS="Unknown";;
    esac

    log_info "Operating System: $OS"

    # Check architecture
    ARCH=$(uname -m)
    log_info "Architecture: $ARCH"

    # Check if we're in the right directory
    if [[ ! -f "Cargo.toml" ]] || ! grep -q "bitnet" Cargo.toml; then
        log_error "This script must be run from the BitNet-rs repository root"
        exit 1
    fi

    log_info "✅ System check passed"
}

# Install Rust toolchain
install_rust() {
    log_info "Setting up Rust toolchain..."

    if command_exists rustc; then
        RUST_VERSION=$(rustc --version | cut -d' ' -f2)
        log_info "Rust already installed: $RUST_VERSION"

        # Check if version is recent enough
        REQUIRED_VERSION="1.89.0"
        if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$RUST_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
            log_warn "Rust version $RUST_VERSION is older than required $REQUIRED_VERSION"
            log_info "Updating Rust..."
            rustup update stable
        fi
    else
        log_info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Install required components
    log_info "Installing Rust components..."
    rustup component add rustfmt clippy llvm-tools-preview

    # Install useful targets
    log_info "Installing additional targets..."
    case "$OS" in
        Linux)
            rustup target add x86_64-unknown-linux-musl
            if [[ "$ARCH" == "x86_64" ]]; then
                rustup target add aarch64-unknown-linux-gnu
            fi
            ;;
        macOS)
            if [[ "$ARCH" == "x86_64" ]]; then
                rustup target add aarch64-apple-darwin
            else
                rustup target add x86_64-apple-darwin
            fi
            ;;
        Windows)
            rustup target add x86_64-pc-windows-gnu
            ;;
    esac

    log_info "✅ Rust toolchain setup complete"
}

# Install development tools
install_dev_tools() {
    log_info "Installing development tools..."

    # Essential cargo tools
    local tools=(
        "cargo-audit"           # Security auditing
        "cargo-deny"            # License and dependency checking
        "cargo-machete"         # Find unused dependencies
        "cargo-outdated"        # Check for outdated dependencies
        "cargo-llvm-cov"        # Code coverage
        "cargo-criterion"       # Benchmarking
        "cargo-expand"          # Macro expansion
        "cargo-watch"           # File watching
        "cargo-edit"            # Cargo.toml editing
    )

    for tool in "${tools[@]}"; do
        if ! command_exists "$tool"; then
            log_info "Installing $tool..."
            cargo install "$tool"
        else
            log_debug "$tool already installed"
        fi
    done

    # Cross-compilation tool (optional)
    if [[ "$OS" == "Linux" ]]; then
        if ! command_exists cross; then
            log_info "Installing cross for cross-compilation..."
            cargo install cross --git https://github.com/cross-rs/cross
        fi
    fi

    log_info "✅ Development tools installed"
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."

    case "$OS" in
        Linux)
            if command_exists apt; then
                log_info "Installing dependencies with apt..."
                sudo apt update
                sudo apt install -y build-essential pkg-config libssl-dev

                # Optional: GPU support
                if lspci | grep -i nvidia >/dev/null; then
                    log_info "NVIDIA GPU detected, consider installing CUDA toolkit"
                    log_warn "CUDA installation is optional and can be done separately"
                fi
            elif command_exists yum; then
                log_info "Installing dependencies with yum..."
                sudo yum groupinstall -y "Development Tools"
                sudo yum install -y openssl-devel pkg-config
            elif command_exists pacman; then
                log_info "Installing dependencies with pacman..."
                sudo pacman -S --needed base-devel openssl pkg-config
            else
                log_warn "Unknown Linux package manager, please install build tools manually"
            fi
            ;;
        macOS)
            if command_exists brew; then
                log_info "Installing dependencies with Homebrew..."
                brew install pkg-config openssl
            else
                log_warn "Homebrew not found, please install it first: https://brew.sh"
                log_info "Installing Xcode Command Line Tools..."
                xcode-select --install || true
            fi
            ;;
        Windows)
            log_info "On Windows, ensure you have:"
            log_info "  - Visual Studio Build Tools or Visual Studio with C++ support"
            log_info "  - Git for Windows"
            log_warn "Some dependencies may need manual installation"
            ;;
    esac

    log_info "✅ System dependencies setup complete"
}

# Setup development configuration
setup_dev_config() {
    log_info "Setting up development configuration..."

    # Create .cargo/config.toml if it doesn't exist
    if [[ ! -f ".cargo/config.toml" ]]; then
        log_info "Creating .cargo/config.toml..."
        mkdir -p .cargo
        cat > .cargo/config.toml << 'EOF'
# Cargo configuration for BitNet-rs development

[build]
# Disable crossval feature by default for fast builds
rustflags = []

[env]
# Environment variables for development
RUST_BACKTRACE = "1"

[alias]
# Convenient aliases for development
check-all = "check --workspace --all-targets --all-features"
test-all = "test --workspace --all-targets --features cpu"
bench-all = "bench --workspace --features cpu"
doc-all = "doc --workspace --features cpu --no-deps"
fmt-all = "fmt --all"
clippy-all = "clippy --workspace --all-targets --features cpu -- -D warnings"

# Quality checks
quality = "clippy --workspace --all-targets --features cpu -- -D warnings"
security = "audit"
coverage = "llvm-cov --workspace --features cpu --html"

# Cross-validation (requires setup)
crossval = "test --features crossval"
crossval-bench = "bench --features crossval"

[target.'cfg(target_os = "linux")']
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

[profile.dev]
# Faster builds for development
debug = 1
opt-level = 0

[profile.dev-optimized]
inherits = "dev"
opt-level = 2
debug = 1

[profile.bench]
debug = true
EOF
    else
        log_debug ".cargo/config.toml already exists"
    fi

    # Setup git hooks (optional)
    if [[ -d ".git" ]]; then
        log_info "Setting up git hooks..."
        mkdir -p .git/hooks

        # Pre-commit hook
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for BitNet-rs

set -e

echo "Running pre-commit checks..."

# Format check
if ! cargo fmt --all -- --check; then
    echo "❌ Code formatting issues found. Run 'cargo fmt --all' to fix."
    exit 1
fi

# Clippy check
if ! cargo clippy --workspace --all-targets --features cpu -- -D warnings; then
    echo "❌ Clippy issues found. Please fix the warnings."
    exit 1
fi

# Quick test
if ! cargo test --workspace --features cpu --lib; then
    echo "❌ Tests failed. Please fix the failing tests."
    exit 1
fi

echo "✅ Pre-commit checks passed"
EOF
        chmod +x .git/hooks/pre-commit
        log_info "Git pre-commit hook installed"
    fi

    log_info "✅ Development configuration complete"
}

# Setup IDE configuration
setup_ide_config() {
    log_info "Setting up IDE configuration..."

    # VS Code settings
    if [[ ! -d ".vscode" ]]; then
        mkdir -p .vscode

        # Settings
        cat > .vscode/settings.json << 'EOF'
{
    "rust-analyzer.cargo.features": ["cpu"],
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.checkOnSave.extraArgs": ["--", "-D", "warnings"],
    "rust-analyzer.cargo.buildScripts.enable": true,
    "rust-analyzer.procMacro.enable": true,
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer",
        "editor.formatOnSave": true
    },
    "rust-analyzer.lens.enable": true,
    "rust-analyzer.lens.run.enable": true,
    "rust-analyzer.lens.debug.enable": true
}
EOF

        # Extensions recommendations
        cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "rust-lang.rust-analyzer",
        "vadimcn.vscode-lldb",
        "serayuzgur.crates",
        "tamasfe.even-better-toml",
        "usernamehw.errorlens",
        "ms-vscode.test-adapter-converter"
    ]
}
EOF

        # Tasks
        cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "cargo check",
            "type": "cargo",
            "command": "check",
            "args": ["--workspace", "--all-targets", "--features", "cpu"],
            "group": "build",
            "presentation": {
                "clear": true
            }
        },
        {
            "label": "cargo test",
            "type": "cargo",
            "command": "test",
            "args": ["--workspace", "--features", "cpu"],
            "group": "test",
            "presentation": {
                "clear": true
            }
        },
        {
            "label": "cargo bench",
            "type": "cargo",
            "command": "bench",
            "args": ["--workspace", "--features", "cpu"],
            "group": "test",
            "presentation": {
                "clear": true
            }
        },
        {
            "label": "cargo doc",
            "type": "cargo",
            "command": "doc",
            "args": ["--workspace", "--features", "cpu", "--no-deps", "--open"],
            "group": "build",
            "presentation": {
                "clear": true
            }
        }
    ]
}
EOF

        log_info "VS Code configuration created"
    else
        log_debug "VS Code configuration already exists"
    fi

    log_info "✅ IDE configuration complete"
}

# Verify installation
verify_setup() {
    log_info "Verifying development environment..."

    # Check Rust installation
    if ! command_exists rustc; then
        log_error "Rust not found in PATH"
        return 1
    fi

    local rust_version=$(rustc --version)
    log_info "Rust version: $rust_version"

    # Check cargo tools
    local tools=("cargo-audit" "cargo-deny" "cargo-clippy" "cargo-fmt")
    for tool in "${tools[@]}"; do
        if command_exists "$tool"; then
            log_debug "✅ $tool available"
        else
            log_warn "❌ $tool not available"
        fi
    done

    # Test basic build
    log_info "Testing basic build..."
    if cargo check --workspace --features cpu; then
        log_info "✅ Basic build successful"
    else
        log_error "❌ Basic build failed"
        return 1
    fi

    # Test formatting
    log_info "Testing code formatting..."
    if cargo fmt --all -- --check; then
        log_info "✅ Code formatting OK"
    else
        log_warn "❌ Code formatting issues (run 'cargo fmt --all' to fix)"
    fi

    # Test clippy
    log_info "Testing clippy..."
    if cargo clippy --workspace --all-targets --features cpu -- -D warnings; then
        log_info "✅ Clippy checks passed"
    else
        log_warn "❌ Clippy issues found"
    fi

    log_info "✅ Development environment verification complete"
}

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Set up BitNet-rs development environment.

OPTIONS:
    --skip-rust         Skip Rust installation
    --skip-tools        Skip development tools installation
    --skip-system       Skip system dependencies installation
    --skip-config       Skip development configuration
    --skip-ide          Skip IDE configuration
    --skip-verify       Skip verification
    --help              Show this help message

EXAMPLES:
    $0                  # Full setup
    $0 --skip-system    # Skip system dependencies
    $0 --skip-ide       # Skip IDE configuration

The script will:
1. Check system requirements
2. Install/update Rust toolchain
3. Install development tools
4. Install system dependencies
5. Setup development configuration
6. Setup IDE configuration
7. Verify the installation

For cross-validation setup, run:
    ./scripts/dev-crossval.sh
EOF
}

# Parse command line arguments
SKIP_RUST=false
SKIP_TOOLS=false
SKIP_SYSTEM=false
SKIP_CONFIG=false
SKIP_IDE=false
SKIP_VERIFY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-rust)
            SKIP_RUST=true
            shift
            ;;
        --skip-tools)
            SKIP_TOOLS=true
            shift
            ;;
        --skip-system)
            SKIP_SYSTEM=true
            shift
            ;;
        --skip-config)
            SKIP_CONFIG=true
            shift
            ;;
        --skip-ide)
            SKIP_IDE=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_banner

    check_system

    if [[ "$SKIP_RUST" != true ]]; then
        install_rust
    fi

    if [[ "$SKIP_TOOLS" != true ]]; then
        install_dev_tools
    fi

    if [[ "$SKIP_SYSTEM" != true ]]; then
        install_system_deps
    fi

    if [[ "$SKIP_CONFIG" != true ]]; then
        setup_dev_config
    fi

    if [[ "$SKIP_IDE" != true ]]; then
        setup_ide_config
    fi

    if [[ "$SKIP_VERIFY" != true ]]; then
        verify_setup
    fi

    log_info ""
    log_info "🎉 BitNet-rs development environment setup complete!"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Run 'cargo test --workspace --features cpu' to run tests"
    log_info "  2. Run 'cargo doc --workspace --features cpu --open' to view documentation"
    log_info "  3. For cross-validation setup: './scripts/dev-crossval.sh'"
    log_info "  4. Start coding! 🦀"
    log_info ""
    log_info "Useful commands:"
    log_info "  cargo check-all     # Check all code"
    log_info "  cargo test-all      # Run all tests"
    log_info "  cargo quality       # Run quality checks"
    log_info "  cargo coverage      # Generate coverage report"
    log_info ""
}

# Run main function
main "$@"
