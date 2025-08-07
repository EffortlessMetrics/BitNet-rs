#!/bin/bash
# BitNet.rs Cross-Validation Development Setup
# One-liner setup for cross-validation development

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

# Print banner
print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
   ____                    __     __    _ _     _       _   _             
  / ___|_ __ ___  ___ ___  \ \   / /_ _| (_) __| | __ _| |_(_) ___  _ __  
 | |   | '__/ _ \/ __/ __|  \ \ / / _` | | |/ _` |/ _` | __| |/ _ \| '_ \ 
 | |___| | | (_) \__ \__ \   \ V / (_| | | | (_| | (_| | |_| | (_) | | | |
  \____|_|  \___/|___/___/    \_/ \__,_|_|_|\__,_|\__,_|\__|_|\___/|_| |_|
  
  Quick Setup for Cross-Validation Development
EOF
    echo -e "${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "Linux";;
        Darwin*)    echo "macOS";;
        CYGWIN*|MINGW*|MSYS*) echo "Windows";;
        *)          echo "Unknown";;
    esac
}

# Install system dependencies
install_system_deps() {
    local os=$(detect_os)
    log_info "Installing system dependencies for $os..."
    
    case "$os" in
        Linux)
            if command_exists apt; then
                sudo apt update
                sudo apt install -y clang libclang-dev cmake build-essential git
            elif command_exists yum; then
                sudo yum groupinstall -y "Development Tools"
                sudo yum install -y clang clang-devel cmake git
            elif command_exists pacman; then
                sudo pacman -S --needed clang cmake base-devel git
            else
                log_error "Unsupported Linux distribution"
                log_info "Please install: clang, cmake, build tools, git"
                exit 1
            fi
            ;;
        macOS)
            if ! command_exists clang; then
                log_info "Installing Xcode Command Line Tools..."
                xcode-select --install
            fi
            if command_exists brew; then
                brew install cmake
            else
                log_warn "Homebrew not found. Please install CMake manually."
            fi
            ;;
        Windows)
            log_error "Windows setup requires manual installation:"
            log_info "  1. Install Visual Studio with C++ tools"
            log_info "  2. Install CMake from https://cmake.org/"
            log_info "  3. Install LLVM/Clang from https://llvm.org/"
            log_info "  4. Install Git from https://git-scm.com/"
            exit 1
            ;;
        *)
            log_error "Unsupported operating system: $os"
            exit 1
            ;;
    esac
    
    log_info "✅ System dependencies installed"
}

# Setup C++ implementation
setup_cpp_impl() {
    log_info "Setting up C++ implementation..."
    
    # Check if fetch script exists
    if [[ ! -f "ci/fetch_bitnet_cpp.sh" ]]; then
        log_error "fetch_bitnet_cpp.sh not found. Are you in the BitNet.rs repository root?"
        exit 1
    fi
    
    # Make script executable
    chmod +x ci/fetch_bitnet_cpp.sh
    
    # Run the fetch script
    log_info "Downloading and building C++ implementation..."
    ./ci/fetch_bitnet_cpp.sh
    
    # Source environment if available
    local env_script="$HOME/.cache/bitnet_cpp/setup_env.sh"
    if [[ -f "$env_script" ]]; then
        log_info "Setting up environment variables..."
        source "$env_script"
        
        # Add to current shell profile
        local shell_profile=""
        if [[ -n "${BASH_VERSION:-}" ]]; then
            shell_profile="$HOME/.bashrc"
        elif [[ -n "${ZSH_VERSION:-}" ]]; then
            shell_profile="$HOME/.zshrc"
        fi
        
        if [[ -n "$shell_profile" && -f "$shell_profile" ]]; then
            if ! grep -q "bitnet_cpp/setup_env.sh" "$shell_profile"; then
                echo "" >> "$shell_profile"
                echo "# BitNet C++ cross-validation environment" >> "$shell_profile"
                echo "if [[ -f \"$env_script\" ]]; then" >> "$shell_profile"
                echo "    source \"$env_script\"" >> "$shell_profile"
                echo "fi" >> "$shell_profile"
                log_info "Added environment setup to $shell_profile"
            fi
        fi
    fi
    
    log_info "✅ C++ implementation setup complete"
}

# Generate test fixtures
generate_fixtures() {
    log_info "Generating test fixtures..."
    
    # Check if xtask is available
    if ! cargo xtask --help >/dev/null 2>&1; then
        log_warn "xtask not available, skipping fixture generation"
        return
    fi
    
    # Generate deterministic fixtures
    cargo xtask gen-fixtures --deterministic --prompts 5
    
    log_info "✅ Test fixtures generated"
}

# Test cross-validation setup
test_setup() {
    log_info "Testing cross-validation setup..."
    
    # Test that crossval feature compiles
    log_info "Testing compilation with crossval feature..."
    if cargo check --features crossval; then
        log_info "✅ Cross-validation compilation successful"
    else
        log_error "❌ Cross-validation compilation failed"
        return 1
    fi
    
    # Test basic cross-validation functionality
    log_info "Testing basic cross-validation functionality..."
    if cargo test --features crossval cpp_availability --lib; then
        log_info "✅ Basic cross-validation test passed"
    else
        log_warn "❌ Basic cross-validation test failed (C++ implementation may not be ready)"
    fi
    
    # Test fixture loading
    log_info "Testing fixture loading..."
    if cargo test --features crossval test_fixture_compatibility --lib; then
        log_info "✅ Fixture loading test passed"
    else
        log_warn "❌ Fixture loading test failed (fixtures may not be available)"
    fi
    
    log_info "✅ Cross-validation setup testing complete"
}

# Create development shortcuts
create_shortcuts() {
    log_info "Creating development shortcuts..."
    
    # Update .cargo/config.toml with crossval aliases
    local cargo_config=".cargo/config.toml"
    if [[ -f "$cargo_config" ]]; then
        # Check if crossval aliases already exist
        if ! grep -q "crossval.*=" "$cargo_config"; then
            log_info "Adding crossval aliases to .cargo/config.toml..."
            
            # Add crossval aliases to existing config
            cat >> "$cargo_config" << 'EOF'

# Cross-validation aliases
[alias.crossval-aliases]
crossval-test = "test --features crossval"
crossval-bench = "bench --features crossval"
crossval-check = "check --features crossval"
crossval-doc = "doc --features crossval --no-deps"
crossval-fixtures = "xtask gen-fixtures --deterministic"
crossval-validate = "xtask validate-fixtures"
crossval-clean = "xtask clean-fixtures"
EOF
        fi
    fi
    
    # Create a quick test script
    cat > scripts/quick-crossval-test.sh << 'EOF'
#!/bin/bash
# Quick cross-validation test script

set -e

echo "🧪 Running quick cross-validation tests..."

# Check compilation
echo "1. Testing compilation..."
cargo check --features crossval

# Run basic tests
echo "2. Running basic tests..."
cargo test --features crossval --lib -- --test-threads=1

# Generate fixtures if needed
echo "3. Checking fixtures..."
if [[ ! -f "crossval/fixtures/minimal_test.json" ]]; then
    echo "   Generating fixtures..."
    cargo xtask gen-fixtures --deterministic --prompts 3
fi

# Run a quick benchmark
echo "4. Running quick benchmark..."
cargo bench --features crossval -- --sample-size 10

echo "✅ Quick cross-validation test complete!"
EOF
    chmod +x scripts/quick-crossval-test.sh
    
    log_info "✅ Development shortcuts created"
}

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Quick setup for BitNet.rs cross-validation development.

This script will:
1. Install required system dependencies (clang, cmake, etc.)
2. Download and build the C++ BitNet implementation
3. Generate test fixtures for cross-validation
4. Test the cross-validation setup
5. Create development shortcuts

OPTIONS:
    --skip-deps         Skip system dependency installation
    --skip-cpp          Skip C++ implementation setup
    --skip-fixtures     Skip test fixture generation
    --skip-test         Skip setup testing
    --skip-shortcuts    Skip development shortcuts creation
    --help              Show this help message

EXAMPLES:
    $0                  # Full setup
    $0 --skip-deps      # Skip system dependencies (if already installed)
    $0 --skip-test      # Skip testing (faster setup)

After setup, you can use:
    cargo test --features crossval          # Run cross-validation tests
    cargo bench --features crossval         # Run performance benchmarks
    ./scripts/quick-crossval-test.sh        # Quick test script
EOF
}

# Parse command line arguments
SKIP_DEPS=false
SKIP_CPP=false
SKIP_FIXTURES=false
SKIP_TEST=false
SKIP_SHORTCUTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-cpp)
            SKIP_CPP=true
            shift
            ;;
        --skip-fixtures)
            SKIP_FIXTURES=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --skip-shortcuts)
            SKIP_SHORTCUTS=true
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
    
    # Check if we're in the right directory
    if [[ ! -f "Cargo.toml" ]] || ! grep -q "bitnet" Cargo.toml; then
        log_error "This script must be run from the BitNet.rs repository root"
        exit 1
    fi
    
    # Check if basic Rust development is set up
    if ! command_exists cargo; then
        log_error "Rust/Cargo not found. Please run './scripts/dev-setup.sh' first"
        exit 1
    fi
    
    log_info "Setting up cross-validation development environment..."
    
    if [[ "$SKIP_DEPS" != true ]]; then
        install_system_deps
    fi
    
    if [[ "$SKIP_CPP" != true ]]; then
        setup_cpp_impl
    fi
    
    if [[ "$SKIP_FIXTURES" != true ]]; then
        generate_fixtures
    fi
    
    if [[ "$SKIP_TEST" != true ]]; then
        test_setup
    fi
    
    if [[ "$SKIP_SHORTCUTS" != true ]]; then
        create_shortcuts
    fi
    
    log_info ""
    log_info "🎉 Cross-validation development setup complete!"
    log_info ""
    log_info "You can now use:"
    log_info "  cargo test --features crossval          # Run cross-validation tests"
    log_info "  cargo bench --features crossval         # Run performance benchmarks"
    log_info "  ./scripts/quick-crossval-test.sh        # Quick test script"
    log_info ""
    log_info "Environment variables are set in your shell profile."
    log_info "Restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
    log_info ""
    log_info "Happy cross-validating! 🦀⚡"
}

# Run main function
main "$@"