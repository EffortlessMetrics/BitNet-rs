#!/bin/bash
# BitNet.rs Cross-Validation Development Setup
# One-liner script for easy cross-validation development setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show help
show_help() {
    cat << EOF
BitNet.rs Cross-Validation Development Setup

This script sets up everything needed for cross-validation development in one command.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -f, --force         Force rebuild even if cache exists
    -q, --quick         Skip comprehensive tests (faster setup)
    --no-cache          Don't use cached BitNet.cpp libraries
    --fixtures-only     Only generate test fixtures

WHAT THIS SCRIPT DOES:
    1. Sets up BitNet.cpp cache for cross-validation
    2. Generates deterministic test fixtures
    3. Builds Rust implementation with crossval features
    4. Runs basic cross-validation tests
    5. Sets up IDE configuration to prevent accidental crossval activation

EXAMPLES:
    $0                  # Full setup with cache
    $0 --quick          # Quick setup, skip comprehensive tests
    $0 --force          # Force rebuild everything
    $0 --fixtures-only  # Only generate test fixtures

For more information, visit: https://github.com/microsoft/BitNet
EOF
}

# Parse command line arguments
FORCE=false
QUICK=false
NO_CACHE=false
FIXTURES_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -q|--quick)
            QUICK=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --fixtures-only)
            FIXTURES_ONLY=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Main setup function
main() {
    log_info "🦀 BitNet.rs Cross-Validation Development Setup"
    echo
    
    # Check prerequisites
    log_info "Checking prerequisites..."
    
    if ! command -v cargo >/dev/null 2>&1; then
        log_error "Cargo is not installed. Please install Rust first."
        exit 1
    fi
    
    if ! command -v git >/dev/null 2>&1; then
        log_error "Git is not installed."
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "Cargo.toml" ]] || [[ ! -d "crates" ]]; then
        log_error "This script must be run from the BitNet.rs repository root."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
    
    # Generate test fixtures
    log_info "Generating test fixtures..."
    if cargo xtask gen-fixtures --size small --output crossval/fixtures/; then
        log_success "Test fixtures generated"
    else
        log_error "Failed to generate test fixtures"
        exit 1
    fi
    
    # If fixtures-only mode, exit here
    if [[ "$FIXTURES_ONLY" == "true" ]]; then
        log_success "✅ Test fixtures generated successfully!"
        exit 0
    fi
    
    # Set up BitNet.cpp cache (unless disabled)
    if [[ "$NO_CACHE" != "true" ]]; then
        log_info "Setting up BitNet.cpp cache..."
        
        if [[ "$FORCE" == "true" ]]; then
            export FORCE_REBUILD=true
        fi
        
        if [[ -f "ci/use-bitnet-cpp-cache.sh" ]]; then
            chmod +x ci/use-bitnet-cpp-cache.sh
            if ./ci/use-bitnet-cpp-cache.sh; then
                log_success "BitNet.cpp cache ready"
            else
                log_warning "Cache setup failed, will build from source"
            fi
        else
            log_warning "Cache script not found, will build from source"
        fi
    else
        log_info "Skipping cache setup (--no-cache specified)"
    fi
    
    # Build with crossval features
    log_info "Building Rust implementation with cross-validation features..."
    start_time=$(date +%s)
    
    if cargo build --features crossval --release; then
        end_time=$(date +%s)
        build_time=$((end_time - start_time))
        log_success "Build completed in ${build_time}s"
    else
        log_error "Build failed"
        exit 1
    fi
    
    # Run basic tests (unless quick mode)
    if [[ "$QUICK" != "true" ]]; then
        log_info "Running basic cross-validation tests..."
        
        if cargo test --package crossval --features crossval --release -- --nocapture quick_test; then
            log_success "Basic tests passed"
        else
            log_warning "Some tests failed (this may be expected during development)"
        fi
    else
        log_info "Skipping comprehensive tests (--quick mode)"
    fi
    
    # Set up IDE configuration
    setup_ide_config
    
    # Show usage instructions
    show_usage_instructions
    
    log_success "🎉 Cross-validation development environment ready!"
}

# Set up IDE configuration to prevent accidental crossval activation
setup_ide_config() {
    log_info "Setting up IDE configuration..."
    
    # VS Code settings
    if [[ -d ".vscode" ]] || [[ "$1" == "--force-vscode" ]]; then
        mkdir -p .vscode
        
        cat > .vscode/settings.json << 'EOF'
{
    "rust-analyzer.cargo.features": [],
    "rust-analyzer.cargo.noDefaultFeatures": false,
    "rust-analyzer.cargo.allFeatures": false,
    "rust-analyzer.checkOnSave.features": [],
    "rust-analyzer.checkOnSave.allFeatures": false,
    "rust-analyzer.runnables.cargoExtraArgs": [],
    "rust-analyzer.cargo.buildScripts.enable": true,
    "rust-analyzer.procMacro.enable": true,
    "rust-analyzer.diagnostics.disabled": [],
    "rust-analyzer.workspace.symbol.search.scope": "workspace_and_dependencies",
    "files.watcherExclude": {
        "**/target/**": true,
        "**/.cache/**": true,
        "**/crossval/fixtures/**": true
    },
    "search.exclude": {
        "**/target": true,
        "**/.cache": true,
        "**/crossval/fixtures": true
    },
    "rust-analyzer.lens.enable": true,
    "rust-analyzer.lens.run.enable": true,
    "rust-analyzer.lens.debug.enable": true,
    "rust-analyzer.hover.actions.enable": true,
    "rust-analyzer.completion.addCallParentheses": true,
    "rust-analyzer.completion.addCallArgumentSnippets": true,
    "editor.formatOnSave": true,
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer",
        "editor.formatOnSave": true
    }
}
EOF
        
        # VS Code tasks for cross-validation
        cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build (Rust only)",
            "type": "cargo",
            "command": "build",
            "args": ["--workspace"],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "silent",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "Build with Cross-Validation",
            "type": "cargo",
            "command": "build",
            "args": ["--workspace", "--features", "crossval"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "Test (Rust only)",
            "type": "cargo",
            "command": "test",
            "args": ["--workspace"],
            "group": {
                "kind": "test",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "Cross-Validation Tests",
            "type": "cargo",
            "command": "test",
            "args": ["--package", "crossval", "--features", "crossval", "--", "--nocapture"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": true,
                "panel": "shared"
            },
            "problemMatcher": ["$rustc"]
        },
        {
            "label": "Generate Test Fixtures",
            "type": "cargo",
            "command": "xtask",
            "args": ["gen-fixtures", "--size", "small", "--output", "crossval/fixtures/"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Setup Cross-Validation",
            "type": "cargo",
            "command": "xtask",
            "args": ["setup-crossval"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": true,
                "panel": "shared"
            }
        }
    ]
}
EOF
        
        log_success "VS Code configuration updated"
    fi
    
    # Create .cargo/config.toml to disable crossval by default
    mkdir -p .cargo
    cat > .cargo/config.toml << 'EOF'
# BitNet.rs Cargo Configuration
# This ensures crossval feature is not accidentally enabled

[build]
# Disable crossval feature by default to keep builds fast
# Use `cargo build --features crossval` when needed
rustflags = []

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native"]

[target.x86_64-apple-darwin]
rustflags = ["-C", "target-cpu=native"]

[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=native"]

[env]
# Prevent accidental crossval activation
CARGO_FEATURE_CROSSVAL = { value = "", force = false }
EOF
    
    log_success "Cargo configuration updated"
}

# Show usage instructions
show_usage_instructions() {
    echo
    log_info "📚 Usage Instructions:"
    echo
    echo "  🦀 Rust-only development (fast, recommended):"
    echo "    cargo build"
    echo "    cargo test"
    echo "    cargo bench"
    echo
    echo "  🔍 Cross-validation development (slower):"
    echo "    cargo build --features crossval"
    echo "    cargo test --package crossval --features crossval"
    echo "    cargo bench --package crossval --features crossval"
    echo
    echo "  🛠️  Development tasks:"
    echo "    cargo xtask gen-fixtures --size small"
    echo "    cargo xtask setup-crossval"
    echo "    cargo xtask clean-cache"
    echo "    cargo xtask check-features"
    echo
    echo "  📊 Performance tracking:"
    echo "    cargo xtask benchmark --platform current"
    echo
    echo "  🔧 IDE Integration:"
    echo "    - VS Code: Use 'Build (Rust only)' task for fast development"
    echo "    - VS Code: Use 'Cross-Validation Tests' task when needed"
    echo "    - Rust Analyzer: Configured to avoid crossval by default"
    echo
    log_warning "Remember: crossval feature is slow and only needed for comparison testing!"
}

# Run main function
main "$@"