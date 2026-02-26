#!/bin/bash
# BitNet-rs Installation Script
# This script installs the latest BitNet-rs binaries for Unix-like systems

set -e

# Configuration
REPO="microsoft/BitNet"
INSTALL_DIR="${BITNET_INSTALL_DIR:-$HOME/.local/bin}"
TEMP_DIR=$(mktemp -d)
GITHUB_API="https://api.github.com/repos/$REPO"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Cleanup function
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Help function
show_help() {
    cat << EOF
BitNet-rs Installation Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -d, --dir DIR       Installation directory (default: ~/.local/bin)
    -v, --version VER   Install specific version (default: latest)
    --cli-only          Install only the CLI tool
    --server-only       Install only the server
    --force             Force reinstallation even if already installed

EXAMPLES:
    $0                          # Install latest version to ~/.local/bin
    $0 -d /usr/local/bin        # Install to system directory
    $0 -v v1.0.0                # Install specific version
    $0 --cli-only               # Install only bitnet-cli

ENVIRONMENT VARIABLES:
    BITNET_INSTALL_DIR          Installation directory
    GITHUB_TOKEN                GitHub token for API access (optional)

For more information, visit: https://github.com/$REPO
EOF
}

# Parse command line arguments
INSTALL_CLI=true
INSTALL_SERVER=true
VERSION="latest"
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --cli-only)
            INSTALL_CLI=true
            INSTALL_SERVER=false
            shift
            ;;
        --server-only)
            INSTALL_CLI=false
            INSTALL_SERVER=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Detect platform and architecture
detect_platform() {
    local os arch

    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)

    case "$os" in
        linux*)
            case "$arch" in
                x86_64|amd64)
                    echo "x86_64-unknown-linux-gnu"
                    ;;
                aarch64|arm64)
                    echo "aarch64-unknown-linux-gnu"
                    ;;
                *)
                    log_error "Unsupported architecture: $arch"
                    exit 1
                    ;;
            esac
            ;;
        darwin*)
            case "$arch" in
                x86_64)
                    echo "x86_64-apple-darwin"
                    ;;
                arm64)
                    echo "aarch64-apple-darwin"
                    ;;
                *)
                    log_error "Unsupported architecture: $arch"
                    exit 1
                    ;;
            esac
            ;;
        *)
            log_error "Unsupported operating system: $os"
            log_info "This script supports Linux and macOS only."
            log_info "For Windows, please use the PowerShell script or download binaries manually."
            exit 1
            ;;
    esac
}

# Get latest release version
get_latest_version() {
    local api_url="$GITHUB_API/releases/latest"
    local auth_header=""

    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        auth_header="Authorization: token $GITHUB_TOKEN"
    fi

    if command -v curl >/dev/null 2>&1; then
        if [[ -n "$auth_header" ]]; then
            curl -s -H "$auth_header" "$api_url" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'
        else
            curl -s "$api_url" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'
        fi
    elif command -v wget >/dev/null 2>&1; then
        if [[ -n "$auth_header" ]]; then
            wget -q -O - --header="$auth_header" "$api_url" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'
        else
            wget -q -O - "$api_url" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'
        fi
    else
        log_error "Neither curl nor wget is available"
        exit 1
    fi
}

# Download and extract binary
download_and_install() {
    local platform version download_url filename

    platform=$(detect_platform)

    if [[ "$VERSION" == "latest" ]]; then
        version=$(get_latest_version)
        if [[ -z "$version" ]]; then
            log_error "Failed to get latest version"
            exit 1
        fi
    else
        version="$VERSION"
    fi

    log_info "Installing BitNet-rs $version for $platform"

    # Construct download URL
    filename="bitnet-${platform}.tar.gz"
    download_url="https://github.com/$REPO/releases/download/$version/$filename"

    log_info "Downloading from: $download_url"

    # Download
    cd "$TEMP_DIR"
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$filename" "$download_url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$filename" "$download_url"
    else
        log_error "Neither curl nor wget is available"
        exit 1
    fi

    # Verify download
    if [[ ! -f "$filename" ]]; then
        log_error "Download failed: $filename not found"
        exit 1
    fi

    # Extract
    log_info "Extracting binaries..."
    tar -xzf "$filename"

    # Create installation directory
    mkdir -p "$INSTALL_DIR"

    # Install binaries
    local installed_count=0

    if [[ "$INSTALL_CLI" == "true" ]] && [[ -f "bitnet-cli" ]]; then
        if [[ "$FORCE" == "true" ]] || [[ ! -f "$INSTALL_DIR/bitnet-cli" ]]; then
            cp bitnet-cli "$INSTALL_DIR/"
            chmod +x "$INSTALL_DIR/bitnet-cli"
            log_success "Installed bitnet-cli to $INSTALL_DIR"
            ((installed_count++))
        else
            log_warning "bitnet-cli already exists (use --force to overwrite)"
        fi
    fi

    if [[ "$INSTALL_SERVER" == "true" ]] && [[ -f "bitnet-server" ]]; then
        if [[ "$FORCE" == "true" ]] || [[ ! -f "$INSTALL_DIR/bitnet-server" ]]; then
            cp bitnet-server "$INSTALL_DIR/"
            chmod +x "$INSTALL_DIR/bitnet-server"
            log_success "Installed bitnet-server to $INSTALL_DIR"
            ((installed_count++))
        else
            log_warning "bitnet-server already exists (use --force to overwrite)"
        fi
    fi

    if [[ $installed_count -eq 0 ]]; then
        log_warning "No binaries were installed"
        exit 1
    fi
}

# Check if installation directory is in PATH
check_path() {
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        log_warning "Installation directory $INSTALL_DIR is not in your PATH"
        log_info "Add it to your PATH by adding this line to your shell profile:"
        log_info "  export PATH=\"$INSTALL_DIR:\$PATH\""

        # Suggest specific shell profile files
        if [[ -n "${BASH_VERSION:-}" ]]; then
            log_info "For bash, add to ~/.bashrc or ~/.bash_profile"
        elif [[ -n "${ZSH_VERSION:-}" ]]; then
            log_info "For zsh, add to ~/.zshrc"
        fi
    fi
}

# Verify installation
verify_installation() {
    local verified=0

    if [[ "$INSTALL_CLI" == "true" ]] && [[ -x "$INSTALL_DIR/bitnet-cli" ]]; then
        log_info "Verifying bitnet-cli installation..."
        if "$INSTALL_DIR/bitnet-cli" --version >/dev/null 2>&1; then
            log_success "bitnet-cli is working correctly"
            ((verified++))
        else
            log_error "bitnet-cli verification failed"
        fi
    fi

    if [[ "$INSTALL_SERVER" == "true" ]] && [[ -x "$INSTALL_DIR/bitnet-server" ]]; then
        log_info "Verifying bitnet-server installation..."
        if "$INSTALL_DIR/bitnet-server" --version >/dev/null 2>&1; then
            log_success "bitnet-server is working correctly"
            ((verified++))
        else
            log_error "bitnet-server verification failed"
        fi
    fi

    return $((verified > 0 ? 0 : 1))
}

# Main installation process
main() {
    log_info "🦀 BitNet-rs Installation Script"
    log_info "Installing to: $INSTALL_DIR"

    # Check prerequisites
    if ! command -v tar >/dev/null 2>&1; then
        log_error "tar is required but not installed"
        exit 1
    fi

    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        log_error "Either curl or wget is required but neither is installed"
        exit 1
    fi

    # Perform installation
    download_and_install

    # Verify installation
    if verify_installation; then
        log_success "🎉 BitNet-rs installation completed successfully!"

        # Show usage examples
        echo
        log_info "Quick start examples:"
        if [[ "$INSTALL_CLI" == "true" ]]; then
            echo "  $INSTALL_DIR/bitnet-cli --help"
        fi
        if [[ "$INSTALL_SERVER" == "true" ]]; then
            echo "  $INSTALL_DIR/bitnet-server --port 8080"
        fi

        # Check PATH
        check_path

        echo
        log_info "For documentation and examples, visit:"
        log_info "  https://github.com/$REPO"
    else
        log_error "Installation verification failed"
        exit 1
    fi
}

# Run main function
main "$@"
