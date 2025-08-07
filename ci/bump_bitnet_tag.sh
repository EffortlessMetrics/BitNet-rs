#!/bin/bash
# Version management system for BitNet C++ dependency
# This script helps update the pinned version of the external BitNet.cpp implementation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
BITNET_CPP_REPO="https://github.com/microsoft/BitNet.git"
CACHE_DIR="${BITNET_CPP_PATH:-$HOME/.cache/bitnet_cpp}"
VERSION_FILE="$SCRIPT_DIR/bitnet_cpp_version.txt"
CHECKSUM_FILE="$SCRIPT_DIR/bitnet_cpp_checksums.txt"

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

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Version management for BitNet C++ dependency.

COMMANDS:
    current                 Show current pinned version
    list                   List available versions from upstream
    update TAG             Update to specific version/tag
    latest                 Update to latest release
    check                  Check if current version is up to date
    validate               Validate current version and checksums
    generate-checksums     Generate checksums for current version

OPTIONS:
    -f, --force            Force update even if version is current
    -y, --yes              Skip confirmation prompts
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message

EXAMPLES:
    $0 current                    # Show current version
    $0 list                       # List available versions
    $0 update v1.2.0             # Update to specific version
    $0 latest                     # Update to latest release
    $0 check                      # Check for updates
    $0 validate                   # Validate current setup
    $0 generate-checksums         # Generate new checksums

FILES:
    $VERSION_FILE    # Current pinned version
    $CHECKSUM_FILE   # Checksums for verification
EOF
}

# Get current pinned version
get_current_version() {
    if [[ -f "$VERSION_FILE" ]]; then
        cat "$VERSION_FILE"
    else
        echo "unknown"
    fi
}

# Set new version
set_version() {
    local new_version="$1"
    echo "$new_version" > "$VERSION_FILE"
    log_info "Updated version file to: $new_version"
}

# List available versions from upstream
list_versions() {
    log_info "Fetching available versions from upstream..."
    
    # Create temporary directory for listing
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    # Clone just to get tags
    git clone --bare --filter=blob:none "$BITNET_CPP_REPO" "$temp_dir/repo.git" >/dev/null 2>&1
    
    cd "$temp_dir/repo.git"
    
    log_info "Available versions:"
    git tag --sort=-version:refname | head -20
    
    log_info ""
    log_info "Latest release:"
    git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || echo "No semantic version tags found"
}

# Get latest release version
get_latest_version() {
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    git clone --bare --filter=blob:none "$BITNET_CPP_REPO" "$temp_dir/repo.git" >/dev/null 2>&1
    cd "$temp_dir/repo.git"
    
    # Try to find latest semantic version tag
    local latest=$(git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
    
    if [[ -z "$latest" ]]; then
        # Fall back to any tag
        latest=$(git tag --sort=-version:refname | head -1)
    fi
    
    echo "$latest"
}

# Check if version exists upstream
version_exists() {
    local version="$1"
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    git clone --bare --filter=blob:none "$BITNET_CPP_REPO" "$temp_dir/repo.git" >/dev/null 2>&1
    cd "$temp_dir/repo.git"
    
    git tag | grep -q "^$version$"
}

# Update to specific version
update_version() {
    local new_version="$1"
    local force="$2"
    local skip_confirm="$3"
    
    local current_version=$(get_current_version)
    
    log_info "Current version: $current_version"
    log_info "Target version: $new_version"
    
    # Check if version exists
    if ! version_exists "$new_version"; then
        log_error "Version '$new_version' does not exist upstream"
        log_info "Run '$0 list' to see available versions"
        exit 1
    fi
    
    # Check if already current
    if [[ "$current_version" == "$new_version" && "$force" != "true" ]]; then
        log_info "Already on version $new_version"
        log_info "Use --force to update anyway"
        exit 0
    fi
    
    # Confirm update
    if [[ "$skip_confirm" != "true" ]]; then
        echo -n "Update from $current_version to $new_version? [y/N] "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Update cancelled"
            exit 0
        fi
    fi
    
    # Update version file
    set_version "$new_version"
    
    # Clean existing cache to force re-download
    if [[ -d "$CACHE_DIR" ]]; then
        log_info "Cleaning existing cache..."
        rm -rf "$CACHE_DIR"
    fi
    
    # Fetch new version
    log_info "Fetching new version..."
    BITNET_CPP_TAG="$new_version" "$SCRIPT_DIR/fetch_bitnet_cpp.sh"
    
    # Generate new checksums
    log_info "Generating checksums for new version..."
    generate_checksums
    
    log_info "Successfully updated to version $new_version"
    log_warn "Remember to test cross-validation with the new version:"
    log_warn "  cargo test --features crossval"
}

# Check if current version is up to date
check_updates() {
    local current_version=$(get_current_version)
    local latest_version=$(get_latest_version)
    
    log_info "Current version: $current_version"
    log_info "Latest version: $latest_version"
    
    if [[ "$current_version" == "$latest_version" ]]; then
        log_info "✓ Up to date"
        return 0
    else
        log_warn "Update available: $current_version → $latest_version"
        log_info "Run '$0 update $latest_version' to update"
        return 1
    fi
}

# Validate current version and checksums
validate_version() {
    local current_version=$(get_current_version)
    
    log_info "Validating version: $current_version"
    
    # Check if cache exists
    if [[ ! -d "$CACHE_DIR" ]]; then
        log_error "Cache directory not found: $CACHE_DIR"
        log_info "Run '$SCRIPT_DIR/fetch_bitnet_cpp.sh' to download"
        return 1
    fi
    
    # Check git tag in cache
    cd "$CACHE_DIR"
    if [[ ! -d ".git" ]]; then
        log_error "Cache is not a git repository"
        return 1
    fi
    
    local actual_version=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
    if [[ "$actual_version" != "$current_version" ]]; then
        log_error "Version mismatch:"
        log_error "  Expected: $current_version"
        log_error "  Actual: $actual_version"
        return 1
    fi
    
    log_info "✓ Version matches: $current_version"
    
    # Validate checksums if available
    if [[ -f "$CHECKSUM_FILE" && -s "$CHECKSUM_FILE" ]]; then
        log_info "Validating checksums..."
        
        if command -v sha256sum >/dev/null 2>&1; then
            if sha256sum -c "$CHECKSUM_FILE" >/dev/null 2>&1; then
                log_info "✓ Checksums valid"
            else
                log_error "✗ Checksum validation failed"
                return 1
            fi
        elif command -v shasum >/dev/null 2>&1; then
            if shasum -a 256 -c "$CHECKSUM_FILE" >/dev/null 2>&1; then
                log_info "✓ Checksums valid"
            else
                log_error "✗ Checksum validation failed"
                return 1
            fi
        else
            log_warn "No checksum utility found - skipping validation"
        fi
    else
        log_warn "No checksums available for validation"
    fi
    
    log_info "✓ Validation passed"
    return 0
}

# Generate checksums for current version
generate_checksums() {
    log_info "Generating checksums..."
    
    if [[ ! -d "$CACHE_DIR" ]]; then
        log_error "Cache directory not found: $CACHE_DIR"
        log_info "Run '$SCRIPT_DIR/fetch_bitnet_cpp.sh' first"
        return 1
    fi
    
    cd "$CACHE_DIR"
    
    # Generate checksums for key files
    local temp_checksums=$(mktemp)
    
    # Find important files to checksum
    find . -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "CMakeLists.txt" | \
        sort | \
        xargs sha256sum > "$temp_checksums" 2>/dev/null || \
        xargs shasum -a 256 > "$temp_checksums" 2>/dev/null
    
    if [[ -s "$temp_checksums" ]]; then
        # Add header to checksum file
        cat > "$CHECKSUM_FILE" << EOF
# SHA256 checksums for BitNet C++ implementation
# Generated on $(date)
# Version: $(get_current_version)

EOF
        cat "$temp_checksums" >> "$CHECKSUM_FILE"
        rm "$temp_checksums"
        
        local checksum_count=$(grep -v '^#' "$CHECKSUM_FILE" | wc -l)
        log_info "Generated checksums for $checksum_count files"
        log_info "Checksums saved to: $CHECKSUM_FILE"
    else
        log_error "Failed to generate checksums"
        rm -f "$temp_checksums"
        return 1
    fi
}

# Parse command line arguments
COMMAND=""
FORCE=false
SKIP_CONFIRM=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        current|list|latest|check|validate|generate-checksums)
            COMMAND="$1"
            shift
            ;;
        update)
            COMMAND="update"
            if [[ $# -lt 2 ]]; then
                log_error "update command requires a version argument"
                print_usage
                exit 1
            fi
            UPDATE_VERSION="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
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

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Execute command
case "$COMMAND" in
    current)
        echo "Current version: $(get_current_version)"
        ;;
    list)
        list_versions
        ;;
    update)
        update_version "$UPDATE_VERSION" "$FORCE" "$SKIP_CONFIRM"
        ;;
    latest)
        latest_version=$(get_latest_version)
        update_version "$latest_version" "$FORCE" "$SKIP_CONFIRM"
        ;;
    check)
        check_updates
        ;;
    validate)
        validate_version
        ;;
    generate-checksums)
        generate_checksums
        ;;
    "")
        log_error "No command specified"
        print_usage
        exit 1
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac