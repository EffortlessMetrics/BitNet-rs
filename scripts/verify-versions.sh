#!/bin/bash
# Version Verification Script for BitNet.rs
# Ensures all crates have consistent versions before publishing

set -e

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

# Get version from Cargo.toml
get_version() {
    local cargo_toml="$1"
    if [[ -f "$cargo_toml" ]]; then
        grep '^version = ' "$cargo_toml" | head -1 | sed 's/version = "\(.*\)"/\1/'
    else
        echo "FILE_NOT_FOUND"
    fi
}

# Main verification
main() {
    log_info "🔍 Verifying BitNet.rs crate versions..."
    
    # Get root version
    local root_version
    root_version=$(get_version "Cargo.toml")
    
    if [[ "$root_version" == "FILE_NOT_FOUND" ]]; then
        log_error "Root Cargo.toml not found"
        exit 1
    fi
    
    log_info "Root crate version: $root_version"
    
    # List of crates to check
    local crates=(
        "crates/bitnet-common"
        "crates/bitnet-models"
        "crates/bitnet-quantization"
        "crates/bitnet-kernels"
        "crates/bitnet-inference"
        "crates/bitnet-tokenizers"
        "crates/bitnet-server"
        "crates/bitnet-cli"
        "crates/bitnet-ffi"
        "crates/bitnet-py"
        "crates/bitnet-wasm"
        "crates/bitnet-sys"
    )
    
    local errors=0
    local warnings=0
    
    # Check each crate
    for crate_dir in "${crates[@]}"; do
        local cargo_toml="$crate_dir/Cargo.toml"
        local crate_name=$(basename "$crate_dir")
        
        if [[ -f "$cargo_toml" ]]; then
            local crate_version
            crate_version=$(get_version "$cargo_toml")
            
            if [[ "$crate_version" == "$root_version" ]]; then
                log_success "$crate_name: $crate_version ✓"
            else
                log_error "$crate_name: $crate_version (expected: $root_version) ✗"
                ((errors++))
            fi
            
            # Check internal dependencies
            local internal_deps
            internal_deps=$(grep -E '^bitnet-[a-z-]+ = ' "$cargo_toml" | grep 'version = ' || true)
            
            if [[ -n "$internal_deps" ]]; then
                while IFS= read -r dep_line; do
                    local dep_version
                    dep_version=$(echo "$dep_line" | sed 's/.*version = "\([^"]*\)".*/\1/')
                    
                    if [[ "$dep_version" != "$root_version" ]]; then
                        local dep_name
                        dep_name=$(echo "$dep_line" | sed 's/^\([^=]*\) = .*/\1/')
                        log_warning "$crate_name depends on $dep_name version $dep_version (expected: $root_version)"
                        ((warnings++))
                    fi
                done <<< "$internal_deps"
            fi
        else
            log_warning "$crate_name: Cargo.toml not found"
            ((warnings++))
        fi
    done
    
    # Check workspace dependencies
    log_info "Checking workspace dependencies..."
    local workspace_version
    workspace_version=$(grep -A 20 '^\[workspace\.package\]' Cargo.toml | grep '^version = ' | head -1 | sed 's/version = "\(.*\)"/\1/')
    
    if [[ "$workspace_version" == "$root_version" ]]; then
        log_success "Workspace version: $workspace_version ✓"
    else
        log_error "Workspace version: $workspace_version (expected: $root_version) ✗"
        ((errors++))
    fi
    
    # Summary
    echo
    log_info "Verification Summary:"
    log_info "  Root version: $root_version"
    log_info "  Crates checked: ${#crates[@]}"
    
    if [[ $errors -eq 0 ]]; then
        log_success "  Errors: $errors ✓"
    else
        log_error "  Errors: $errors ✗"
    fi
    
    if [[ $warnings -eq 0 ]]; then
        log_success "  Warnings: $warnings ✓"
    else
        log_warning "  Warnings: $warnings ⚠"
    fi
    
    # Exit with appropriate code
    if [[ $errors -gt 0 ]]; then
        echo
        log_error "Version verification failed! Please fix version mismatches before publishing."
        exit 1
    elif [[ $warnings -gt 0 ]]; then
        echo
        log_warning "Version verification completed with warnings."
        log_info "Consider reviewing dependency versions for consistency."
        exit 0
    else
        echo
        log_success "🎉 All versions are consistent! Ready for publishing."
        exit 0
    fi
}

# Show help
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    cat << EOF
BitNet.rs Version Verification Script

This script verifies that all crates in the workspace have consistent versions
before publishing to crates.io.

USAGE:
    $0

The script will:
1. Check that all crate versions match the root version
2. Verify workspace.package version consistency
3. Check internal dependency versions
4. Report any mismatches or inconsistencies

EXIT CODES:
    0 - All versions are consistent
    1 - Version mismatches found (errors)

For more information, visit: https://github.com/microsoft/BitNet
EOF
    exit 0
fi

# Run main function
main "$@"