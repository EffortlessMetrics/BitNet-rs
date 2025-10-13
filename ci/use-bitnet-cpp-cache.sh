#!/bin/bash
# Script to use cached BitNet.cpp libraries in CI
# This dramatically reduces CI build time from ~7min to <1min

set -e

# Configuration
REGISTRY="ghcr.io"
IMAGE_NAME="microsoft/bitnet/bitnet-cpp-cache"
CACHE_DIR="${BITNET_CPP_CACHE_DIR:-$HOME/.cache/bitnet_cpp}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"

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

# Get the pinned BitNet.cpp version
get_pinned_version() {
    if [[ -f "ci/fetch_bitnet_cpp.sh" ]]; then
        grep -o 'BITNET_CPP_TAG="[^"]*"' ci/fetch_bitnet_cpp.sh | cut -d'"' -f2 || echo "main"
    else
        echo "main"
    fi
}

# Get platform architecture
get_platform_arch() {
    case "$(uname -m)" in
        x86_64) echo "amd64" ;;
        aarch64|arm64) echo "arm64" ;;
        *) echo "amd64" ;; # Default fallback
    esac
}

# Generate cache key
generate_cache_key() {
    local version="$1"
    local platform="$2"

    # Get commit SHA for the version
    if [[ "$version" != "main" && "$version" != "latest" ]]; then
        local commit_sha
        commit_sha=$(curl -s "https://api.github.com/repos/microsoft/BitNet/git/refs/tags/$version" | jq -r '.object.sha // "unknown"' 2>/dev/null || echo "unknown")
    else
        local commit_sha
        commit_sha=$(curl -s "https://api.github.com/repos/microsoft/BitNet/commits/main" | jq -r '.sha // "unknown"' 2>/dev/null || echo "unknown")
    fi

    echo "${version}-${commit_sha:0:8}-${platform}"
}

# Check if cache exists locally
check_local_cache() {
    local cache_key="$1"
    local cache_path="$CACHE_DIR/$cache_key"

    if [[ -d "$cache_path" && -f "$cache_path/cache-metadata.json" ]]; then
        log_info "Found local cache at $cache_path"
        return 0
    else
        log_info "No local cache found for $cache_key"
        return 1
    fi
}

# Pull cache from container registry
pull_cache_from_registry() {
    local cache_key="$1"
    local cache_path="$CACHE_DIR/$cache_key"

    log_info "Pulling cache from container registry..."

    local image_tag="$REGISTRY/$IMAGE_NAME:$cache_key"

    # Check if Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not available, cannot pull cache"
        return 1
    fi

    # Try to pull the cache image
    if docker pull "$image_tag" >/dev/null 2>&1; then
        log_success "Pulled cache image: $image_tag"

        # Extract cache to local directory
        mkdir -p "$cache_path"

        # Run container and copy files
        local container_id
        container_id=$(docker create "$image_tag")

        if docker cp "$container_id:/opt/bitnet_cpp/." "$cache_path/"; then
            log_success "Extracted cache to $cache_path"
            docker rm "$container_id" >/dev/null 2>&1
            return 0
        else
            log_error "Failed to extract cache from container"
            docker rm "$container_id" >/dev/null 2>&1
            return 1
        fi
    else
        log_warning "Cache image not found in registry: $image_tag"
        return 1
    fi
}

# Use cached libraries
use_cached_libraries() {
    local cache_path="$1"

    log_info "Setting up cached BitNet.cpp libraries..."

    # Set environment variables
    export BITNET_CPP_ROOT="$cache_path"
    export BITNET_CPP_INCLUDE_DIR="$cache_path/include"
    export BITNET_CPP_LIB_DIR="$cache_path/lib"
    export LD_LIBRARY_PATH="$cache_path/lib:${LD_LIBRARY_PATH:-}"
    export PKG_CONFIG_PATH="$cache_path/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

    # Verify cache integrity
    if [[ -f "$cache_path/cache-metadata.json" ]]; then
        log_info "Cache metadata:"
        cat "$cache_path/cache-metadata.json" | jq . 2>/dev/null || cat "$cache_path/cache-metadata.json"
    fi

    # Verify libraries exist
    if [[ -d "$cache_path/lib" && -d "$cache_path/include" ]]; then
        log_success "Cache libraries are ready:"
        log_info "  Libraries: $(find "$cache_path/lib" -name '*.so*' -o -name '*.a' | wc -l) files"
        log_info "  Headers: $(find "$cache_path/include" -name '*.h' -o -name '*.hpp' | wc -l) files"

        # Create a marker file to indicate cache is active
        echo "BITNET_CPP_CACHED=true" > "$cache_path/.cache-active"
        echo "BITNET_CPP_ROOT=$cache_path" >> "$cache_path/.cache-active"

        return 0
    else
        log_error "Cache verification failed: missing libraries or headers"
        return 1
    fi
}

# Fallback to building from source
fallback_to_source_build() {
    log_warning "Falling back to building BitNet.cpp from source..."

    if [[ -f "ci/fetch_bitnet_cpp.sh" ]]; then
        log_info "Running fetch_bitnet_cpp.sh..."
        bash ci/fetch_bitnet_cpp.sh
    else
        log_error "No fallback build script found"
        return 1
    fi
}

# Main function
main() {
    log_info "🚀 BitNet.cpp Cache Manager"

    # Get version and platform
    local version
    version=$(get_pinned_version)
    local platform
    platform=$(get_platform_arch)

    log_info "Target version: $version"
    log_info "Platform: $platform"

    # Generate cache key
    local cache_key
    cache_key=$(generate_cache_key "$version" "$platform")
    log_info "Cache key: $cache_key"

    local cache_path="$CACHE_DIR/$cache_key"

    # Skip cache if force rebuild is requested
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        log_warning "Force rebuild requested, skipping cache"
        fallback_to_source_build
        return $?
    fi

    # Check local cache first
    if check_local_cache "$cache_key"; then
        if use_cached_libraries "$cache_path"; then
            log_success "✅ Using local cached BitNet.cpp libraries"
            return 0
        else
            log_warning "Local cache is corrupted, removing..."
            rm -rf "$cache_path"
        fi
    fi

    # Try to pull from registry
    if pull_cache_from_registry "$cache_key"; then
        if use_cached_libraries "$cache_path"; then
            log_success "✅ Using cached BitNet.cpp libraries from registry"
            return 0
        else
            log_warning "Registry cache is corrupted, removing..."
            rm -rf "$cache_path"
        fi
    fi

    # Fallback to source build
    log_warning "No usable cache found, building from source..."
    fallback_to_source_build

    # Cache the built libraries for next time
    if [[ -d "$HOME/.cache/bitnet_cpp/build" ]]; then
        log_info "Caching built libraries for future use..."
        mkdir -p "$cache_path"
        cp -r "$HOME/.cache/bitnet_cpp/build/"* "$cache_path/" 2>/dev/null || true

        # Create metadata
        echo "{
          \"version\": \"$version\",
          \"platform\": \"$platform\",
          \"build_date\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
          \"source\": \"local_build\",
          \"cache_key\": \"$cache_key\"
        }" > "$cache_path/cache-metadata.json"
    fi
}

# Show help
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    cat << EOF
BitNet.cpp Cache Manager

This script manages cached BitNet.cpp libraries to speed up CI builds.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --help, -h          Show this help message

ENVIRONMENT VARIABLES:
    BITNET_CPP_CACHE_DIR    Cache directory (default: ~/.cache/bitnet_cpp)
    FORCE_REBUILD           Force rebuild from source (default: false)

The script will:
1. Check for local cached libraries
2. Pull from GitHub Container Registry if not found locally
3. Fall back to building from source if no cache is available
4. Set up environment variables for using the cached libraries

EXIT CODES:
    0 - Success (cache used or source build completed)
    1 - Error occurred

For more information, visit: https://github.com/microsoft/BitNet
EOF
    exit 0
fi

# Run main function
main "$@"
