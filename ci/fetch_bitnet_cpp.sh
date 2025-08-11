#!/bin/bash
# Fetch and build the external BitNet C++ implementation
# This script downloads Microsoft's official BitNet.cpp for cross-validation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
BITNET_CPP_REPO="https://github.com/microsoft/BitNet.git"
BITNET_CPP_TAG="${BITNET_CPP_TAG:-v1.0.0}"  # Default version, can be overridden
CACHE_DIR="${BITNET_CPP_PATH:-$HOME/.cache/bitnet_cpp}"
BUILD_DIR="$CACHE_DIR/build"

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

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    if ! command -v git >/dev/null 2>&1; then
        missing_deps+=("git")
    fi
    
    if ! command -v cmake >/dev/null 2>&1; then
        missing_deps+=("cmake")
    fi
    
    if ! command -v make >/dev/null 2>&1; then
        missing_deps+=("make")
    fi
    
    # Check for C++ compiler
    if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
        missing_deps+=("g++ or clang++")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_error "Please install them and try again:"
        log_error "  Ubuntu/Debian: apt install git cmake build-essential"
        log_error "  macOS: xcode-select --install && brew install cmake"
        log_error "  Windows: Install Visual Studio with C++ tools and CMake"
        exit 1
    fi
}

# Verify checksum of downloaded content
verify_checksum() {
    local target_dir="$1"
    local expected_file="$SCRIPT_DIR/bitnet_cpp_checksums.txt"
    
    if [[ ! -f "$expected_file" ]]; then
        log_warn "No checksum file found at $expected_file"
        log_warn "Skipping checksum verification (not recommended for production)"
        return 0
    fi
    
    log_info "Verifying checksums..."
    
    cd "$target_dir"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c "$expected_file"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -c "$expected_file"
    else
        log_warn "No checksum utility found (sha256sum or shasum)"
        log_warn "Skipping checksum verification"
        return 0
    fi
    
    log_info "Checksum verification passed"
}

# Clone or update the repository
fetch_source() {
    log_info "Fetching BitNet C++ implementation..."
    log_info "Repository: $BITNET_CPP_REPO"
    log_info "Tag/Version: $BITNET_CPP_TAG"
    log_info "Cache directory: $CACHE_DIR"
    
    if [[ -d "$CACHE_DIR/.git" ]]; then
        log_info "Existing repository found, updating..."
        cd "$CACHE_DIR"
        
        # Fetch latest changes
        git fetch origin
        
        # Check if we're already on the right tag
        current_tag=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
        if [[ "$current_tag" == "$BITNET_CPP_TAG" ]]; then
            log_info "Already on correct tag: $BITNET_CPP_TAG"
            return 0
        fi
        
        # Clean any local changes
        git reset --hard
        git clean -fd
        
        # Checkout the specified tag
        git checkout "$BITNET_CPP_TAG"
    else
        log_info "Cloning fresh repository..."
        
        # Create cache directory
        mkdir -p "$(dirname "$CACHE_DIR")"
        
        # Clone the repository
        git clone --depth 1 --branch "$BITNET_CPP_TAG" "$BITNET_CPP_REPO" "$CACHE_DIR"
        cd "$CACHE_DIR"
    fi
    
    log_info "Source code fetched successfully"
    
    # Verify checksum if available
    # verify_checksum "$CACHE_DIR"
}

# Build the C++ implementation
build_cpp() {
    log_info "Building BitNet C++ implementation..."
    
    cd "$CACHE_DIR"
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure with CMake
    log_info "Configuring build with CMake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install"
    
    # Build
    log_info "Building (this may take a few minutes)..."
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    # Install to local directory
    log_info "Installing to local directory..."
    make install
    
    log_info "Build completed successfully"
}

# Apply patches if any exist
apply_patches() {
    log_info "Checking for patches to apply..."
    
    if [[ -x "$SCRIPT_DIR/apply_patches.sh" ]]; then
        log_info "Applying patches..."
        "$SCRIPT_DIR/apply_patches.sh"
    else
        log_info "No patch application script found - using C++ implementation as-is"
    fi
}

# Validate the build
validate_build() {
    log_info "Validating build..."
    
    local lib_dir="$BUILD_DIR/install/lib"
    local include_dir="$BUILD_DIR/install/include"
    
    # Check for expected files
    if [[ ! -d "$lib_dir" ]]; then
        log_error "Library directory not found: $lib_dir"
        return 1
    fi
    
    if [[ ! -d "$include_dir" ]]; then
        log_error "Include directory not found: $include_dir"
        return 1
    fi
    
    # Look for library files
    local lib_files=($(find "$lib_dir" -name "*.so" -o -name "*.dylib" -o -name "*.dll" 2>/dev/null))
    if [[ ${#lib_files[@]} -eq 0 ]]; then
        log_warn "No shared libraries found in $lib_dir"
        log_warn "This may be expected if only static libraries were built"
    else
        log_info "Found ${#lib_files[@]} library file(s)"
    fi
    
    # Look for header files
    local header_files=($(find "$include_dir" -name "*.h" -o -name "*.hpp" 2>/dev/null))
    if [[ ${#header_files[@]} -eq 0 ]]; then
        log_error "No header files found in $include_dir"
        return 1
    else
        log_info "Found ${#header_files[@]} header file(s)"
    fi
    
    log_info "Build validation passed"
}

# Create environment setup script
create_env_script() {
    local env_script="$CACHE_DIR/setup_env.sh"
    
    log_info "Creating environment setup script: $env_script"
    
    cat > "$env_script" << EOF
#!/bin/bash
# Environment setup for BitNet C++ cross-validation
# Source this file to set up environment variables

export BITNET_CPP_PATH="$CACHE_DIR"
export BITNET_CPP_LIB_PATH="$BUILD_DIR/install/lib"
export BITNET_CPP_INCLUDE_PATH="$BUILD_DIR/install/include"

# Add to library path
if [[ "\$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="\$BITNET_CPP_LIB_PATH:\$DYLD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="\$BITNET_CPP_LIB_PATH:\$LD_LIBRARY_PATH"
fi

# Add to pkg-config path if it exists
if [[ -d "\$BITNET_CPP_LIB_PATH/pkgconfig" ]]; then
    export PKG_CONFIG_PATH="\$BITNET_CPP_LIB_PATH/pkgconfig:\$PKG_CONFIG_PATH"
fi

echo "BitNet C++ environment configured:"
echo "  Path: \$BITNET_CPP_PATH"
echo "  Libraries: \$BITNET_CPP_LIB_PATH"
echo "  Headers: \$BITNET_CPP_INCLUDE_PATH"
EOF
    
    chmod +x "$env_script"
}

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Fetch and build the external BitNet C++ implementation for cross-validation.

OPTIONS:
    -t, --tag TAG       Specify BitNet.cpp tag/version (default: $BITNET_CPP_TAG)
    -p, --path PATH     Specify cache directory (default: $CACHE_DIR)
    -f, --force         Force rebuild even if already built
    -c, --clean         Clean build directory before building
    -h, --help          Show this help message

ENVIRONMENT VARIABLES:
    BITNET_CPP_TAG      Override default tag/version
    BITNET_CPP_PATH     Override default cache directory

EXAMPLES:
    $0                          # Use defaults
    $0 --tag v1.1.0            # Use specific version
    $0 --force                  # Force rebuild
    $0 --clean --force          # Clean rebuild

After successful build, source the environment setup:
    source $CACHE_DIR/setup_env.sh
EOF
}

# Parse command line arguments
FORCE_BUILD=false
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            BITNET_CPP_TAG="$2"
            shift 2
            ;;
        -p|--path)
            CACHE_DIR="$2"
            BUILD_DIR="$CACHE_DIR/build"
            shift 2
            ;;
        -f|--force)
            FORCE_BUILD=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
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

# Main execution
main() {
    log_info "BitNet C++ Fetch and Build Script"
    log_info "=================================="
    
    # Check if already built and not forcing rebuild
    if [[ -d "$BUILD_DIR" && -f "$BUILD_DIR/install/lib" && "$FORCE_BUILD" != true ]]; then
        log_info "BitNet C++ already built at $CACHE_DIR"
        log_info "Use --force to rebuild or --clean --force for clean rebuild"
        log_info "To use: source $CACHE_DIR/setup_env.sh"
        exit 0
    fi
    
    # Check dependencies
    check_dependencies
    
    # Clean if requested
    if [[ "$CLEAN_BUILD" == true && -d "$BUILD_DIR" ]]; then
        log_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi
    
    # Fetch source code
    fetch_source
    
    # Apply patches
    apply_patches
    
    # Build
    build_cpp
    
    # Validate
    validate_build
    
    # Create environment script
    create_env_script
    
    log_info "BitNet C++ setup completed successfully!"
    log_info ""
    log_info "To use in your shell:"
    log_info "  source $CACHE_DIR/setup_env.sh"
    log_info ""
    log_info "To use in Rust cross-validation:"
    log_info "  export BITNET_CPP_PATH=$CACHE_DIR"
    log_info "  cargo test --features crossval"
    log_info ""
    log_info "Cache location: $CACHE_DIR"
    log_info "Build artifacts: $BUILD_DIR"
}

# Run main function
main "$@"