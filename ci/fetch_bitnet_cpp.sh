#!/usr/bin/env bash
# Fetches and builds the Microsoft BitNet C++ implementation for cross-validation
# This provides the ground truth for our Rust implementation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
CACHE_DIR="${BITNET_CPP_DIR:-${BITNET_CPP_CACHE:-$HOME/.cache/bitnet_cpp}}"
REPO_URL="${BITNET_CPP_REPO:-https://github.com/microsoft/BitNet.git}"
# Use main branch since Microsoft BitNet doesn't use release tags
# This is the official Microsoft BitNet repository
DEFAULT_REV="main"  # Main branch with latest BitNet implementation
REV="${BITNET_CPP_REV:-$DEFAULT_REV}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Parse arguments
CLEAN=0
FORCE=0
CMAKE_FLAGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --rev|--tag)
            REV="$2"
            shift 2
            ;;
        --repo)
            REPO_URL="$2"
            shift 2
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --cmake-flags)
            CMAKE_FLAGS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--tag TAG] [--repo URL] [--clean] [--force] [--cmake-flags \"...\"]"
            echo "  --tag TAG    Git revision/branch to checkout (default: $DEFAULT_REV)"
            echo "  --repo URL   Git repository URL (default: $REPO_URL)"
            echo "  --clean      Clean build before compiling"
            echo "  --force      Force rebuild even if already built"
            echo "  --cmake-flags  Additional flags to pass to cmake"
            exit 0
            ;;
        --print-setup)
            # Just print setup instructions and exit
            PRINT_SETUP_ONLY=1
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If just printing setup instructions
if [[ "${PRINT_SETUP_ONLY:-0}" -eq 1 ]]; then
    # Determine library paths for environment setup
    if [[ "$OSTYPE" == "darwin"* ]]; then
        LD_VAR="DYLD_LIBRARY_PATH"
        LIB_EXT="dylib"
    else
        LD_VAR="LD_LIBRARY_PATH"
        LIB_EXT="so"
    fi
    
    BUILD_DIR="$CACHE_DIR/build"
    LIB_PATHS="$BUILD_DIR/3rdparty/llama.cpp/src:$BUILD_DIR/3rdparty/llama.cpp/ggml/src"
    
    log_info ""
    log_info "================================================================"
    log_info "BitNet C++ setup instructions:"
    log_info "================================================================"
    log_info "Repository:     $CACHE_DIR"
    log_info "Build:          $BUILD_DIR"
    log_info ""
    log_info "To use for cross-validation:"
    log_info ""
    log_info "  export BITNET_CPP_DIR='$CACHE_DIR'"
    log_info "  export ${LD_VAR}='${LIB_PATHS}:\$${LD_VAR}'"
    log_info "  export OMP_NUM_THREADS=1    # For determinism"
    log_info "  export GGML_NUM_THREADS=1"
    log_info ""
    log_info "Then run:"
    log_info "  cargo test --features crossval -p crossval"
    log_info "================================================================"
    exit 0
fi

# Check if already built (skip expensive operations if possible)
if [[ -d "$CACHE_DIR/build" ]] && [[ $FORCE -eq 0 ]] && [[ $CLEAN -eq 0 ]]; then
    # Quick check if libraries exist
    if [[ "$OSTYPE" == "darwin"* ]]; then
        LIB_CHECK="$CACHE_DIR/build/3rdparty/llama.cpp/src/libllama.dylib"
    else
        LIB_CHECK="$CACHE_DIR/build/3rdparty/llama.cpp/src/libllama.so"
    fi
    
    if [[ -f "$LIB_CHECK" ]]; then
        log_info "BitNet C++ already built at $CACHE_DIR"
        log_info "Use --force to rebuild or --clean for clean rebuild"
        # Still print setup instructions
        exec "$0" --print-setup
        exit 0
    fi
fi

# Clone or update repository
if [[ ! -d "$CACHE_DIR/.git" ]]; then
    log_info "Cloning BitNet repository to $CACHE_DIR..."
    git clone --recurse-submodules "$REPO_URL" "$CACHE_DIR"
    cd "$CACHE_DIR"
    git checkout "$REV"
    git submodule update --init --recursive
else
    log_info "Repository exists, checking revision..."
    cd "$CACHE_DIR"
    CURRENT_REV=$(git rev-parse HEAD)
    TARGET_REV=$(git rev-parse "$REV" 2>/dev/null || echo "$REV")
    
    if [[ "$CURRENT_REV" != "$TARGET_REV" ]] || [[ $FORCE -eq 1 ]]; then
        log_info "Updating to revision: $REV"
        git fetch --tags origin
        git checkout "$REV"
        git submodule update --init --recursive
    else
        log_info "Already at revision: $REV"
    fi
fi

log_info "Repository at commit: $(git rev-parse HEAD)"

# Handle Git LFS if available (safe to run even if not needed)
if command -v git-lfs >/dev/null 2>&1; then
    git -C "$CACHE_DIR" lfs install --local || true
    git -C "$CACHE_DIR" lfs pull || true
fi

# Sanity checks - fail fast if critical files are missing
log_info "Verifying critical files..."

# Check for the header that CMake complains about
# This may be a structure issue with the Microsoft BitNet repo
if [[ ! -f "$CACHE_DIR/include/bitnet-lut-kernels.h" ]] && [[ -d "$CACHE_DIR/include" ]]; then
    # Try to use a preset kernel as fallback (Microsoft's workaround)
    PRESET_KERNEL=""
    if [[ -d "$CACHE_DIR/preset_kernels" ]]; then
        for preset_dir in "$CACHE_DIR"/preset_kernels/*/; do
            if [[ -f "$preset_dir/bitnet-lut-kernels-tl2.h" ]]; then
                PRESET_KERNEL="$preset_dir/bitnet-lut-kernels-tl2.h"
                break
            elif [[ -f "$preset_dir/bitnet-lut-kernels.h" ]]; then
                PRESET_KERNEL="$preset_dir/bitnet-lut-kernels.h"
                break
            fi
        done
        
        if [[ -n "$PRESET_KERNEL" ]]; then
            log_warn "bitnet-lut-kernels.h missing, copying from preset: $PRESET_KERNEL"
            mkdir -p "$CACHE_DIR/include"
            cp "$PRESET_KERNEL" "$CACHE_DIR/include/bitnet-lut-kernels.h"
        fi
    fi
    
    # If still missing and needed, just warn but continue
    if [[ ! -f "$CACHE_DIR/include/bitnet-lut-kernels.h" ]]; then
        log_warn "bitnet-lut-kernels.h not found. Build may succeed without it."
        log_warn "If build fails, check the Microsoft BitNet repository structure."
    fi
fi

# Check for llama.cpp submodule
if [[ ! -f "$CACHE_DIR/3rdparty/llama.cpp/CMakeLists.txt" ]]; then
    log_error "FATAL: llama.cpp submodule not initialized properly!"
    log_error "Try: git -C '$CACHE_DIR' submodule update --init --recursive"
    exit 1
fi

# Check for critical headers we'll need for bindings
# Be more flexible since repository structure may vary
log_info "Checking for critical headers..."

# Check if we have basic CMakeLists.txt (minimal requirement)
if [[ ! -f "$CACHE_DIR/CMakeLists.txt" ]]; then
    log_error "FATAL: No CMakeLists.txt found - is this a valid BitNet repository?"
    exit 1
fi

# Look for essential headers, but don't fail if some are missing
CRITICAL_HEADERS=(
    "include/ggml-bitnet.h"
    "3rdparty/llama.cpp/include/llama.h"
    "3rdparty/llama.cpp/ggml/include/ggml.h"
    "include/llama.h"
    "src/llama.h"
)

FOUND_HEADERS=0
for header in "${CRITICAL_HEADERS[@]}"; do
    if [[ -f "$CACHE_DIR/$header" ]]; then
        log_info "  ✓ Found: $header"
        FOUND_HEADERS=$((FOUND_HEADERS + 1))
    fi
done

if [[ $FOUND_HEADERS -eq 0 ]]; then
    log_error "FATAL: No critical headers found. Repository may be incomplete."
    log_error "Expected at least one of: ${CRITICAL_HEADERS[*]}"
    exit 1
else
    log_info "  Found $FOUND_HEADERS critical headers - proceeding with build"
fi

# Create build directory
BUILD_DIR="$CACHE_DIR/build"
if [[ $CLEAN -eq 1 ]] && [[ -d "$BUILD_DIR" ]]; then
    log_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure and build
log_info "Configuring BitNet build (CPU-only, shared libs for FFI)..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBITNET_BUILD_TESTS=OFF \
    -DBITNET_BUILD_EXAMPLES=ON \
    -DBITNET_CUDA=OFF \
    -DBITNET_METAL=OFF \
    -DBITNET_BLAS=OFF \
    $CMAKE_FLAGS

log_info "Building BitNet (this may take a few minutes)..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Post-build verification
log_info "Verifying build artifacts..."

# Check for the main static libraries (since we build with LLAMA_STATIC=ON)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LLAMA_LIB="$BUILD_DIR/3rdparty/llama.cpp/src/libllama.a"
    GGML_LIB="$BUILD_DIR/3rdparty/llama.cpp/ggml/src/libggml.a"
    LIB_EXT="a"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    LLAMA_LIB="$BUILD_DIR/3rdparty/llama.cpp/src/libllama.a"
    GGML_LIB="$BUILD_DIR/3rdparty/llama.cpp/ggml/src/libggml.a"
    LIB_EXT="a"
else
    log_warn "Unknown OS type: $OSTYPE, assuming Linux"
    LLAMA_LIB="$BUILD_DIR/3rdparty/llama.cpp/src/libllama.a"
    GGML_LIB="$BUILD_DIR/3rdparty/llama.cpp/ggml/src/libggml.a"
    LIB_EXT="a"
fi

# Find actual library locations (build paths can vary)
FOUND_LIBS=()
for search_dir in "$BUILD_DIR" "$BUILD_DIR/3rdparty/llama.cpp" "$BUILD_DIR/3rdparty/llama.cpp/src"; do
    if [[ -f "$search_dir/libllama.$LIB_EXT" ]]; then
        LLAMA_LIB="$search_dir/libllama.$LIB_EXT"
        FOUND_LIBS+=("$LLAMA_LIB")
        break
    fi
done

for search_dir in "$BUILD_DIR" "$BUILD_DIR/3rdparty/llama.cpp/ggml" "$BUILD_DIR/3rdparty/llama.cpp/ggml/src"; do
    if [[ -f "$search_dir/libggml.$LIB_EXT" ]]; then
        GGML_LIB="$search_dir/libggml.$LIB_EXT"
        FOUND_LIBS+=("$GGML_LIB")
        break
    fi
done

if [[ ${#FOUND_LIBS[@]} -eq 0 ]]; then
    log_error "FATAL: No static libraries found after build!"
    log_error "Expected at least libllama.$LIB_EXT"
    log_error "Build may have failed. Check CMake output above."
    exit 1
fi

log_info "Found libraries:"
for lib in "${FOUND_LIBS[@]}"; do
    log_info "  - $lib"
done

# Check for CLI binary (optional, for manual testing)
POSSIBLE_BINARIES=(
    "$BUILD_DIR/bitnet-cli"
    "$BUILD_DIR/bitnet_cli"
    "$BUILD_DIR/bin/bitnet-cli"
    "$BUILD_DIR/bin/bitnet_cli"
    "$BUILD_DIR/main"
    "$BUILD_DIR/bin/main"
    "$BUILD_DIR/examples/main"
    "$BUILD_DIR/utils/main"
)

FOUND_BINARY=""
for binary in "${POSSIBLE_BINARIES[@]}"; do
    if [[ -f "$binary" ]]; then
        FOUND_BINARY="$binary"
        log_info "CLI binary found: $binary"
        break
    fi
done

if [[ -z "$FOUND_BINARY" ]]; then
    log_warn "No CLI binary found (this may be expected for library-only builds)"
    log_info "Built library files should be sufficient for FFI cross-validation"
fi

# Determine library paths for environment setup
if [[ "$OSTYPE" == "darwin"* ]]; then
    LD_VAR="DYLD_LIBRARY_PATH"
else
    LD_VAR="LD_LIBRARY_PATH"
fi

LIB_PATHS="$(dirname "$LLAMA_LIB")"
if [[ "$(dirname "$GGML_LIB")" != "$(dirname "$LLAMA_LIB")" ]]; then
    LIB_PATHS="$LIB_PATHS:$(dirname "$GGML_LIB")"
fi

log_info ""
log_info "================================================================"
log_info "BitNet C++ build complete!"
log_info "================================================================"
log_info "Repository:     $CACHE_DIR"
log_info "Build:          $BUILD_DIR"
log_info "Git revision:   $REV ($(git -C "$CACHE_DIR" rev-parse --short HEAD))"
log_info ""
log_info "To use for cross-validation:"
log_info ""
log_info "  export BITNET_CPP_DIR='$CACHE_DIR'"
log_info "  export ${LD_VAR}='${LIB_PATHS}:\$${LD_VAR}'"
log_info "  export OMP_NUM_THREADS=1    # For determinism"
log_info "  export GGML_NUM_THREADS=1"
log_info ""
log_info "Then run:"
log_info "  cargo test --features crossval -p crossval"
log_info ""
log_info "Or use convenience script:"
log_info "  ./scripts/crossval.sh /path/to/model.gguf"
log_info "================================================================"