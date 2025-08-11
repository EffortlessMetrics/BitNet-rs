#!/usr/bin/env bash
# Fetches and builds the Microsoft BitNet C++ implementation for cross-validation
# This provides the ground truth for our Rust implementation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
CACHE_DIR="${BITNET_CPP_DIR:-${BITNET_CPP_CACHE:-$HOME/.cache/bitnet_cpp}}"
REPO_URL="${BITNET_CPP_REPO:-https://github.com/microsoft/BitNet.git}"
# Pin to specific commit for reproducibility
# This is the latest stable release with working llama.cpp integration
DEFAULT_REV="b1-65-ggml"  # v1.0 release with BitNet b1.58 support
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
while [[ $# -gt 0 ]]; do
    case $1 in
        --rev|--tag)
            REV="$2"
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
        --help)
            echo "Usage: $0 [--tag TAG] [--clean] [--force]"
            echo "  --tag TAG    Git revision/tag to checkout (default: $DEFAULT_REV)"
            echo "  --clean      Clean build before compiling"
            echo "  --force      Force rebuild even if already built"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
# This is a known issue with the Microsoft BitNet repo structure
if [[ ! -f "$CACHE_DIR/include/bitnet-lut-kernels.h" ]]; then
    # Try to use a preset kernel as fallback (Microsoft's workaround)
    PRESET_KERNEL=""
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
        cp "$PRESET_KERNEL" "$CACHE_DIR/include/bitnet-lut-kernels.h"
    else
        log_error "FATAL: bitnet-lut-kernels.h not found and no preset available!"
        log_error "This is a known issue with the Microsoft BitNet repo."
        log_error "See: https://github.com/microsoft/BitNet/issues"
        exit 1
    fi
fi

# Check for llama.cpp submodule
if [[ ! -f "$CACHE_DIR/3rdparty/llama.cpp/CMakeLists.txt" ]]; then
    log_error "FATAL: llama.cpp submodule not initialized properly!"
    log_error "Try: git -C '$CACHE_DIR' submodule update --init --recursive"
    exit 1
fi

# Check for critical headers we'll need for bindings
CRITICAL_HEADERS=(
    "include/ggml-bitnet.h"
    "3rdparty/llama.cpp/include/llama.h"
    "3rdparty/llama.cpp/ggml/include/ggml.h"
)

for header in "${CRITICAL_HEADERS[@]}"; do
    if [[ ! -f "$CACHE_DIR/$header" ]]; then
        log_error "FATAL: Required header not found: $header"
        exit 1
    fi
done

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
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_CUDA=OFF \
    -DLLAMA_METAL=OFF \
    -DLLAMA_BLAS=OFF \
    -DLLAMA_ALL_WARNINGS=OFF

log_info "Building BitNet (this may take a few minutes)..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Post-build verification
log_info "Verifying build artifacts..."

# Check for the main shared libraries
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LLAMA_LIB="$BUILD_DIR/3rdparty/llama.cpp/src/libllama.so"
    GGML_LIB="$BUILD_DIR/3rdparty/llama.cpp/ggml/src/libggml.so"
    LIB_EXT="so"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    LLAMA_LIB="$BUILD_DIR/3rdparty/llama.cpp/src/libllama.dylib"
    GGML_LIB="$BUILD_DIR/3rdparty/llama.cpp/ggml/src/libggml.dylib"
    LIB_EXT="dylib"
else
    log_warn "Unknown OS type: $OSTYPE, assuming Linux"
    LLAMA_LIB="$BUILD_DIR/3rdparty/llama.cpp/src/libllama.so"
    GGML_LIB="$BUILD_DIR/3rdparty/llama.cpp/ggml/src/libggml.so"
    LIB_EXT="so"
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
    log_error "FATAL: No shared libraries found after build!"
    log_error "Expected at least libllama.$LIB_EXT"
    log_error "Build may have failed or produced static libraries only."
    exit 1
fi

log_info "Found libraries:"
for lib in "${FOUND_LIBS[@]}"; do
    log_info "  - $lib"
done

# Check for CLI binary (optional, for manual testing)
if [[ -f "$BUILD_DIR/bin/llama-cli" ]]; then
    log_info "CLI binary found: $BUILD_DIR/bin/llama-cli"
elif [[ -f "$BUILD_DIR/3rdparty/llama.cpp/bin/llama-cli" ]]; then
    log_info "CLI binary found: $BUILD_DIR/3rdparty/llama.cpp/bin/llama-cli"
else
    log_warn "No llama-cli binary found (OK if not needed for FFI)"
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