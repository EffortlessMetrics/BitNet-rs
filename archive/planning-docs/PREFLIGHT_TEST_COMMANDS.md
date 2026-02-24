# Preflight Test Commands

## Quick Test Guide

### 1. Test Failure Case (No Libraries)

```bash
# Clear environment to ensure clean test
unset BITNET_CPP_DIR
unset BITNET_CPP_PATH
unset BITNET_CROSSVAL_LIBDIR

# Rebuild to reset detection
cargo clean -p bitnet-crossval
cargo build -p xtask --no-default-features --features crossval

# Test verbose output (shows detailed diagnostics)
cargo run -p xtask --no-default-features --features crossval -- \
  preflight --backend bitnet --verbose

# Test non-verbose output (concise)
cargo run -p xtask --no-default-features --features crossval -- \
  preflight --backend bitnet
```

**Expected Output (Verbose):**
- ❌ Backend libraries: NOT AVAILABLE
- Environment Variables section showing "(not set)"
- Library Search Paths with detailed search results
- Actionable "Next Steps" with 4 steps

**Expected Output (Non-Verbose):**
- Error message with setup instructions
- No detailed diagnostics

### 2. Test Success Case (With Libraries)

```bash
# Set environment to point to C++ libraries
export BITNET_CPP_DIR="/home/steven/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src:$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src:$LD_LIBRARY_PATH"

# Rebuild with environment set
cargo clean -p bitnet-crossval
cargo clean -p xtask
cargo build -p xtask --no-default-features --features crossval

# Test verbose output (shows libraries found)
cargo run -p xtask --no-default-features --features crossval -- \
  preflight --backend llama --verbose

# Test non-verbose output
cargo run -p xtask --no-default-features --features crossval -- \
  preflight --backend llama
```

**Expected Output (Verbose):**
- ✓ Backend libraries: AVAILABLE
- Environment Variables showing BITNET_CPP_DIR value
- Library Search Paths with libraries listed
- Build Configuration section
- Status: "All required libraries detected at build time"

**Expected Output (Non-Verbose):**
- ✓ llama.cpp backend is available

### 3. Test All Backends Status

```bash
# Check all backends at once
cargo run -p xtask --no-default-features --features crossval -- preflight

# With verbose flag (shows detailed status)
cargo run -p xtask --no-default-features --features crossval -- preflight --verbose
```

**Expected Output (No Libraries):**
```
Backend Library Status:

  ✗ bitnet.cpp: NOT AVAILABLE
    Setup: eval "$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)"

  ✗ llama.cpp: NOT AVAILABLE
    Setup: eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

No C++ backends available. Cross-validation will not work.
Run setup commands above to install backends.
```

**Expected Output (LLaMA Available):**
```
Backend Library Status:

  ✗ bitnet.cpp: NOT AVAILABLE
    Setup: eval "$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)"

  ✓ llama.cpp: AVAILABLE
    Libraries: libllama*, libggml*
```

## Detailed Diagnostics Examples

### Failure Case - Full Verbose Output

```
❌ Backend 'bitnet.cpp' libraries: NOT AVAILABLE

Environment Variables:
  BITNET_CPP_DIR = (not set)
  BITNET_CPP_PATH = (not set)
  BITNET_CROSSVAL_LIBDIR = (not set)
  LD_LIBRARY_PATH = /usr/lib:/usr/local/lib

Library Search Paths:
  ✓ /home/user/.cache/bitnet_cpp/build (exists)
    Searched for: ["libbitnet"]
    No libraries found in this directory
  ✗ /home/user/.cache/bitnet_cpp/build/lib (not found)
  ✓ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (exists)
    Searched for: ["libbitnet"]
    Found (other libraries):
      - libllama.so
      - libllama.a
  ✓ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src (exists)
    Searched for: ["libbitnet"]
    Found (other libraries):
      - libggml.so
      - libggml.a
  ✗ /home/user/.cache/bitnet_cpp/lib (not found)

Searched for:
  - libbitnet.so / libbitnet.dylib / libbitnet.a

Next Steps:
  1. Run setup command:
     eval "$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)"

  2. Or manually install and set environment:
     export BITNET_CPP_DIR=/path/to/bitnet.cpp
     export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build:$LD_LIBRARY_PATH

  3. Rebuild xtask to detect libraries:
     cargo clean -p xtask && cargo build -p xtask --features crossval-all

  4. Re-run preflight:
     cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
```

### Success Case - Full Verbose Output

```
✓ Backend 'llama.cpp' libraries: AVAILABLE

Environment Variables:
  BITNET_CPP_DIR = /home/user/.cache/bitnet_cpp
  BITNET_CPP_PATH = (not set)
  BITNET_CROSSVAL_LIBDIR = (not set)
  LD_LIBRARY_PATH = /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src:/home/user...

Library Search Paths:
  ✓ /home/user/.cache/bitnet_cpp/build (exists)
  ✗ /home/user/.cache/bitnet_cpp/build/lib (not found)
  ✓ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (exists)
    - libllama.so
    - libllama.a
  ✓ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src (exists)
    - libggml.so
    - libggml.a
  ✗ /home/user/.cache/bitnet_cpp/lib (not found)

Required Libraries: ["libllama", "libggml"]
Status: All required libraries detected at build time

Build Configuration:
  CROSSVAL_HAS_LLAMA=true
  Linked: libllama, libggml (dynamic)
  Standard library: libstdc++ (dynamic)
```

## Platform-Specific Tests

### Linux
```bash
# Check LD_LIBRARY_PATH is shown
cargo run -p xtask --features crossval -- preflight --backend llama --verbose | grep LD_LIBRARY_PATH
```

### macOS
```bash
# Check DYLD_LIBRARY_PATH is shown
cargo run -p xtask --features crossval -- preflight --backend llama --verbose | grep DYLD_LIBRARY_PATH
```

### Windows
```bash
# Check PATH is shown
cargo run -p xtask --features crossval -- preflight --backend llama --verbose | grep "PATH ="
```

## Validation Checklist

- [x] Verbose mode shows environment variables
- [x] Verbose mode shows library search paths
- [x] Verbose mode shows libraries found (with filenames)
- [x] Verbose mode shows build configuration
- [x] Non-verbose mode stays concise
- [x] Failure messages are actionable (4-step recovery)
- [x] Success messages show what was found
- [x] Platform-specific env vars shown correctly
- [x] Long env values truncated to 80 chars
- [x] Missing paths marked with ✗
- [x] Existing paths marked with ✓
- [x] Other libraries distinguished from required libraries
