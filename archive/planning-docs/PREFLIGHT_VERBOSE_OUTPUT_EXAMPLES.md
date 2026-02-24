# Preflight Verbose Output Examples

This document demonstrates the enhanced verbose diagnostics for the `xtask preflight` command.

## Command Usage

```bash
# Check specific backend with verbose diagnostics
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
cargo run -p xtask --features crossval-all -- preflight --backend llama --verbose

# Check all backends (non-verbose)
cargo run -p xtask --features crossval-all -- preflight
```

## Example: Failure Case (Libraries Not Found)

When libraries are NOT detected at build time, verbose mode shows comprehensive diagnostics:

```bash
$ cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose
```

**Output:**

```
Checking bitnet.cpp backend...
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

Error: Backend 'bitnet.cpp' selected but required libraries not found.
```

## Example: Success Case (Libraries Found)

When libraries ARE detected at build time, verbose mode shows success diagnostics:

```bash
$ cargo run -p xtask --features crossval-all -- preflight --backend llama --verbose
```

**Output:**

```
Checking llama.cpp backend...
✓ Backend 'llama.cpp' libraries: AVAILABLE

Environment Variables:
  BITNET_CPP_DIR = /home/user/.cache/bitnet_cpp
  BITNET_CPP_PATH = (not set)
  BITNET_CROSSVAL_LIBDIR = (not set)
  LD_LIBRARY_PATH = /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src:/home/use...

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

## Non-Verbose Mode

When run without `--verbose`, output is concise:

```bash
$ cargo run -p xtask --features crossval-all -- preflight --backend llama
```

**Output:**

```
Checking llama.cpp backend...
✓ llama.cpp backend is available
```

## Check All Backends

When run without specifying a backend:

```bash
$ cargo run -p xtask --features crossval-all -- preflight
```

**Output (when both backends available):**

```
Backend Library Status:

  ✓ bitnet.cpp: AVAILABLE
    Libraries: libbitnet*

  ✓ llama.cpp: AVAILABLE
    Libraries: libllama*, libggml*

Both backends available. Dual-backend cross-validation supported.
```

**Output (when no backends available):**

```
Backend Library Status:

  ✗ bitnet.cpp: NOT AVAILABLE
    Setup: eval "$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)"

  ✗ llama.cpp: NOT AVAILABLE
    Setup: eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

No C++ backends available. Cross-validation will not work.
Run setup commands above to install backends.
```

## Features of Verbose Mode

### 1. Environment Variable Status
- Shows all relevant env vars (BITNET_CPP_DIR, LD_LIBRARY_PATH, etc.)
- Truncates long values to 80 characters for readability
- Clearly indicates when variables are not set

### 2. Library Search Paths
- Lists all directories searched (mirrors crossval/build.rs logic)
- Shows existence status for each path
- Lists libraries found in each path
- Distinguishes between required libraries and other libraries

### 3. Actionable Next Steps
- Platform-specific instructions (Linux: LD_LIBRARY_PATH, macOS: DYLD_LIBRARY_PATH)
- Exact commands to run for setup
- Clear explanation that detection happens at BUILD time
- Step-by-step recovery instructions

### 4. Build Configuration
- Shows compile-time flags set
- Lists linked libraries and linkage type (static/dynamic)
- Platform-specific standard library linkage

## Implementation Details

The verbose diagnostics are implemented in `xtask/src/crossval/preflight.rs`:

- `print_verbose_success_diagnostics()`: Called when libraries found
- `print_verbose_failure_diagnostics()`: Called when libraries NOT found
- `print_env_var_status()`: Displays environment variable values
- `get_library_search_paths()`: Mirrors crossval/build.rs search logic
- `find_libs_in_path()`: Scans directories for required libraries

These functions provide comprehensive diagnostics while keeping non-verbose mode concise.
