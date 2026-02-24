# Preflight Verbose Output Implementation Summary

## Task: L4.1 - Enhance Preflight Verbose Output

**Goal**: Make `cargo run -p xtask -- preflight --backend bitnet --verbose` show detailed diagnostic information.

## Implementation

### Files Modified

**`xtask/src/crossval/preflight.rs`** - Enhanced with verbose diagnostics

### New Functions Added

#### 1. `print_verbose_success_diagnostics(backend: CppBackend)`
Called when libraries are successfully detected at build time.

**Output includes:**
- Environment variables checked (BITNET_CPP_DIR, LD_LIBRARY_PATH, etc.)
- Library search paths with existence status
- Libraries found in each path (with filenames)
- Required libraries status
- Build configuration flags
- Platform-specific linkage information

#### 2. `print_verbose_failure_diagnostics(backend: CppBackend)`
Called when libraries are NOT found at build time.

**Output includes:**
- Environment variable status (showing what's missing)
- Library search paths (what was searched)
- What libraries were searched for
- What was found (if any) in each path
- Actionable next steps:
  1. Setup command to run
  2. Manual environment variable instructions
  3. Rebuild instructions
  4. Re-run preflight command

#### 3. `print_env_var_status(var_name: &str)`
Helper function to display environment variable values.

**Features:**
- Shows variable name and value
- Indicates "(not set)" if not present
- Truncates values > 80 characters for readability

#### 4. `get_library_search_paths() -> Vec<PathBuf>`
Mirrors the search logic from `crossval/build.rs`.

**Search priority:**
1. `BITNET_CROSSVAL_LIBDIR` (explicit override)
2. `BITNET_CPP_DIR/build` and subdirectories
3. Default: `$HOME/.cache/bitnet_cpp/build` and subdirectories

**Paths checked:**
- `{root}/build`
- `{root}/build/lib`
- `{root}/build/3rdparty/llama.cpp/src`
- `{root}/build/3rdparty/llama.cpp/ggml/src`
- `{root}/lib`

#### 5. `find_libs_in_path(path: &Path, backend: CppBackend) -> Option<Vec<String>>`
Scans a directory for libraries matching the backend's requirements.

**Features:**
- Matches library prefixes (libbitnet, libllama, libggml)
- Returns library filenames (e.g., "libllama.so", "libggml.a")
- Returns None if no matching libraries found

## Modified Function

### `preflight_backend_libs(backend: CppBackend, verbose: bool) -> Result<()>`

**Changes:**
- Added call to `print_verbose_failure_diagnostics()` when libraries not found AND verbose=true
- Added call to `print_verbose_success_diagnostics()` when libraries found AND verbose=true
- Kept non-verbose output concise (single line status)

## Usage Examples

### Verbose Failure Case
```bash
$ cargo run -p xtask --features crossval-all -- preflight --backend bitnet --verbose

Checking bitnet.cpp backend...
❌ Backend 'bitnet.cpp' libraries: NOT AVAILABLE

Environment Variables:
  BITNET_CPP_DIR = (not set)
  ...

Library Search Paths:
  ✓ /home/user/.cache/bitnet_cpp/build (exists)
    Searched for: ["libbitnet"]
    No libraries found in this directory
  ...

Next Steps:
  1. Run setup command: eval "$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)"
  2. Or manually install...
  3. Rebuild xtask...
  4. Re-run preflight...
```

### Verbose Success Case
```bash
$ cargo run -p xtask --features crossval-all -- preflight --backend llama --verbose

Checking llama.cpp backend...
✓ Backend 'llama.cpp' libraries: AVAILABLE

Environment Variables:
  BITNET_CPP_DIR = /home/user/.cache/bitnet_cpp
  ...

Library Search Paths:
  ✓ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (exists)
    - libllama.so
    - libllama.a
  ...

Build Configuration:
  CROSSVAL_HAS_LLAMA=true
  Linked: libllama, libggml (dynamic)
```

### Non-Verbose Mode (unchanged)
```bash
$ cargo run -p xtask --features crossval-all -- preflight --backend llama

Checking llama.cpp backend...
✓ llama.cpp backend is available
```

## Key Features

### 1. Comprehensive Diagnostics
- Shows exactly what was checked
- Shows exactly what was found
- Platform-specific paths and env vars

### 2. Actionable Failure Messages
- Step-by-step recovery instructions
- Platform-specific commands
- Clear explanation of build-time vs runtime

### 3. Maintains Backward Compatibility
- Non-verbose mode unchanged
- Error messages preserved
- Exit codes unchanged

### 4. Platform Support
- Linux: Shows LD_LIBRARY_PATH, libstdc++ linkage
- macOS: Shows DYLD_LIBRARY_PATH, libc++ linkage
- Windows: Shows PATH (though WSL recommended)

## Testing

### Test Cases

1. **No libraries (bitnet backend)**
   ```bash
   cargo run -p xtask --features crossval -- preflight --backend bitnet --verbose
   ```
   - Expected: Detailed failure diagnostics
   - Shows: Environment vars not set, paths searched, next steps

2. **Libraries found (llama backend)**
   ```bash
   # After: export BITNET_CPP_DIR=/home/user/.cache/bitnet_cpp
   cargo clean -p bitnet-crossval && cargo build -p xtask --features crossval
   cargo run -p xtask --features crossval -- preflight --backend llama --verbose
   ```
   - Expected: Detailed success diagnostics
   - Shows: Libraries found, paths, build config

3. **Non-verbose mode**
   ```bash
   cargo run -p xtask --features crossval -- preflight --backend bitnet
   ```
   - Expected: Concise error message
   - No verbose output

4. **All backends**
   ```bash
   cargo run -p xtask --features crossval -- preflight
   ```
   - Expected: Summary of all backend status
   - Shows setup commands for missing backends

## Acceptance Criteria

✅ **Verbose mode shows comprehensive diagnostics**
- Environment variables
- Library search paths
- Libraries found
- Build configuration

✅ **Non-verbose mode stays concise**
- Single line success: "✓ backend is available"
- Brief error with setup instructions

✅ **Failure messages are actionable**
- Step-by-step next steps
- Exact commands to run
- Platform-specific instructions

✅ **Success shows what was found and where**
- Library paths and filenames
- Build-time configuration
- Environment variable values

## Future Enhancements (Optional)

1. **Symbol checking**: Verify specific symbols exist in libraries (e.g., using `nm`)
2. **Header file verification**: Check for required header files
3. **Version detection**: Extract and display library versions
4. **Dependency graph**: Show which libraries depend on which
5. **Runtime verification**: Test loading libraries at runtime (beyond build-time check)

## Notes

- The verbose diagnostics mirror the logic in `crossval/build.rs` to ensure consistency
- Library detection happens at BUILD time, not runtime - this is clearly documented
- The implementation handles missing directories gracefully
- Long environment variable values are truncated to 80 characters for readability
- Platform-specific code uses `#[cfg(target_os = "...")]` attributes
