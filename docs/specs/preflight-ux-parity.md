# Preflight UX Parity Enhancement Specification

**Status**: Draft
**Created**: 2025-10-25
**Feature**: Unified backend diagnostics for preflight command
**Priority**: Medium (Quality of Life)
**Scope**: `xtask/src/crossval/preflight.rs`, `xtask/src/crossval/backend.rs`, `crossval/build.rs`

---

## 1. Problem Statement

The `xtask preflight` command displays C++ backend availability diagnostics with minor UX inconsistencies between BitNet.cpp and llama.cpp implementations. While both backends provide comprehensive diagnostic output, four specific gaps reduce clarity and coverage:

### 1.1 Issue 1: Setup Command Flag Inconsistency

**Severity**: Low-Medium (breaks UX parity)
**Impact**: User confusion about command syntax

**Current state:**
```bash
# BitNet setup command
eval "$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)"

# LLaMA setup command
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

**Problem**: BitNet includes `--bitnet` flag while LLaMA does not, creating inconsistency in recovery instructions. The `setup-cpp-auto` command auto-detects backends, making the explicit flag redundant and confusing.

**Location**: `xtask/src/crossval/backend.rs` line 102

### 1.2 Issue 2: Missing Standalone LLaMA.cpp Search Path

**Severity**: Medium (coverage gap)
**Impact**: Standalone llama.cpp builds not detected

**Current search paths** (6 total):
1. `BITNET_CROSSVAL_LIBDIR` override
2. `$BITNET_CPP_DIR/build`
3. `$BITNET_CPP_DIR/build/lib`
4. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src` (embedded llama only)
5. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src` (embedded ggml only)
6. `$BITNET_CPP_DIR/lib`

**Gap**: Standalone llama.cpp installations place libraries in `build/bin/` (CMake standard layout), which is not searched.

**Locations**:
- `crossval/build.rs` lines 70-76 (build-time detection)
- `xtask/src/crossval/preflight.rs` lines 572-576 (runtime display)

### 1.3 Issue 3: Generic Path Labels Lacking Context

**Severity**: Low (clarity gap)
**Impact**: Users don't understand why certain paths are searched

**Current output:**
```
4. BITNET_CPP_DIR/build/3rdparty/llama.cpp/src
   ✗ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (not found)
```

**Gap**: No indication that this path is for "embedded llama.cpp dependency" vs "main bitnet libraries". Users can't distinguish between critical paths and optional dependency paths.

**Location**: `xtask/src/crossval/preflight.rs` lines 245-272 (search path display logic)

### 1.4 Issue 4: Build Metadata Missing Library Information

**Severity**: Low (information gap)
**Impact**: Diagnostics don't show what was searched for

**Current output:**
```
Build-Time Detection Metadata
─────────────────────────────────────────────────────────
CROSSVAL_HAS_BITNET = false
Last xtask build: 2025-10-25 14:32:18 UTC
Build feature flags: crossval-all
```

**Gap**: Doesn't indicate which specific libraries were searched (e.g., `libbitnet` for BitNet, `libllama` + `libggml` for LLaMA). Users can't tell if detection failed due to wrong library names or missing paths.

**Location**: `xtask/src/crossval/preflight.rs` lines 653-674 (metadata formatting)

---

## 2. Architecture

### 2.1 Enhanced Output Format

The preflight command uses a 7-section structure for verbose success output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Backend 'bitnet.cpp': AVAILABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Environment Configuration
   - BITNET_CPP_DIR, BITNET_CROSSVAL_LIBDIR, LD_LIBRARY_PATH
   - Platform-specific loader variables

2. Library Search Paths (Priority Order)
   - Numbered list with existence status
   - Found libraries enumerated per path
   - [ENHANCEMENT] Context labels for nested paths

3. Required Libraries for Backend
   - Checkmarks for found libraries

4. Build-Time Detection Metadata
   - CROSSVAL_HAS_* constant value
   - [ENHANCEMENT] Searched library names
   - ISO 8601 timestamp
   - Build feature flags

5. Platform-Specific Configuration
   - Platform type (Linux/macOS/Windows)
   - Standard library type
   - RPATH status
   - Loader search order

6. Summary
   - Confirmation + example command

7. [FAILURE PATH] Recovery Steps
   - [ENHANCEMENT] Unified setup command (no --bitnet flag)
   - Two options: auto-setup vs manual
```

### 2.2 Search Path Display Enhancement

**Before (generic):**
```
4. BITNET_CPP_DIR/build/3rdparty/llama.cpp/src
   ✗ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (not found)
```

**After (contextualized):**
```
4. BITNET_CPP_DIR/build/bin (standalone llama.cpp)
   ✓ /home/user/.cache/bitnet_cpp/build/bin (exists)
   Found libraries:
     - libllama.so.3
     - libggml.so

5. BITNET_CPP_DIR/build/3rdparty/llama.cpp/src (embedded llama.cpp)
   ✗ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (not found)
```

**Path context labels** help users understand:
- Which paths are for embedded dependencies vs standalone installations
- Why certain paths might not exist (e.g., no embedded llama.cpp in standalone builds)
- What each library component represents (ggml library, llama.cpp inference, etc.)

### 2.3 Build Metadata Enhancement

**Before:**
```
Build-Time Detection Metadata
─────────────────────────────────────────────────────────
CROSSVAL_HAS_BITNET = false
Last xtask build: 2025-10-25 14:32:18 UTC
Build feature flags: crossval-all
```

**After:**
```
Build-Time Detection Metadata
─────────────────────────────────────────────────────────
CROSSVAL_HAS_BITNET = false
Required libraries: libbitnet
Last xtask build: 2025-10-25 14:32:18 UTC
Build feature flags: crossval-all
```

This clearly shows **what was searched for** during build-time detection, helping diagnose:
- Library name mismatches (e.g., `libbitnet.so` vs `libbitnet-core.so`)
- Version suffix issues (e.g., `libllama.so.3` vs `libllama.so`)
- Multiple library requirements (LLaMA needs both `libllama` + `libggml`)

### 2.4 Search Path Priority with Coverage

**New search order** (7 paths instead of 6):

| Priority | Path | Label | Used By | Notes |
|----------|------|-------|---------|-------|
| 0 | BITNET_CROSSVAL_LIBDIR | Override | Both | Explicit user override |
| 1 | build | Main output | Both | CMake default |
| 2 | build/bin | Standalone layout | LLaMA | **NEW**: Standalone llama.cpp |
| 3 | build/lib | Library subdir | Both | Common layout |
| 4 | build/3rdparty/llama.cpp/src | Embedded llama | BitNet | BitNet.cpp dependency |
| 5 | build/3rdparty/llama.cpp/ggml/src | Embedded ggml | BitNet | BitNet.cpp ggml |
| 6 | lib | Top-level | Both | Alternative layout |

**Rationale for build/bin priority:**
- Standalone llama.cpp commonly uses `build/bin/` for binaries + shared libraries
- Inserting after `build/` but before `build/lib/` maintains priority of main output directory
- Does not affect BitNet.cpp (which uses embedded llama in `build/3rdparty/`)

---

## 3. API Contracts

### 3.1 Backend Enum Methods

**File**: `xtask/src/crossval/backend.rs`

#### 3.1.1 `setup_command()` Unified Return

**Before:**
```rust
pub fn setup_command(&self) -> &'static str {
    match self {
        Self::BitNet => "eval \"$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)\"",
        Self::Llama => "eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"",
    }
}
```

**After:**
```rust
pub fn setup_command(&self) -> &'static str {
    // Unified command - auto-detection handles both backends
    "eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\""
}
```

**Contract**: Returns identical setup command for all backends, relying on auto-detection logic in `setup-cpp-auto` to determine which backend(s) to install.

**Backward compatibility**: Existing users with `--bitnet` flag in scripts will continue to work (flag is optional).

#### 3.1.2 `required_libs()` Unchanged

```rust
pub fn required_libs(&self) -> &[&'static str] {
    match self {
        Self::BitNet => &["libbitnet"],
        Self::Llama => &["libllama", "libggml"],
    }
}
```

**Contract**: Returns library stems (without `.so`/`.dylib` extension) for build-time and runtime detection.

### 3.2 Search Path Generation

**File**: `xtask/src/crossval/preflight.rs`

#### 3.2.1 `get_library_search_paths()` Enhancement

**Function signature** (unchanged):
```rust
fn get_library_search_paths() -> Vec<std::path::PathBuf>
```

**Implementation change** (insert new path):

**Before** (lines 572-576):
```rust
let root = Path::new(&bitnet_root);
paths.push(root.join("build"));
paths.push(root.join("build/lib"));
paths.push(root.join("build/3rdparty/llama.cpp/src"));
// ... continue
```

**After**:
```rust
let root = Path::new(&bitnet_root);
paths.push(root.join("build"));
paths.push(root.join("build/bin"));  // NEW: Standalone llama.cpp layout
paths.push(root.join("build/lib"));
paths.push(root.join("build/3rdparty/llama.cpp/src"));
// ... continue
```

**Contract**: Returns 7 paths in priority order. New path inserted at index 2 (after `build/`, before `build/lib/`).

**Backward compatibility**: Existing paths unchanged; only adds new search location.

#### 3.2.2 `get_path_context_label()` New Helper

**New function**:
```rust
/// Returns context label for special search paths
fn get_path_context_label(path: &Path) -> &'static str {
    let path_str = path.to_string_lossy();

    if path_str.contains("3rdparty/llama.cpp/src") {
        " (embedded llama.cpp)"
    } else if path_str.contains("3rdparty/llama.cpp/ggml") {
        " (embedded ggml)"
    } else if path_str.ends_with("build/bin") {
        " (standalone llama.cpp)"
    } else if path_str.contains("CROSSVAL_LIBDIR") {
        " (explicit override)"
    } else {
        ""  // No label for standard paths
    }
}
```

**Contract**: Returns static string labels for path clarification. Empty string for paths that don't need context.

**Usage**: Appended to path display in verbose output.

### 3.3 Output Formatting Functions

**File**: `xtask/src/crossval/preflight.rs`

#### 3.3.1 `format_build_metadata()` Enhancement

**Before** (lines 666-671):
```rust
format!(
    "Build-Time Detection Metadata\n\
     {}\n\
     CROSSVAL_HAS_{} = {}\n\
     Last xtask build: {}\n\
     Build feature flags: crossval-all",
    SEPARATOR_LIGHT, backend_name, has_backend, timestamp
)
```

**After**:
```rust
let required_libs = backend.required_libs().join(", ");
format!(
    "Build-Time Detection Metadata\n\
     {}\n\
     CROSSVAL_HAS_{} = {}\n\
     Required libraries: {}\n\
     Last xtask build: {}\n\
     Build feature flags: crossval-all",
    SEPARATOR_LIGHT, backend_name, has_backend, required_libs, timestamp
)
```

**Contract**: Adds "Required libraries" line showing what build system searched for. Uses `join(", ")` for multi-lib backends (LLaMA).

**Backward compatibility**: Appends new line; existing output structure preserved.

#### 3.3.2 Search Path Display Enhancement (lines 245-272)

**Current logic**:
```rust
let path_desc = if let Some(parent) = path.parent() {
    if parent.ends_with("bitnet_cpp") {
        format!("BITNET_CPP_DIR/{}",
                path.file_name().and_then(|s| s.to_str()).unwrap_or(""))
    } else {
        path.display().to_string()
    }
} else {
    path.display().to_string()
};

println!("  {}. {}", idx + 1, path_desc);
```

**Enhanced logic**:
```rust
let path_desc = if let Some(parent) = path.parent() {
    if parent.ends_with("bitnet_cpp") {
        format!("BITNET_CPP_DIR/{}",
                path.file_name().and_then(|s| s.to_str()).unwrap_or(""))
    } else {
        path.display().to_string()
    }
} else {
    path.display().to_string()
};

let context_label = get_path_context_label(path);  // NEW

println!("  {}. {}{}", idx + 1, path_desc, context_label);  // MODIFIED
```

**Contract**: Appends context label to path description. Empty labels have no visual impact.

---

## 4. Implementation Plan

### 4.1 Phase 1: Setup Command Unification (Priority 1)

**Estimated effort**: 5 minutes
**Risk**: Very Low (backward compatible)

#### Step 1.1: Update Backend Enum

**File**: `xtask/src/crossval/backend.rs`
**Line**: 100-105

**Change**:
```diff
 pub fn setup_command(&self) -> &'static str {
-    match self {
-        Self::BitNet => "eval \"$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)\"",
-        Self::Llama => "eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"",
-    }
+    // Unified command - auto-detection handles both backends
+    "eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\""
 }
```

**Verification**:
```bash
cargo test -p xtask --lib crossval::backend
cargo run -p xtask -- preflight --backend bitnet  # Check error message
```

### 4.2 Phase 2: Add build/bin Search Path (Priority 2)

**Estimated effort**: 15 minutes
**Risk**: Low (adds new search location without removing existing)

#### Step 2.1: Update Build-Time Detection

**File**: `crossval/build.rs`
**Lines**: 70-76

**Change**:
```diff
 let mut possible_lib_dirs = vec![
     Path::new(&bitnet_root).join("build"),
+    Path::new(&bitnet_root).join("build/bin"),  // Standalone llama.cpp
     Path::new(&bitnet_root).join("build/lib"),
     Path::new(&bitnet_root).join("build/3rdparty/llama.cpp/src"),
     Path::new(&bitnet_root).join("build/3rdparty/llama.cpp/ggml/src"),
     Path::new(&bitnet_root).join("lib"),
 ];
```

**Impact**: Build script now searches 7 paths instead of 6.

#### Step 2.2: Update Runtime Display

**File**: `xtask/src/crossval/preflight.rs`
**Lines**: 572-576

**Change**:
```diff
 let root = Path::new(&bitnet_root);
 paths.push(root.join("build"));
+paths.push(root.join("build/bin"));  // Standalone llama.cpp layout
 paths.push(root.join("build/lib"));
 paths.push(root.join("build/3rdparty/llama.cpp/src"));
 paths.push(root.join("build/3rdparty/llama.cpp/ggml/src"));
 paths.push(root.join("lib"));
```

**Verification**:
```bash
# Rebuild crossval to pick up new search paths
cargo clean -p crossval && cargo build -p crossval --features crossval-all

# Rebuild xtask to embed new paths in diagnostics
cargo clean -p xtask && cargo build -p xtask --features crossval-all

# Test with verbose output
cargo run -p xtask -- preflight --backend llama --verbose

# Should show:
#   2. BITNET_CPP_DIR/build (exists)
#   3. BITNET_CPP_DIR/build/bin (standalone llama.cpp)
```

### 4.3 Phase 3: Add Path Context Labels (Priority 3)

**Estimated effort**: 20 minutes
**Risk**: Very Low (append-only change)

#### Step 3.1: Add Helper Function

**File**: `xtask/src/crossval/preflight.rs`
**Location**: After `find_libs_in_path()` function (around line 603)

**Add**:
```rust
/// Returns context label for special search paths
fn get_path_context_label(path: &Path) -> &'static str {
    let path_str = path.to_string_lossy();

    if path_str.contains("3rdparty/llama.cpp/src") {
        " (embedded llama.cpp)"
    } else if path_str.contains("3rdparty/llama.cpp/ggml") {
        " (embedded ggml)"
    } else if path_str.ends_with("build/bin") {
        " (standalone llama.cpp)"
    } else if path_str.contains("CROSSVAL_LIBDIR") {
        " (explicit override)"
    } else {
        ""  // No label for standard paths
    }
}
```

#### Step 3.2: Integrate into Display Logic

**File**: `xtask/src/crossval/preflight.rs`
**Lines**: 245-272 (search path display in `print_verbose_success_diagnostics`)

**Change**:
```diff
 let path_desc = if let Some(parent) = path.parent() {
     if parent.ends_with("bitnet_cpp") {
         format!("BITNET_CPP_DIR/{}",
                 path.file_name().and_then(|s| s.to_str()).unwrap_or(""))
     } else {
         path.display().to_string()
     }
 } else {
     path.display().to_string()
 };

+let context_label = get_path_context_label(path);
-println!("  {}. {}", idx + 1, path_desc);
+println!("  {}. {}{}", idx + 1, path_desc, context_label);
```

**Apply same pattern** in `print_verbose_failure_diagnostics()` (lines 371-423).

**Verification**:
```bash
cargo run -p xtask -- preflight --backend bitnet --verbose

# Should show context labels:
#   3. BITNET_CPP_DIR/build/bin (standalone llama.cpp)
#   5. BITNET_CPP_DIR/build/3rdparty/llama.cpp/src (embedded llama.cpp)
#   6. BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src (embedded ggml)
```

### 4.4 Phase 4: Enhance Build Metadata (Priority 4)

**Estimated effort**: 10 minutes
**Risk**: Very Low (appends new line)

#### Step 4.1: Update Metadata Function

**File**: `xtask/src/crossval/preflight.rs`
**Lines**: 653-674

**Change**:
```diff
 fn format_build_metadata(backend: CppBackend) -> String {
     let has_backend = match backend {
         CppBackend::BitNet => HAS_BITNET,
         CppBackend::Llama => HAS_LLAMA,
     };

     let backend_name = match backend {
         CppBackend::BitNet => "BITNET",
         CppBackend::Llama => "LLAMA",
     };

     let timestamp = get_xtask_build_timestamp().unwrap_or_else(|| "unknown".to_string());
+    let required_libs = backend.required_libs().join(", ");

     format!(
         "Build-Time Detection Metadata\n\
          {}\n\
          CROSSVAL_HAS_{} = {}\n\
+         Required libraries: {}\n\
          Last xtask build: {}\n\
          Build feature flags: crossval-all",
-        SEPARATOR_LIGHT, backend_name, has_backend, timestamp
+        SEPARATOR_LIGHT, backend_name, has_backend, required_libs, timestamp
     )
 }
```

**Verification**:
```bash
cargo run -p xtask -- preflight --backend bitnet --verbose

# Should show:
#   Build-Time Detection Metadata
#   ─────────────────────────────────────────────────────────
#   CROSSVAL_HAS_BITNET = true
#   Required libraries: libbitnet
#   Last xtask build: 2025-10-25 14:32:18 UTC
#   Build feature flags: crossval-all

cargo run -p xtask -- preflight --backend llama --verbose

# Should show:
#   Required libraries: libllama, libggml
```

---

## 5. Test Requirements

### 5.1 Unit Tests

#### 5.1.1 Backend Setup Command Unification

**File**: `xtask/src/crossval/backend.rs` (add test module if missing)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_command_unified() {
        // Both backends should return identical setup command
        let bitnet_cmd = CppBackend::BitNet.setup_command();
        let llama_cmd = CppBackend::Llama.setup_command();

        assert_eq!(bitnet_cmd, llama_cmd, "Setup commands should be identical");
        assert!(!bitnet_cmd.contains("--bitnet"), "Setup command should not have --bitnet flag");
        assert!(bitnet_cmd.contains("setup-cpp-auto"), "Setup command should use setup-cpp-auto");
    }
}
```

#### 5.1.2 Search Path Coverage

**File**: `xtask/src/crossval/preflight.rs` (add to test module)

```rust
#[test]
fn test_search_paths_include_build_bin() {
    let paths = get_library_search_paths();

    // Should have 7 paths total (6 original + 1 new)
    assert_eq!(paths.len(), 7, "Should search 7 paths");

    // build/bin should be present
    let has_build_bin = paths.iter().any(|p| {
        p.to_string_lossy().ends_with("build/bin")
    });
    assert!(has_build_bin, "Should include build/bin path for standalone llama.cpp");

    // Verify priority order: build comes before build/bin
    let build_idx = paths.iter().position(|p| p.ends_with("build")).unwrap();
    let build_bin_idx = paths.iter().position(|p| p.ends_with("build/bin")).unwrap();
    assert!(build_idx < build_bin_idx, "build/ should come before build/bin");
}
```

#### 5.1.3 Context Label Generation

```rust
#[test]
fn test_path_context_labels() {
    use std::path::PathBuf;

    // Embedded llama.cpp paths get labels
    let embedded_llama = PathBuf::from("/opt/bitnet/build/3rdparty/llama.cpp/src");
    assert_eq!(get_path_context_label(&embedded_llama), " (embedded llama.cpp)");

    let embedded_ggml = PathBuf::from("/opt/bitnet/build/3rdparty/llama.cpp/ggml/src");
    assert_eq!(get_path_context_label(&embedded_ggml), " (embedded ggml)");

    // Standalone llama.cpp path gets label
    let standalone = PathBuf::from("/opt/bitnet/build/bin");
    assert_eq!(get_path_context_label(&standalone), " (standalone llama.cpp)");

    // Standard paths get no label
    let standard = PathBuf::from("/opt/bitnet/build");
    assert_eq!(get_path_context_label(&standard), "");
}
```

#### 5.1.4 Build Metadata Format

```rust
#[test]
fn test_build_metadata_includes_libraries() {
    let bitnet_meta = format_build_metadata(CppBackend::BitNet);
    assert!(bitnet_meta.contains("Required libraries: libbitnet"),
            "BitNet metadata should show required library");

    let llama_meta = format_build_metadata(CppBackend::Llama);
    assert!(llama_meta.contains("Required libraries: libllama, libggml"),
            "LLaMA metadata should show both required libraries");
}
```

### 5.2 Integration Tests

#### 5.2.1 Manual Verification Checklist

**Test Scenario 1: Success Path with Verbose Output**

```bash
# Prerequisite: C++ libraries installed
cargo clean -p crossval && cargo clean -p xtask
cargo build -p xtask --features crossval-all

cargo run -p xtask -- preflight --backend bitnet --verbose
```

**Expected output sections:**
1. ✓ Header: "Backend 'bitnet.cpp': AVAILABLE"
2. ✓ Environment Configuration (BITNET_CPP_DIR, LD_LIBRARY_PATH, etc.)
3. ✓ Library Search Paths showing 7 paths with context labels:
   - Path 2: `build` (no label)
   - Path 3: `build/bin (standalone llama.cpp)`
   - Path 4: `build/lib` (no label)
   - Path 5: `build/3rdparty/llama.cpp/src (embedded llama.cpp)`
   - Path 6: `build/3rdparty/llama.cpp/ggml/src (embedded ggml)`
4. ✓ Required Libraries: `✓ libbitnet.so (found at build time)`
5. ✓ Build Metadata with "Required libraries: libbitnet"
6. ✓ Platform-Specific Configuration
7. ✓ Summary with example command

**Test Scenario 2: Failure Path with Recovery Instructions**

```bash
# Prerequisite: No C++ libraries installed
BITNET_CPP_DIR="" cargo run -p xtask -- preflight --backend llama
```

**Expected output:**
1. ✓ Error header: "Backend 'llama.cpp' libraries NOT FOUND"
2. ✓ CRITICAL explanation about build-time detection
3. ✓ Required libraries: `libllama.so, libggml.so`
4. ✓ RECOVERY STEPS with unified setup command (no `--bitnet` flag):
   ```
   eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
   ```
5. ✓ TROUBLESHOOTING section with verbose flag suggestion

**Test Scenario 3: Cross-Backend Consistency**

```bash
# Compare BitNet vs LLaMA output structure
cargo run -p xtask -- preflight --backend bitnet --verbose > /tmp/bitnet-verbose.txt
cargo run -p xtask -- preflight --backend llama --verbose > /tmp/llama-verbose.txt

# Verify sections match (excluding library-specific content)
diff -u /tmp/bitnet-verbose.txt /tmp/llama-verbose.txt
```

**Expected**:
- ✓ Identical section headers
- ✓ Identical environment variable layout
- ✓ Same number of search paths (7)
- ✓ Same context label patterns
- ✓ Identical setup command (no `--bitnet` difference)

### 5.3 Platform-Specific Tests

**Linux:**
```bash
# Verify LD_LIBRARY_PATH handling
cargo run -p xtask -- preflight --backend bitnet --verbose | grep "LD_LIBRARY_PATH"
```

**macOS:**
```bash
# Verify DYLD_LIBRARY_PATH handling
cargo run -p xtask -- preflight --backend llama --verbose | grep "DYLD_LIBRARY_PATH"
```

**Windows (WSL/native):**
```bash
# Verify PATH handling
cargo run -p xtask -- preflight --backend bitnet --verbose | grep "PATH"
```

---

## 6. Verification Criteria

### 6.1 Success Metrics

#### Metric 1: Setup Command Consistency

**Test**:
```bash
cargo run -p xtask -- preflight --backend bitnet 2>&1 | grep "setup-cpp-auto"
cargo run -p xtask -- preflight --backend llama 2>&1 | grep "setup-cpp-auto"
```

**Pass Criteria**:
- Both backends show identical command
- No `--bitnet` flag in either command
- Command uses `setup-cpp-auto --emit=sh`

#### Metric 2: Search Path Coverage

**Test**:
```bash
cargo run -p xtask -- preflight --backend llama --verbose 2>&1 | grep "build/bin"
```

**Pass Criteria**:
- `build/bin` appears in search path list
- Appears before `build/lib` (priority 3)
- Shows context label: "(standalone llama.cpp)"

#### Metric 3: Context Label Clarity

**Test**:
```bash
cargo run -p xtask -- preflight --backend bitnet --verbose 2>&1 | grep "embedded"
```

**Pass Criteria**:
- Paths with `3rdparty/llama.cpp/src` show "(embedded llama.cpp)"
- Paths with `3rdparty/llama.cpp/ggml` show "(embedded ggml)"
- Standard paths (`build`, `build/lib`, `lib`) have no labels

#### Metric 4: Build Metadata Completeness

**Test**:
```bash
cargo run -p xtask -- preflight --backend bitnet --verbose 2>&1 | grep "Required libraries"
cargo run -p xtask -- preflight --backend llama --verbose 2>&1 | grep "Required libraries"
```

**Pass Criteria**:
- BitNet shows: "Required libraries: libbitnet"
- LLaMA shows: "Required libraries: libllama, libggml"
- Line appears in Build-Time Detection Metadata section

### 6.2 Backward Compatibility

#### Check 1: Existing Scripts Continue Working

**Test**:
```bash
# Old command with --bitnet flag should still work
eval "$(cargo run -p xtask -- setup-cpp-auto --bitnet --emit=sh)"
echo $?  # Should be 0 (success)
```

**Pass Criteria**: Exit code 0, no errors

#### Check 2: Output Structure Preserved

**Test**:
```bash
# Count section headers (should still be 7 in verbose success)
cargo run -p xtask -- preflight --backend bitnet --verbose 2>&1 | \
  grep -E "^(Environment Configuration|Library Search Paths|Required Libraries|Build-Time|Platform-Specific|Summary)" | \
  wc -l
```

**Pass Criteria**: 6 section headers (same as before)

#### Check 3: Error Format Unchanged

**Test**:
```bash
# Error message structure should be preserved
BITNET_CPP_DIR="" cargo run -p xtask -- preflight --backend bitnet 2>&1 | \
  grep -E "(RECOVERY STEPS|TROUBLESHOOTING|Option A|Option B)"
```

**Pass Criteria**: All 4 keywords present in error output

### 6.3 Cross-Platform Verification

**Platform matrix**:

| Platform | Loader Var | Test Command |
|----------|-----------|--------------|
| Linux | LD_LIBRARY_PATH | `cargo run -p xtask -- preflight --backend bitnet --verbose \| grep LD_LIBRARY_PATH` |
| macOS | DYLD_LIBRARY_PATH | Same with grep DYLD_LIBRARY_PATH |
| Windows/WSL | PATH | Same with grep "PATH" |

**Pass Criteria**: Correct platform-specific variable displayed in Environment Configuration section

---

## 7. Appendices

### 7.1 File Locations Reference

| Component | File | Lines |
|-----------|------|-------|
| Setup command | `xtask/src/crossval/backend.rs` | 100-105 |
| Build-time search paths | `crossval/build.rs` | 70-76 |
| Runtime search path display | `xtask/src/crossval/preflight.rs` | 555-579 |
| Path context labels | `xtask/src/crossval/preflight.rs` | After 603 (new) |
| Verbose success output | `xtask/src/crossval/preflight.rs` | 191-329 |
| Verbose failure output | `xtask/src/crossval/preflight.rs` | 332-528 |
| Build metadata formatting | `xtask/src/crossval/preflight.rs` | 653-674 |
| Backend constants | `crossval/src/lib.rs` | 23-30 |

### 7.2 Search Path Priority Visualization

```
Priority 0: BITNET_CROSSVAL_LIBDIR (if set)
            └─ Explicit user override

Priority 1: $BITNET_CPP_DIR/build
            └─ CMake main output (both backends)

Priority 2: $BITNET_CPP_DIR/build/bin  [NEW]
            └─ Standalone llama.cpp (common layout)

Priority 3: $BITNET_CPP_DIR/build/lib
            └─ Library subdirectory (both backends)

Priority 4: $BITNET_CPP_DIR/build/3rdparty/llama.cpp/src
            └─ Embedded llama.cpp in BitNet (BitNet only)

Priority 5: $BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src
            └─ Embedded ggml in BitNet (BitNet only)

Priority 6: $BITNET_CPP_DIR/lib
            └─ Top-level library directory (alternative layout)
```

### 7.3 Output Format Comparison

**Before Enhancements:**
```
Library Search Paths (Priority Order)
─────────────────────────────────────────────────────────
  1. BITNET_CROSSVAL_LIBDIR override
     (not set - using default search order)

  2. BITNET_CPP_DIR/build
     ✓ /home/user/.cache/bitnet_cpp/build (exists)
     Found libraries:
       - libbitnet.so

  3. BITNET_CPP_DIR/build/lib
     ✗ /home/user/.cache/bitnet_cpp/build/lib (not found)

Build-Time Detection Metadata
─────────────────────────────────────────────────────────
CROSSVAL_HAS_BITNET = true
Last xtask build: 2025-10-25 14:32:18 UTC
Build feature flags: crossval-all
```

**After Enhancements:**
```
Library Search Paths (Priority Order)
─────────────────────────────────────────────────────────
  1. BITNET_CROSSVAL_LIBDIR override
     (not set - using default search order)

  2. BITNET_CPP_DIR/build
     ✓ /home/user/.cache/bitnet_cpp/build (exists)
     Found libraries:
       - libbitnet.so

  3. BITNET_CPP_DIR/build/bin (standalone llama.cpp)
     ✗ /home/user/.cache/bitnet_cpp/build/bin (not found)

  4. BITNET_CPP_DIR/build/lib
     ✗ /home/user/.cache/bitnet_cpp/build/lib (not found)

  5. BITNET_CPP_DIR/build/3rdparty/llama.cpp/src (embedded llama.cpp)
     ✗ /home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src (not found)

Build-Time Detection Metadata
─────────────────────────────────────────────────────────
CROSSVAL_HAS_BITNET = true
Required libraries: libbitnet
Last xtask build: 2025-10-25 14:32:18 UTC
Build feature flags: crossval-all
```

### 7.4 Implementation Checklist

- [ ] Phase 1: Update `backend.rs` `setup_command()` to unified command
- [ ] Phase 1: Add unit test for setup command consistency
- [ ] Phase 2: Add `build/bin` to `crossval/build.rs` search paths
- [ ] Phase 2: Add `build/bin` to `preflight.rs` `get_library_search_paths()`
- [ ] Phase 2: Rebuild crossval + xtask, verify 7 paths shown
- [ ] Phase 3: Add `get_path_context_label()` helper function
- [ ] Phase 3: Integrate context labels into verbose success output
- [ ] Phase 3: Integrate context labels into verbose failure output
- [ ] Phase 3: Add unit test for context label generation
- [ ] Phase 4: Update `format_build_metadata()` to include required libraries
- [ ] Phase 4: Add unit test for metadata format
- [ ] All: Run manual verification checklist (3 scenarios)
- [ ] All: Verify platform-specific output (Linux/macOS/Windows)
- [ ] All: Verify backward compatibility (old commands still work)
- [ ] All: Update CLAUDE.md if preflight usage changes (unlikely)

### 7.5 Related Documentation

- `docs/howto/cpp-setup.md` - C++ reference setup guide
- `docs/explanation/dual-backend-crossval.md` - Dual-backend architecture
- `xtask/src/crossval/README.md` - Cross-validation tooling (if exists)
- `CLAUDE.md` - Project conventions and workflows

---

## 8. Summary

This specification addresses four UX gaps in the preflight command:

1. **Setup Command Inconsistency** (5 min, very low risk): Unify BitNet and LLaMA setup commands by removing redundant `--bitnet` flag
2. **Missing Search Path** (15 min, low risk): Add `build/bin` to support standalone llama.cpp installations
3. **Generic Path Labels** (20 min, very low risk): Add context labels to clarify embedded vs standalone paths
4. **Incomplete Metadata** (10 min, very low risk): Show required library names in build metadata

**Total estimated effort**: ~50 minutes
**Overall risk**: Very Low (all changes are backward compatible and append-only)

**Success criteria**:
- Both backends show identical setup command
- 7 search paths displayed with appropriate context labels
- Build metadata shows what was searched for
- Existing scripts and workflows continue working without modification

**Next steps**:
1. Implement Phase 1 (setup command unification)
2. Add unit tests
3. Implement Phases 2-4 sequentially
4. Run manual verification checklist
5. Update this spec status to "Implemented" with PR reference
