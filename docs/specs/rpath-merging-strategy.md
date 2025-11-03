# RPATH Merging Strategy: Unified Multi-Backend Library Path Resolution

**Version:** 1.0.0
**Status:** Draft
**Created:** 2025-10-25
**Component:** `xtask/build.rs`
**Feature Scope:** Cross-validation infrastructure (requires `crossval`, `crossval-all`, or `ffi` features)

---

## 1. Problem Statement

### 1.1 Current Limitation

The current `xtask/build.rs` RPATH embedding logic assumes all C++ backend libraries (BitNet.cpp and llama.cpp) reside in a **single directory**. This is achieved through:

```rust
// Current implementation (xtask/build.rs lines 25-59)
fn embed_crossval_rpath() {
    // Priority 1: BITNET_CROSSVAL_LIBDIR (single directory)
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        emit_rpath(&lib_dir);  // Single path only
        return;
    }

    // Priority 2: BITNET_CPP_DIR/build/bin (assumes bundled layout)
    // ...
}
```

**Why this works today:**
- BitNet.cpp's CMake build layout bundles llama.cpp libraries in the same directory:
  ```
  ~/.cache/bitnet_cpp/build/bin/
  ├── libbitnet.so
  ├── libllama.so
  └── libggml.so
  ```

**Why this is insufficient:**
- **Standalone llama.cpp builds**: Users may build llama.cpp independently (e.g., via system package manager or custom install location)
- **BitNet-only deployments**: Some users only need BitNet.cpp libraries, not llama.cpp
- **Docker/CI scenarios**: Multi-stage builds may place libraries in separate optimized locations
- **Development workflows**: Developers may want to test against different llama.cpp versions without rebuilding BitNet.cpp

### 1.2 User Impact

Without multi-path RPATH support, users encounter these failure modes:

**Scenario A: Separate llama.cpp installation**
```bash
# User has:
#   - BitNet.cpp in /opt/bitnet/lib/libbitnet.so
#   - llama.cpp in /usr/local/lib/libllama.so

export BITNET_CROSSVAL_LIBDIR=/opt/bitnet/lib  # Only sets BitNet path
cargo build -p xtask --features crossval-all

# Runtime error:
cargo run -p xtask -- crossval-per-token --model test.gguf
# Error: libllama.so: cannot open shared object file: No such file or directory
```

**Workaround (cumbersome):**
```bash
# Create symlink directory (fragile)
mkdir -p /tmp/merged_libs
ln -s /opt/bitnet/lib/libbitnet.so /tmp/merged_libs/
ln -s /usr/local/lib/libllama.so /tmp/merged_libs/
export BITNET_CROSSVAL_LIBDIR=/tmp/merged_libs
cargo clean -p xtask && cargo build -p xtask --features crossval-all
```

### 1.3 Requirements

**Functional Requirements:**

1. **FR1**: Support separate library paths for BitNet and llama backends via distinct environment variables
2. **FR2**: Merge multiple RPATH entries into a single colon-separated value compatible with `-Wl,-rpath` syntax
3. **FR3**: Deduplicate identical paths to avoid redundant search overhead
4. **FR4**: Preserve relative ordering (BitNet paths before llama paths) for deterministic search behavior
5. **FR5**: Maintain backward compatibility with existing `BITNET_CROSSVAL_LIBDIR` (single-path override)
6. **FR6**: Emit diagnostic warnings showing merged RPATH for debugging and transparency

**Non-Functional Requirements:**

1. **NFR1**: Zero runtime overhead (RPATH resolution occurs at binary load time, not during execution)
2. **NFR2**: Platform-aware: Linux/macOS use `-Wl,-rpath`, Windows use PATH (no RPATH)
3. **NFR3**: Graceful degradation: Build succeeds even if libraries are missing (STUB mode)
4. **NFR4**: Developer ergonomics: Clear error messages if paths are invalid or libraries not found

---

## 2. Architecture

### 2.1 Environment Variable Schema

Introduce **two new environment variables** for fine-grained control:

| Variable | Purpose | Example | Priority |
|----------|---------|---------|----------|
| `CROSSVAL_RPATH_BITNET` | Path to BitNet.cpp libraries | `/opt/bitnet/lib` | 2 |
| `CROSSVAL_RPATH_LLAMA` | Path to llama.cpp libraries | `/usr/local/lib` | 3 |
| `BITNET_CROSSVAL_LIBDIR` | **Legacy**: Single directory override | `/tmp/merged_libs` | 1 (backward compat) |
| `BITNET_CPP_DIR` | **Fallback**: Auto-detect from build root | `~/.cache/bitnet_cpp` | 4 |

**Priority Order** (highest to lowest):

1. **`BITNET_CROSSVAL_LIBDIR`** (if set and exists) → Use as-is, skip merging (backward compatibility)
2. **`CROSSVAL_RPATH_BITNET` + `CROSSVAL_RPATH_LLAMA`** → Merge with deduplication
3. **Fallback to `BITNET_CPP_DIR/build/bin`** → Existing auto-discovery logic

### 2.2 Merge Algorithm

**Pseudocode:**

```rust
fn merge_rpath_entries() -> Option<String> {
    let mut rpath_dirs: Vec<PathBuf> = Vec::new();

    // Priority 1: Legacy single-directory override (backward compatibility)
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        let path = Path::new(&lib_dir);
        if path.exists() {
            return Some(path.display().to_string());  // No merge, use as-is
        } else {
            warn!("BITNET_CROSSVAL_LIBDIR set but directory does not exist: {}", lib_dir);
        }
    }

    // Priority 2: Read new granular environment variables
    if let Ok(bitnet_path) = env::var("CROSSVAL_RPATH_BITNET") {
        let path = Path::new(&bitnet_path);
        if path.exists() {
            rpath_dirs.push(path.to_path_buf());
        } else {
            warn!("CROSSVAL_RPATH_BITNET set but directory does not exist: {}", bitnet_path);
        }
    }

    if let Ok(llama_path) = env::var("CROSSVAL_RPATH_LLAMA") {
        let path = Path::new(&llama_path);
        if path.exists() {
            rpath_dirs.push(path.to_path_buf());
        } else {
            warn!("CROSSVAL_RPATH_LLAMA set but directory does not exist: {}", llama_path);
        }
    }

    // If we collected paths, merge and deduplicate
    if !rpath_dirs.is_empty() {
        return Some(merge_and_deduplicate(rpath_dirs));
    }

    // Priority 3: Fallback to BITNET_CPP_DIR auto-discovery (existing logic)
    if let Ok(cpp_dir) = env::var("BITNET_CPP_DIR") {
        let build_bin = Path::new(&cpp_dir).join("build/bin");
        if build_bin.exists() {
            return Some(build_bin.display().to_string());
        }

        let build_dir = Path::new(&cpp_dir).join("build");
        if build_dir.exists() {
            return Some(build_dir.display().to_string());
        }
    }

    None  // No RPATH available (graceful fallback)
}

fn merge_and_deduplicate(paths: Vec<PathBuf>) -> String {
    let mut seen = HashSet::new();
    let mut merged = Vec::new();

    for path in paths {
        // Canonicalize to resolve symlinks and normalize paths
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                warn!("Failed to canonicalize path: {}", path.display());
                continue;  // Skip invalid paths
            }
        };

        // Deduplicate using canonical path
        if seen.insert(canonical.clone()) {
            merged.push(canonical);
        }
    }

    // Join with colon separator (POSIX RPATH syntax)
    merged
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(":")
}
```

### 2.3 Deduplication Strategy

**Goal**: Avoid redundant RPATH entries while preserving search order.

**Algorithm**:
1. **Canonicalize paths**: Resolve symlinks, `.`, `..` to absolute normalized paths
2. **Use HashSet for uniqueness**: Track seen canonical paths (case-sensitive on Linux, case-insensitive on macOS)
3. **Preserve insertion order**: Use `Vec` to maintain BitNet → llama ordering

**Edge Cases**:

| Scenario | Input | Output | Rationale |
|----------|-------|--------|-----------|
| **Identical paths** | `BITNET=/opt/lib`, `LLAMA=/opt/lib` | `/opt/lib` | Deduplicate exact matches |
| **Symlink equivalence** | `BITNET=/opt/lib`, `LLAMA=/opt/bitnet` (symlink to `/opt/lib`) | `/opt/lib` | Canonicalize resolves symlinks |
| **Non-existent path** | `BITNET=/invalid/path` | (skip) | Emit warning, omit from RPATH |
| **Relative paths** | `BITNET=./build/lib` | `/abs/path/to/build/lib` | Canonicalize converts to absolute |

**Platform-Specific Considerations**:

- **Linux**: Paths are case-sensitive (`/Opt/Lib` ≠ `/opt/lib`)
- **macOS**: APFS is case-insensitive by default (`/Opt/Lib` == `/opt/lib`)
  - Solution: Use `canonicalize()` which normalizes case on macOS
- **Windows**: N/A (Windows uses PATH, not RPATH)

### 2.4 Platform Handling

```rust
fn emit_merged_rpath(merged_rpath: &str) {
    println!("cargo:rustc-link-search=native={}", merged_rpath);

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", merged_rpath);
        println!(
            "cargo:warning=xtask: Embedded merged RPATH for runtime loader: {}",
            merged_rpath
        );
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", merged_rpath);
        println!(
            "cargo:warning=xtask: Embedded merged RPATH for runtime loader: {}",
            merged_rpath
        );
    }

    #[cfg(target_os = "windows")]
    {
        println!(
            "cargo:warning=xtask: Windows detected - RPATH not applicable. \
             Ensure libraries are in PATH or use setup-cpp-auto. \
             Merged paths: {}",
            merged_rpath
        );
    }
}
```

**Windows Notes**:
- RPATH concept does not exist on Windows (ELF/Mach-O specific)
- Windows loader searches:
  1. Executable directory
  2. Current working directory
  3. System directories (System32, etc.)
  4. `PATH` environment variable
- Users must set `PATH` before running xtask on Windows (documented in `docs/howto/cpp-setup.md`)

---

## 3. API Contracts

### 3.1 Input Environment Variables

| Variable | Type | Required | Default | Validation |
|----------|------|----------|---------|------------|
| `CROSSVAL_RPATH_BITNET` | Absolute path | Optional | (none) | Must exist as directory, warn if not |
| `CROSSVAL_RPATH_LLAMA` | Absolute path | Optional | (none) | Must exist as directory, warn if not |
| `BITNET_CROSSVAL_LIBDIR` | Absolute path | Optional | (none) | Override: skips merging if set |
| `BITNET_CPP_DIR` | Absolute path | Optional | (none) | Fallback: auto-detect `build/bin/` |

**Contract Guarantees**:
1. **Backward Compatibility**: `BITNET_CROSSVAL_LIBDIR` takes precedence over new variables
2. **Graceful Fallback**: Missing variables default to auto-discovery from `BITNET_CPP_DIR`
3. **Validation**: Emit `cargo:warning` for invalid paths, omit from RPATH (do not fail build)
4. **No Surprise Dependencies**: Missing libraries result in STUB mode (cross-validation unavailable), not build failure

### 3.2 Output Rustc Directives

**Linux/macOS**:
```
cargo:rustc-link-search=native=/path1:/path2
cargo:rustc-link-arg=-Wl,-rpath,/path1:/path2
cargo:warning=xtask: Embedded merged RPATH for runtime loader: /path1:/path2
```

**Windows**:
```
cargo:rustc-link-search=native=/path1:/path2
cargo:warning=xtask: Windows detected - RPATH not applicable. Merged paths: /path1:/path2
```

**STUB Mode (no libraries found)**:
```
(no rustc directives emitted)
```

### 3.3 Rerun Triggers

Update `xtask/build.rs` to watch new environment variables:

```rust
println!("cargo:rerun-if-changed=build.rs");
println!("cargo:rerun-if-env-changed=BITNET_CROSSVAL_LIBDIR");  // Existing
println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");          // Existing
println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_BITNET");  // NEW
println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_LLAMA");   // NEW
```

**Behavior**: Changing any of these variables triggers `xtask` rebuild with new RPATH.

---

## 4. Implementation Plan

### 4.1 Code Locations

| File | Lines | Modification |
|------|-------|--------------|
| `xtask/build.rs` | 1-5 | Add `rerun-if-env-changed` for new vars |
| `xtask/build.rs` | 25-59 | Replace `embed_crossval_rpath()` with merge logic |
| `xtask/build.rs` | 61-85 | Update `emit_rpath()` to accept merged string |

### 4.2 Step-by-Step Implementation

**Step 1: Add rerun triggers** (lines 1-5)
```rust
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITNET_CROSSVAL_LIBDIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_BITNET");  // NEW
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_LLAMA");   // NEW

    #[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
    embed_crossval_rpath();
}
```

**Step 2: Replace `embed_crossval_rpath()` function** (lines 25-59)
```rust
#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
fn embed_crossval_rpath() {
    use std::{
        collections::HashSet,
        env,
        path::{Path, PathBuf},
    };

    // Priority 1: Legacy single-directory override (backward compatibility)
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        let lib_path = Path::new(&lib_dir);
        if lib_path.exists() {
            emit_rpath(lib_path.display().to_string());
            return;
        } else {
            println!(
                "cargo:warning=xtask: BITNET_CROSSVAL_LIBDIR set but directory does not exist: {}",
                lib_dir
            );
        }
    }

    // Priority 2: Read granular environment variables and merge
    let mut rpath_candidates: Vec<PathBuf> = Vec::new();

    if let Ok(bitnet_path) = env::var("CROSSVAL_RPATH_BITNET") {
        let path = Path::new(&bitnet_path);
        if path.exists() {
            rpath_candidates.push(path.to_path_buf());
        } else {
            println!(
                "cargo:warning=xtask: CROSSVAL_RPATH_BITNET set but directory does not exist: {}",
                bitnet_path
            );
        }
    }

    if let Ok(llama_path) = env::var("CROSSVAL_RPATH_LLAMA") {
        let path = Path::new(&llama_path);
        if path.exists() {
            rpath_candidates.push(path.to_path_buf());
        } else {
            println!(
                "cargo:warning=xtask: CROSSVAL_RPATH_LLAMA set but directory does not exist: {}",
                llama_path
            );
        }
    }

    // Merge and deduplicate if we have candidates
    if !rpath_candidates.is_empty() {
        let merged = merge_and_deduplicate(rpath_candidates);
        emit_rpath(merged);
        return;
    }

    // Priority 3: Fallback to BITNET_CPP_DIR auto-discovery (existing logic)
    if let Ok(cpp_dir) = env::var("BITNET_CPP_DIR") {
        let build_bin = Path::new(&cpp_dir).join("build/bin");
        if build_bin.exists() {
            emit_rpath(build_bin.display().to_string());
            return;
        }

        let build_dir = Path::new(&cpp_dir).join("build");
        if build_dir.exists() {
            emit_rpath(build_dir.display().to_string());
        }
    }

    // No library directory found - graceful fallback (STUB mode)
}

#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
fn merge_and_deduplicate(paths: Vec<PathBuf>) -> String {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    let mut merged = Vec::new();

    for path in paths {
        // Canonicalize to resolve symlinks and normalize case (macOS)
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "cargo:warning=xtask: Failed to canonicalize path {}: {}. Skipping.",
                    path.display(),
                    e
                );
                continue; // Skip invalid paths
            }
        };

        // Deduplicate using canonical path
        if seen.insert(canonical.clone()) {
            merged.push(canonical);
        }
    }

    // Join with colon separator (POSIX RPATH syntax)
    merged.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(":")
}
```

**Step 3: Update `emit_rpath()` to accept String** (lines 61-85)
```rust
#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
fn emit_rpath(rpath: String) {
    // Emit link search directive (compile-time)
    // Note: rustc-link-search does NOT support colon-separated paths,
    // so we emit the first path only for link-time resolution.
    // The full merged path is used in rustc-link-arg for runtime RPATH.
    let first_path = rpath.split(':').next().unwrap_or(&rpath);
    println!("cargo:rustc-link-search=native={}", first_path);

    // Emit RPATH for runtime library resolution
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
        println!("cargo:warning=xtask: Embedded merged RPATH for runtime loader: {}", rpath);
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
        println!("cargo:warning=xtask: Embedded merged RPATH for runtime loader: {}", rpath);
    }

    #[cfg(target_os = "windows")]
    {
        println!(
            "cargo:warning=xtask: Windows detected - RPATH not applicable. \
             Ensure libraries are in PATH. Merged paths: {}",
            rpath
        );
    }
}
```

### 4.3 Migration Path for Existing Deployments

**Scenario 1: No action needed (bitnet.cpp bundled layout)**
```bash
# Current setup (works as-is)
export BITNET_CPP_DIR=~/.cache/bitnet_cpp
cargo build -p xtask --features crossval-all

# After implementation: No changes required
# Auto-discovery from BITNET_CPP_DIR/build/bin still works
```

**Scenario 2: Adopt granular variables (separate llama.cpp)**
```bash
# Before (workaround with symlinks)
mkdir -p /tmp/merged_libs
ln -s /opt/bitnet/lib/libbitnet.so /tmp/merged_libs/
ln -s /usr/local/lib/libllama.so /tmp/merged_libs/
export BITNET_CROSSVAL_LIBDIR=/tmp/merged_libs

# After (clean, no symlinks)
export CROSSVAL_RPATH_BITNET=/opt/bitnet/lib
export CROSSVAL_RPATH_LLAMA=/usr/local/lib
cargo build -p xtask --features crossval-all

# Verification
readelf -d target/debug/xtask | grep RPATH
# Expected: 0x000000000000000f (RPATH) Library rpath: [/opt/bitnet/lib:/usr/local/lib]
```

**Scenario 3: Override with legacy variable (backward compat)**
```bash
# Existing scripts using BITNET_CROSSVAL_LIBDIR
export BITNET_CROSSVAL_LIBDIR=/custom/path
cargo build -p xtask --features crossval-all

# After implementation: Still works (priority 1)
# New variables ignored if BITNET_CROSSVAL_LIBDIR is set
```

---

## 5. Test Requirements

### 5.1 Unit Tests

Create `xtask/tests/build_rs_rpath_merge.rs` (integration test since `build.rs` is not a lib):

```rust
#[test]
fn test_merge_single_path() {
    let paths = vec![PathBuf::from("/opt/bitnet/lib")];
    let result = merge_and_deduplicate(paths);
    assert_eq!(result, "/opt/bitnet/lib");
}

#[test]
fn test_merge_two_distinct_paths() {
    let paths = vec![
        PathBuf::from("/opt/bitnet/lib"),
        PathBuf::from("/usr/local/lib"),
    ];
    let result = merge_and_deduplicate(paths);
    assert_eq!(result, "/opt/bitnet/lib:/usr/local/lib");
}

#[test]
fn test_deduplicate_identical_paths() {
    let paths = vec![
        PathBuf::from("/opt/lib"),
        PathBuf::from("/opt/lib"),
    ];
    let result = merge_and_deduplicate(paths);
    assert_eq!(result, "/opt/lib");
}

#[test]
fn test_canonicalize_symlink() {
    // Create temp directories
    let temp = tempfile::tempdir().unwrap();
    let real_dir = temp.path().join("real");
    let symlink_dir = temp.path().join("link");

    std::fs::create_dir(&real_dir).unwrap();
    std::os::unix::fs::symlink(&real_dir, &symlink_dir).unwrap();

    let paths = vec![real_dir.clone(), symlink_dir];
    let result = merge_and_deduplicate(paths);

    // Should deduplicate to single canonical path
    assert_eq!(result.matches(':').count(), 0); // No colon = single path
}

#[test]
fn test_ordering_preserved() {
    let paths = vec![
        PathBuf::from("/path/bitnet"),
        PathBuf::from("/path/llama"),
    ];
    let result = merge_and_deduplicate(paths);
    assert_eq!(result, "/path/bitnet:/path/llama");
}

#[test]
fn test_invalid_path_skipped() {
    let paths = vec![
        PathBuf::from("/valid/path"),
        PathBuf::from("/this/does/not/exist"),
    ];
    // merge_and_deduplicate will canonicalize, failing for invalid path
    // Expected: warning emitted, invalid path skipped
}
```

**Note**: Since `build.rs` is not a library, these functions need to be extracted into a separate module for testability. Consider:
1. Move merge logic to `xtask/src/build_helpers.rs`
2. Import in `build.rs` and tests

### 5.2 Integration Tests

**Test Plan**:

| Test Case | Setup | Verification |
|-----------|-------|--------------|
| **IT1: Legacy override** | Set `BITNET_CROSSVAL_LIBDIR=/tmp/test` | `readelf` shows single path `/tmp/test` |
| **IT2: Granular merge** | Set `CROSSVAL_RPATH_BITNET=/a`, `CROSSVAL_RPATH_LLAMA=/b` | `readelf` shows `/a:/b` |
| **IT3: Deduplication** | Set both vars to same path | `readelf` shows single path |
| **IT4: Fallback** | Unset all vars, set `BITNET_CPP_DIR` | `readelf` shows `$BITNET_CPP_DIR/build/bin` |
| **IT5: STUB mode** | Unset all vars | No RPATH in binary, cross-validation fails gracefully |
| **IT6: Invalid path** | Set `CROSSVAL_RPATH_BITNET=/invalid` | Build warning emitted, RPATH omits invalid path |

**Execution**:
```bash
# IT1: Legacy override
export BITNET_CROSSVAL_LIBDIR=/tmp/test_lib
mkdir -p /tmp/test_lib
cargo clean -p xtask
cargo build -p xtask --features crossval-all
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [/tmp/test_lib]

# IT2: Granular merge
unset BITNET_CROSSVAL_LIBDIR
export CROSSVAL_RPATH_BITNET=/tmp/bitnet_test
export CROSSVAL_RPATH_LLAMA=/tmp/llama_test
mkdir -p /tmp/bitnet_test /tmp/llama_test
cargo clean -p xtask
cargo build -p xtask --features crossval-all
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [/tmp/bitnet_test:/tmp/llama_test]

# IT3: Deduplication
export CROSSVAL_RPATH_BITNET=/tmp/shared
export CROSSVAL_RPATH_LLAMA=/tmp/shared
mkdir -p /tmp/shared
cargo clean -p xtask
cargo build -p xtask --features crossval-all
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [/tmp/shared]

# IT6: Invalid path handling
export CROSSVAL_RPATH_BITNET=/invalid/path/does/not/exist
export CROSSVAL_RPATH_LLAMA=/tmp/valid
mkdir -p /tmp/valid
cargo clean -p xtask
cargo build -p xtask --features crossval-all 2>&1 | grep warning
# Expected: cargo:warning=xtask: CROSSVAL_RPATH_BITNET set but directory does not exist
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [/tmp/valid]
```

### 5.3 Regression Tests

**Ensure backward compatibility**:

```bash
# Regression 1: Existing BITNET_CROSSVAL_LIBDIR still works
export BITNET_CROSSVAL_LIBDIR=/opt/merged
mkdir -p /opt/merged
cargo clean -p xtask
cargo build -p xtask --features crossval-all
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [/opt/merged]

# Regression 2: Existing BITNET_CPP_DIR fallback still works
unset BITNET_CROSSVAL_LIBDIR
export BITNET_CPP_DIR=~/.cache/bitnet_cpp
cargo clean -p xtask
cargo build -p xtask --features crossval-all
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [~/.cache/bitnet_cpp/build/bin] (if exists)

# Regression 3: STUB mode still graceful
unset BITNET_CROSSVAL_LIBDIR
unset BITNET_CPP_DIR
unset CROSSVAL_RPATH_BITNET
unset CROSSVAL_RPATH_LLAMA
cargo clean -p xtask
cargo build -p xtask --features crossval-all
# Expected: Build succeeds, no RPATH emitted
cargo run -p xtask -- preflight --backend bitnet
# Expected: Error with helpful message about setup-cpp-auto
```

---

## 6. Verification Criteria

### 6.1 Build-Time Verification

**Success Criteria**:

1. **VC1**: `cargo build -p xtask --features crossval-all` succeeds with or without environment variables
2. **VC2**: Build warnings clearly indicate:
   - Which RPATH entries were merged (if any)
   - Invalid paths detected and skipped
   - Fallback to auto-discovery if no vars set
3. **VC3**: `cargo:rerun-if-env-changed` triggers rebuild when any RPATH variable changes

**Verification Commands**:
```bash
# Check build warnings
cargo clean -p xtask
export CROSSVAL_RPATH_BITNET=/tmp/a
export CROSSVAL_RPATH_LLAMA=/tmp/b
mkdir -p /tmp/a /tmp/b
cargo build -p xtask --features crossval-all 2>&1 | grep "Embedded merged RPATH"
# Expected: cargo:warning=xtask: Embedded merged RPATH for runtime loader: /tmp/a:/tmp/b

# Check rerun trigger
touch xtask/build.rs
cargo build -p xtask --features crossval-all
# Expected: Rebuild triggered

export CROSSVAL_RPATH_BITNET=/tmp/c
mkdir -p /tmp/c
cargo build -p xtask --features crossval-all
# Expected: Rebuild triggered (env var changed)
```

### 6.2 Runtime Verification

**Success Criteria**:

1. **VC4**: `readelf -d target/debug/xtask | grep RPATH` shows merged paths on Linux
2. **VC5**: `otool -l target/debug/xtask | grep -A2 LC_RPATH` shows merged paths on macOS
3. **VC6**: `ldd target/debug/xtask` resolves libraries from merged RPATH paths
4. **VC7**: `cargo run -p xtask -- preflight --backend bitnet` succeeds if libraries in RPATH

**Verification Commands (Linux)**:
```bash
# Set up separate library directories
export CROSSVAL_RPATH_BITNET=/opt/bitnet/lib
export CROSSVAL_RPATH_LLAMA=/usr/local/lib
mkdir -p /opt/bitnet/lib /usr/local/lib

# Build with merged RPATH
cargo clean -p xtask
cargo build -p xtask --features crossval-all

# Verify embedded RPATH
readelf -d target/debug/xtask | grep RPATH
# Expected: 0x000000000000000f (RPATH) Library rpath: [/opt/bitnet/lib:/usr/local/lib]

# Verify runtime resolution
ldd target/debug/xtask | grep libbitnet
# Expected: libbitnet.so => /opt/bitnet/lib/libbitnet.so (0x...)

ldd target/debug/xtask | grep libllama
# Expected: libllama.so => /usr/local/lib/libllama.so (0x...)

# Test runtime execution
cargo run -p xtask -- preflight --backend bitnet --verbose
# Expected: ✓ Backend 'bitnet.cpp': AVAILABLE
```

**Verification Commands (macOS)**:
```bash
# Similar to Linux, using otool instead of readelf
otool -l target/debug/xtask | grep -A 3 "Load command.*LC_RPATH"
# Expected: Shows both /opt/bitnet/lib and /usr/local/lib
```

### 6.3 Acceptance Criteria

**AC1**: Backward Compatibility
- Existing `BITNET_CROSSVAL_LIBDIR` usage continues to work without modification
- Existing `BITNET_CPP_DIR` fallback continues to work

**AC2**: Functional Correctness
- Merged RPATH contains all specified library directories
- Deduplication eliminates redundant paths
- Ordering is preserved (BitNet before llama)

**AC3**: Error Handling
- Invalid paths emit warnings and are skipped (do not fail build)
- Missing all variables results in STUB mode (graceful degradation)
- Clear diagnostic messages in build output

**AC4**: Platform Support
- Linux: RPATH embedded via `-Wl,-rpath`
- macOS: RPATH embedded via `-Wl,-rpath`
- Windows: Warning emitted, instructions for PATH setup

**AC5**: Developer Experience
- No breaking changes to existing workflows
- Clear migration path documented
- Diagnostic warnings aid troubleshooting

---

## 7. Risks and Mitigations

### 7.1 Identified Risks

| Risk ID | Description | Impact | Probability | Mitigation |
|---------|-------------|--------|-------------|------------|
| **R1** | Path canonicalization fails on network mounts | Build warnings | Low | Skip invalid paths, emit warnings |
| **R2** | Case sensitivity issues on macOS | Duplicate paths | Medium | Use `canonicalize()` which normalizes case |
| **R3** | Symlink loops cause infinite recursion | Build hangs | Very Low | `canonicalize()` detects loops (returns error) |
| **R4** | User sets both legacy and new vars | Confusion | Medium | Document priority order, emit warning |
| **R5** | Very long RPATH exceeds linker limits | Link failure | Very Low | Rare in practice; emit error if RPATH > 4KB |

### 7.2 Mitigation Details

**R1: Network Mount Failures**
```rust
let canonical = match path.canonicalize() {
    Ok(p) => p,
    Err(e) => {
        println!(
            "cargo:warning=xtask: Failed to canonicalize path {}: {}. Skipping.",
            path.display(),
            e
        );
        continue; // Skip this path, continue with others
    }
};
```

**R4: Conflicting Variables**
```rust
if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    if env::var("CROSSVAL_RPATH_BITNET").is_ok() || env::var("CROSSVAL_RPATH_LLAMA").is_ok() {
        println!(
            "cargo:warning=xtask: Both BITNET_CROSSVAL_LIBDIR and CROSSVAL_RPATH_* set. \
             Using BITNET_CROSSVAL_LIBDIR (takes precedence)."
        );
    }
    // Use BITNET_CROSSVAL_LIBDIR only
}
```

**R5: RPATH Length Limits**
```rust
const MAX_RPATH_LENGTH: usize = 4096; // Conservative limit

fn merge_and_deduplicate(paths: Vec<PathBuf>) -> String {
    let merged = /* ... merging logic ... */;

    if merged.len() > MAX_RPATH_LENGTH {
        panic!(
            "Merged RPATH exceeds maximum length ({} > {}). \
             Please use BITNET_CROSSVAL_LIBDIR to specify a single directory.",
            merged.len(),
            MAX_RPATH_LENGTH
        );
    }

    merged
}
```

---

## 8. Documentation Updates

### 8.1 Files to Update

| File | Section | Change |
|------|---------|--------|
| `CLAUDE.md` | Environment Variables | Add `CROSSVAL_RPATH_BITNET`, `CROSSVAL_RPATH_LLAMA` |
| `docs/environment-variables.md` | Cross-Validation Configuration | Document new variables with examples |
| `docs/howto/cpp-setup.md` | Advanced Setup | Add section on separate library installations |
| `docs/explanation/dual-backend-crossval.md` | Build System | Update RPATH merging architecture |

### 8.2 Example Documentation Snippet

**For `docs/environment-variables.md`**:

```markdown
### Cross-Validation Configuration

#### `CROSSVAL_RPATH_BITNET` (Optional)

Explicit path to BitNet.cpp library directory for RPATH embedding.

**Example:**
```bash
export CROSSVAL_RPATH_BITNET=/opt/bitnet/lib
cargo build -p xtask --features crossval-all
```

**Use Case:** When BitNet.cpp is installed separately from llama.cpp (e.g., system package manager).

#### `CROSSVAL_RPATH_LLAMA` (Optional)

Explicit path to llama.cpp library directory for RPATH embedding.

**Example:**
```bash
export CROSSVAL_RPATH_LLAMA=/usr/local/lib
cargo build -p xtask --features crossval-all
```

**Use Case:** When llama.cpp is installed via system package manager or custom location.

#### Combined Usage (Separate Libraries)

```bash
# BitNet.cpp in /opt/bitnet, llama.cpp in /usr/local
export CROSSVAL_RPATH_BITNET=/opt/bitnet/lib
export CROSSVAL_RPATH_LLAMA=/usr/local/lib
cargo build -p xtask --features crossval-all

# Verify merged RPATH
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [/opt/bitnet/lib:/usr/local/lib]
```

#### Priority Order

1. **`BITNET_CROSSVAL_LIBDIR`** (legacy): Single directory override, skips merging
2. **`CROSSVAL_RPATH_BITNET` + `CROSSVAL_RPATH_LLAMA`**: Merged with deduplication
3. **`BITNET_CPP_DIR/build/bin`**: Auto-discovery fallback

**Backward Compatibility:** Existing `BITNET_CROSSVAL_LIBDIR` usage is unaffected.
```

---

## 9. Future Enhancements

### 9.1 Potential Improvements (Out of Scope for v1.0.0)

1. **Dynamic Library Discovery**: Auto-detect libraries in standard system paths (`/usr/lib`, `/usr/local/lib`, etc.)
   - Complexity: Would require parsing `ld.so.cache` or `ldconfig` output
   - Benefit: Reduce manual configuration for system-installed libraries

2. **RPATH Compression**: Eliminate common path prefixes to reduce RPATH length
   - Example: `/opt/libs/bitnet:/opt/libs/llama` → `/opt/libs` (if all libs in subdirs)
   - Complexity: Requires library presence validation

3. **Windows Support**: Automatic `PATH` manipulation via xtask wrapper script
   - Approach: Generate `.bat` or `.ps1` wrapper that sets PATH before invoking xtask
   - Limitation: Requires user to source wrapper instead of direct `cargo run`

4. **CI/CD Integration**: Detect CI environment and auto-adjust paths
   - Example: GitHub Actions could use `${{ github.workspace }}/libs`
   - Benefit: Zero-config CI setup

### 9.2 Known Limitations

1. **Absolute Paths Only**: RPATH requires absolute paths, not portable across machines
   - Workaround: Use relative paths in CI, convert to absolute during build
   - Future: Support `$ORIGIN`-relative RPATH (relative to binary location)

2. **No Runtime Relocation**: RPATH hardcoded at build time, not runtime
   - Limitation: Moving libraries after build requires rebuild or `LD_LIBRARY_PATH` override
   - Future: Explore dynamic RPATH patching tools (e.g., `patchelf` on Linux)

3. **Single RPATH Entry**: Linker only accepts one `-Wl,-rpath` argument
   - Current: Merge all paths with `:` separator (correct approach)
   - Limitation: Some linkers may have maximum RPATH length limits (mitigated by 4KB check)

---

## 10. Approval and Sign-Off

**Specification Author:** BitNet.rs Spec Generator (Subagent)
**Reviewers:** (To be assigned)
**Status:** Draft - Awaiting Review
**Next Steps:**
1. Review by maintainers
2. Unit test implementation
3. Integration test validation
4. Documentation updates
5. Merge to main branch

---

## Appendix A: Reference Implementation (Complete Code)

**File: `xtask/build.rs` (Full Replacement)**

```rust
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITNET_CROSSVAL_LIBDIR");
    println!("cargo:rerun-if-env-changed=BITNET_CPP_DIR");
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_BITNET");
    println!("cargo:rerun-if-env-changed=CROSSVAL_RPATH_LLAMA");

    #[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
    embed_crossval_rpath();
}

#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
fn embed_crossval_rpath() {
    use std::{
        collections::HashSet,
        env,
        path::{Path, PathBuf},
    };

    // Priority 1: Legacy single-directory override (backward compatibility)
    if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
        let lib_path = Path::new(&lib_dir);
        if lib_path.exists() {
            // Emit warning if user also set new variables
            if env::var("CROSSVAL_RPATH_BITNET").is_ok() || env::var("CROSSVAL_RPATH_LLAMA").is_ok() {
                println!(
                    "cargo:warning=xtask: Both BITNET_CROSSVAL_LIBDIR and CROSSVAL_RPATH_* set. \
                     Using BITNET_CROSSVAL_LIBDIR (takes precedence)."
                );
            }
            emit_rpath(lib_path.display().to_string());
            return;
        } else {
            println!(
                "cargo:warning=xtask: BITNET_CROSSVAL_LIBDIR set but directory does not exist: {}",
                lib_dir
            );
        }
    }

    // Priority 2: Read granular environment variables and merge
    let mut rpath_candidates: Vec<PathBuf> = Vec::new();

    if let Ok(bitnet_path) = env::var("CROSSVAL_RPATH_BITNET") {
        let path = Path::new(&bitnet_path);
        if path.exists() {
            rpath_candidates.push(path.to_path_buf());
        } else {
            println!(
                "cargo:warning=xtask: CROSSVAL_RPATH_BITNET set but directory does not exist: {}",
                bitnet_path
            );
        }
    }

    if let Ok(llama_path) = env::var("CROSSVAL_RPATH_LLAMA") {
        let path = Path::new(&llama_path);
        if path.exists() {
            rpath_candidates.push(path.to_path_buf());
        } else {
            println!(
                "cargo:warning=xtask: CROSSVAL_RPATH_LLAMA set but directory does not exist: {}",
                llama_path
            );
        }
    }

    // Merge and deduplicate if we have candidates
    if !rpath_candidates.is_empty() {
        let merged = merge_and_deduplicate(rpath_candidates);
        emit_rpath(merged);
        return;
    }

    // Priority 3: Fallback to BITNET_CPP_DIR auto-discovery (existing logic)
    if let Ok(cpp_dir) = env::var("BITNET_CPP_DIR") {
        let build_bin = Path::new(&cpp_dir).join("build/bin");
        if build_bin.exists() {
            emit_rpath(build_bin.display().to_string());
            return;
        }

        let build_dir = Path::new(&cpp_dir).join("build");
        if build_dir.exists() {
            emit_rpath(build_dir.display().to_string());
        }
    }

    // No library directory found - graceful fallback (STUB mode)
}

#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
fn merge_and_deduplicate(paths: Vec<PathBuf>) -> String {
    use std::collections::HashSet;

    const MAX_RPATH_LENGTH: usize = 4096; // Conservative limit for linker

    let mut seen = HashSet::new();
    let mut merged = Vec::new();

    for path in paths {
        // Canonicalize to resolve symlinks and normalize case (macOS)
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "cargo:warning=xtask: Failed to canonicalize path {}: {}. Skipping.",
                    path.display(),
                    e
                );
                continue; // Skip invalid paths
            }
        };

        // Deduplicate using canonical path
        if seen.insert(canonical.clone()) {
            merged.push(canonical);
        }
    }

    // Join with colon separator (POSIX RPATH syntax)
    let result = merged.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(":");

    // Sanity check: RPATH length limit
    if result.len() > MAX_RPATH_LENGTH {
        panic!(
            "Merged RPATH exceeds maximum length ({} > {}). \
             Please use BITNET_CROSSVAL_LIBDIR to specify a single directory, \
             or reduce the number of library paths.",
            result.len(),
            MAX_RPATH_LENGTH
        );
    }

    result
}

#[cfg(any(feature = "crossval", feature = "crossval-all", feature = "ffi"))]
fn emit_rpath(rpath: String) {
    // Emit link search directive (compile-time)
    // Note: rustc-link-search does NOT support colon-separated paths,
    // so we emit the first path only for link-time resolution.
    let first_path = rpath.split(':').next().unwrap_or(&rpath);
    println!("cargo:rustc-link-search=native={}", first_path);

    // Emit RPATH for runtime library resolution
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
        println!("cargo:warning=xtask: Embedded merged RPATH for runtime loader: {}", rpath);
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
        println!("cargo:warning=xtask: Embedded merged RPATH for runtime loader: {}", rpath);
    }

    #[cfg(target_os = "windows")]
    {
        println!(
            "cargo:warning=xtask: Windows detected - RPATH not applicable. \
             Ensure libraries are in PATH or use setup-cpp-auto. \
             Merged paths: {}",
            rpath
        );
    }
}
```

---

## Appendix B: Example Scenarios

### Scenario 1: Developer with System-Installed llama.cpp

**Setup:**
```bash
# Install llama.cpp via package manager
sudo apt install llama.cpp  # Hypothetical package

# Build BitNet.cpp from source
git clone https://github.com/microsoft/BitNet.cpp ~/bitnet
cmake -S ~/bitnet -B ~/bitnet/build
cmake --build ~/bitnet/build

# Configure BitNet.rs
export CROSSVAL_RPATH_BITNET=~/bitnet/build/bin
export CROSSVAL_RPATH_LLAMA=/usr/lib/x86_64-linux-gnu  # System location

cargo build -p xtask --features crossval-all
```

**Verification:**
```bash
readelf -d target/debug/xtask | grep RPATH
# Expected: Library rpath: [/home/user/bitnet/build/bin:/usr/lib/x86_64-linux-gnu]

ldd target/debug/xtask | grep -E 'libbitnet|libllama'
# Expected:
#   libbitnet.so => /home/user/bitnet/build/bin/libbitnet.so
#   libllama.so => /usr/lib/x86_64-linux-gnu/libllama.so
```

### Scenario 2: CI/CD with Separate Build Artifacts

**CI Configuration (.github/workflows/crossval.yml):**
```yaml
- name: Build BitNet.cpp
  run: |
    cmake -S bitnet.cpp -B bitnet.cpp/build
    cmake --build bitnet.cpp/build
    echo "CROSSVAL_RPATH_BITNET=${{ github.workspace }}/bitnet.cpp/build/bin" >> $GITHUB_ENV

- name: Build llama.cpp
  run: |
    cmake -S llama.cpp -B llama.cpp/build
    cmake --build llama.cpp/build
    echo "CROSSVAL_RPATH_LLAMA=${{ github.workspace }}/llama.cpp/build/bin" >> $GITHUB_ENV

- name: Build xtask
  run: cargo build -p xtask --features crossval-all

- name: Verify RPATH
  run: |
    readelf -d target/debug/xtask | grep RPATH
    ldd target/debug/xtask | grep -E 'libbitnet|libllama'

- name: Run cross-validation
  run: cargo run -p xtask -- crossval-per-token --model test.gguf --prompt "test"
```

### Scenario 3: Docker Multi-Stage Build

**Dockerfile:**
```dockerfile
FROM rust:1.90 AS builder

# Stage 1: Build C++ backends
RUN git clone https://github.com/microsoft/BitNet.cpp /bitnet && \
    cmake -S /bitnet -B /bitnet/build && \
    cmake --build /bitnet/build

RUN git clone https://github.com/ggerganov/llama.cpp /llama && \
    cmake -S /llama -B /llama/build && \
    cmake --build /llama/build

# Stage 2: Build BitNet.rs with merged RPATH
ENV CROSSVAL_RPATH_BITNET=/bitnet/build/bin
ENV CROSSVAL_RPATH_LLAMA=/llama/build/bin

COPY . /bitnet-rs
WORKDIR /bitnet-rs
RUN cargo build -p xtask --features crossval-all --release

# Stage 3: Runtime image
FROM debian:bookworm-slim
COPY --from=builder /bitnet/build/bin/libbitnet.so /usr/local/lib/
COPY --from=builder /llama/build/bin/libllama.so /usr/local/lib/
COPY --from=builder /bitnet-rs/target/release/xtask /usr/local/bin/

# RPATH embedded at build time, no LD_LIBRARY_PATH needed
RUN ldconfig
CMD ["/usr/local/bin/xtask", "preflight"]
```

---

**END OF SPECIFICATION**
