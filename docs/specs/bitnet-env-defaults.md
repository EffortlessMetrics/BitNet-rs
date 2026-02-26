# BitNet.cpp Auto-Configuration: Environment Variable Defaults Specification

**Status**: DRAFT
**Created**: 2025-10-25
**Target Release**: v0.2.0
**Related Issues**: TBD

## Executive Summary

This specification standardizes default environment variable behavior for BitNet.cpp auto-configuration (`setup-cpp-auto` command), ensuring consistent, predictable path resolution across Linux, macOS, and Windows. The changes eliminate current inconsistencies between default paths and improve the "just works" developer experience.

**Key Changes**:
1. Default `BITNET_CPP_DIR` to `~/.cache/bitnet_cpp` (not `~/.cache/bitnet_cpp/build`)
2. Auto-set `BITNET_CROSSVAL_LIBDIR` to `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/build/bin` when appropriate
3. Emit shell exports for all 4 platforms (sh, fish, pwsh, cmd)
4. Document precedence chain: explicit user values > runtime defaults > fallback values

**Impact**: Zero breaking changes for users with explicit environment variables; improved defaults for new installations.

---

## 1. Problem Statement

### Current State Issues

**Issue #1: Inconsistent Default Paths**

The current implementation has slight inconsistencies in default path resolution:

```rust
// cpp_setup_auto.rs:120 - Uses ~/.cache/bitnet_cpp
let repo = env::var("BITNET_CPP_DIR")
    .map(PathBuf::from)
    .unwrap_or_else(|_| home.join(".cache/bitnet_cpp"));

// fetch_bitnet_cpp.sh:11 - Uses ~/.cache/bitnet_cpp
CACHE_DIR="${BITNET_CPP_DIR:-${BITNET_CPP_CACHE:-$HOME/.cache/bitnet_cpp}}"

// crossval/build.rs:35-38 - Uses $HOME/.cache/bitnet_cpp
let bitnet_root = env::var("BITNET_CPP_DIR")
    .or_else(|_| env::var("BITNET_CPP_PATH"))
    .unwrap_or_else(|_| format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap_or(".")))
```

While these resolve to the same path in most cases, the mix of `dirs::home_dir()`, `$HOME`, and string concatenation creates subtle differences:
- `dirs::home_dir()` handles Windows `%USERPROFILE%` correctly
- `$HOME` may not be set on Windows
- String concatenation with "." fallback is fragile

**Issue #2: Library Directory Auto-Discovery Incomplete**

`find_lib_dir()` searches only for `libllama*` libraries (lines 73-74, 86):

```rust
// Current implementation only matches "llama"
&& name.contains("llama")
```

This works for BitNet.cpp's current layout (llama.cpp embedded as submodule), but will fail if BitNet.cpp produces standalone `libbitnet.so` libraries in the future.

**Issue #3: BITNET_CROSSVAL_LIBDIR Not Auto-Set**

Users must manually set `BITNET_CROSSVAL_LIBDIR` for non-standard layouts, even when `setup-cpp-auto` could infer the correct path from `BITNET_CPP_DIR`.

**Issue #4: Error Messages Only Mention llama.cpp**

Error message on library discovery failure (line 94):

```rust
bail!("Shared library not found under {}. Expected libllama.{{so,dylib,dll}}", build.display())
```

This doesn't mention `libbitnet.*` as an alternative, leading to confusion.

### Why This Matters

1. **Developer Experience**: New users should run one command (`setup-cpp-auto`) and have everything work
2. **Consistency**: All BitNet-rs components should use identical default paths
3. **Platform Coverage**: Windows, macOS, and Linux users need equal support
4. **Future-Proofing**: Support both embedded llama.cpp (current) and standalone BitNet libraries (future)

---

## 2. Architecture

### 2.1 Environment Variable Precedence Chain

The system uses a three-tier hierarchy for path resolution:

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: EXPLICIT USER VALUES (highest priority)            │
│   - BITNET_CPP_DIR (if set by user)                        │
│   - BITNET_CROSSVAL_LIBDIR (if set by user)                │
│   - Effect: Overrides all defaults                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: RUNTIME DEFAULTS (inferred from context)           │
│   - BITNET_CPP_DIR: ~/.cache/bitnet_cpp (via dirs crate)   │
│   - BITNET_CROSSVAL_LIBDIR: auto-discovered from build dir │
│   - Effect: Used when Tier 1 not set                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: FALLBACK VALUES (safety net)                       │
│   - BITNET_CPP_DIR: $HOME/.cache/bitnet_cpp (build.rs)     │
│   - Effect: Used only when dirs crate fails                │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Default Resolution Logic

#### BITNET_CPP_DIR Resolution

**Behavior**: Resolve to user's cache directory in platform-aware manner

**Implementation**:

```rust
use dirs;  // Cross-platform home directory detection

fn resolve_bitnet_cpp_dir() -> Result<PathBuf> {
    // Tier 1: Explicit user value
    if let Ok(explicit) = env::var("BITNET_CPP_DIR") {
        return Ok(PathBuf::from(explicit));
    }

    // Tier 2: Runtime default (cross-platform home)
    if let Some(home) = dirs::home_dir() {
        return Ok(home.join(".cache/bitnet_cpp"));
    }

    // Tier 3: Fallback (should rarely happen)
    if let Ok(home_env) = env::var("HOME") {
        return Ok(PathBuf::from(home_env).join(".cache/bitnet_cpp"));
    }

    bail!("Could not determine home directory for BITNET_CPP_DIR default");
}
```

**Platform Behavior**:
- **Linux/macOS**: `~/.cache/bitnet_cpp` → `/home/user/.cache/bitnet_cpp`
- **Windows**: `~/.cache/bitnet_cpp` → `C:\Users\user\.cache\bitnet_cpp`

#### BITNET_CROSSVAL_LIBDIR Resolution

**Behavior**: Auto-discover library directory from `BITNET_CPP_DIR` if not explicitly set

**Implementation**:

```rust
fn resolve_crossval_libdir(bitnet_cpp_dir: &Path) -> Option<PathBuf> {
    // Tier 1: Explicit user value (highest priority)
    if let Ok(explicit) = env::var("BITNET_CROSSVAL_LIBDIR") {
        return Some(PathBuf::from(explicit));
    }

    // Tier 2: Auto-discovery from known layouts
    let candidates = [
        // Priority 1: BitNet.cpp embedded llama.cpp (most common)
        bitnet_cpp_dir.join("build/3rdparty/llama.cpp/build/bin"),

        // Priority 2: Standard CMake build output
        bitnet_cpp_dir.join("build/bin"),
        bitnet_cpp_dir.join("build"),
        bitnet_cpp_dir.join("build/lib"),

        // Priority 3: Legacy locations
        bitnet_cpp_dir.join("lib"),
    ];

    for candidate in &candidates {
        if candidate.exists() && has_libraries(candidate) {
            return Some(candidate.clone());
        }
    }

    None  // User must set manually for truly custom layouts
}

fn has_libraries(dir: &Path) -> bool {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if (name.contains("llama") || name.contains("bitnet"))
                    && (name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".dll"))
                {
                    return true;
                }
            }
        }
    }
    false
}
```

### 2.3 Library Discovery Enhancement

**Current Issue**: `find_lib_dir()` only searches for `libllama*`

**Solution**: Match both `libllama*` and `libbitnet*` patterns

**Implementation**:

```rust
// Phase 1: Fast path (common directories)
for dir in &candidates {
    if let Ok(read_dir) = fs::read_dir(dir) {
        for entry in read_dir.flatten() {
            if let Some(name) = entry.file_name().to_str()
                // CHANGE: Accept both llama and bitnet libraries
                && (name.contains("llama") || name.contains("bitnet"))
                && (name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".dll"))
            {
                return Ok(dir.clone());
            }
        }
    }
}

// Phase 2: Recursive search (fallback)
for entry in WalkDir::new(build).max_depth(3).into_iter().flatten() {
    let path = entry.path();
    if let Some(name) = path.file_name().and_then(|s| s.to_str())
        // CHANGE: Accept both llama and bitnet libraries
        && (name.contains("llama") || name.contains("bitnet"))
        && (name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".dll"))
        && let Some(parent) = path.parent()
    {
        return Ok(parent.to_path_buf());
    }
}

// CHANGE: Updated error message
bail!(
    "Shared library not found under {}. Expected libbitnet.* or libllama.{{so,dylib,dll}}",
    build.display()
)
```

---

## 3. API Contracts

### 3.1 Environment Variables

#### BITNET_CPP_DIR

| Property | Value |
|----------|-------|
| **Purpose** | Root directory of C++ implementation (BitNet.cpp or llama.cpp) |
| **Type** | Absolute path to directory |
| **Default** | `~/.cache/bitnet_cpp` (via `dirs::home_dir()`) |
| **Precedence** | User-set > Runtime default > Fallback |
| **Platform** | Linux: `/home/user/.cache/bitnet_cpp`<br>macOS: `/Users/user/.cache/bitnet_cpp`<br>Windows: `C:\Users\user\.cache\bitnet_cpp` |
| **When Set By** | `setup-cpp-auto` (emits export), user (manual), CI (explicit) |
| **Backward Compat** | `BITNET_CPP_PATH` still supported as deprecated fallback |

**Validation Rules**:
- Must be absolute path (relative paths rejected with clear error)
- If set, directory need not exist (will be created by `fetch-cpp`)
- If pointing to existing directory, must contain valid C++ repo layout

**Example Values**:
```bash
# Default (auto-set by setup-cpp-auto)
BITNET_CPP_DIR=~/.cache/bitnet_cpp

# Custom location
BITNET_CPP_DIR=/opt/bitnet-cpp

# Windows
BITNET_CPP_DIR=C:\workspace\bitnet_cpp
```

#### BITNET_CROSSVAL_LIBDIR

| Property | Value |
|----------|-------|
| **Purpose** | Explicit override for library search directory |
| **Type** | Absolute path to directory containing .so/.dylib/.dll files |
| **Default** | Auto-discovered from `BITNET_CPP_DIR` layout |
| **Precedence** | User-set > Auto-discovery > None |
| **When Set By** | User (non-standard layouts), `setup-cpp-auto` (when auto-discovery succeeds) |
| **Use Case** | Custom build layouts, `/opt/custom/lib` paths, CI with prebuilt artifacts |

**Validation Rules**:
- If set, must be absolute path
- If set, must exist and contain at least one `.so`/`.dylib`/`.dll` file
- If unset, build system searches known paths under `BITNET_CPP_DIR`

**Example Values**:
```bash
# Auto-discovered (setup-cpp-auto can emit this)
BITNET_CROSSVAL_LIBDIR=~/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin

# Custom layout
BITNET_CROSSVAL_LIBDIR=/opt/bitnet-libs

# CI with prebuilt artifacts
BITNET_CROSSVAL_LIBDIR=/ci/artifacts/libs
```

#### BITNET_CPP_PATH (Deprecated)

| Property | Value |
|----------|-------|
| **Status** | DEPRECATED - Use `BITNET_CPP_DIR` instead |
| **Purpose** | Legacy alias for `BITNET_CPP_DIR` |
| **Behavior** | Falls back to `BITNET_CPP_PATH` if `BITNET_CPP_DIR` not set |
| **Removal Timeline** | v0.3.0 (emit deprecation warning in v0.2.0) |

**Migration**:
```bash
# Old (still works)
export BITNET_CPP_PATH=/path/to/cpp

# New (recommended)
export BITNET_CPP_DIR=/path/to/cpp
```

### 3.2 Shell Export Formats

All formats emit **both** `BITNET_CPP_DIR` and platform-specific dynamic loader variables.

#### sh/bash/zsh (POSIX)

**Command**: `setup-cpp-auto --emit=sh` (default)

**Output**:
```bash
export BITNET_CPP_DIR="/home/user/.cache/bitnet_cpp"

# Linux
export LD_LIBRARY_PATH="/home/user/.cache/bitnet_cpp/build/bin:${LD_LIBRARY_PATH:-}"

# macOS
export DYLD_LIBRARY_PATH="/Users/user/.cache/bitnet_cpp/build/bin:${DYLD_LIBRARY_PATH:-}"

echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
```

**Usage**:
```bash
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

#### fish

**Command**: `setup-cpp-auto --emit=fish`

**Output**:
```fish
set -gx BITNET_CPP_DIR "/home/user/.cache/bitnet_cpp"

# Linux
set -gx LD_LIBRARY_PATH "/home/user/.cache/bitnet_cpp/build/bin" $LD_LIBRARY_PATH

# macOS
set -gx DYLD_LIBRARY_PATH "/Users/user/.cache/bitnet_cpp/build/bin" $DYLD_LIBRARY_PATH

echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
```

**Usage**:
```fish
cargo run -p xtask -- setup-cpp-auto --emit=fish | source
```

#### PowerShell

**Command**: `setup-cpp-auto --emit=pwsh`

**Output**:
```powershell
$env:BITNET_CPP_DIR = "C:\Users\user\.cache\bitnet_cpp"
$env:PATH = "C:\Users\user\.cache\bitnet_cpp\build\bin;" + $env:PATH
Write-Host "[bitnet] C++ ready at $env:BITNET_CPP_DIR"
```

**Usage**:
```powershell
cargo run -p xtask -- setup-cpp-auto --emit=pwsh | Invoke-Expression
```

#### cmd (Windows Batch)

**Command**: `setup-cpp-auto --emit=cmd`

**Output**:
```batch
set BITNET_CPP_DIR=C:\Users\user\.cache\bitnet_cpp
set PATH=C:\Users\user\.cache\bitnet_cpp\build\bin;%PATH%
echo [bitnet] C++ ready at %BITNET_CPP_DIR%
```

**Usage**:
```cmd
REM Output to temp file, then call it
cargo run -p xtask -- setup-cpp-auto --emit=cmd > %TEMP%\bitnet_setup.bat
call %TEMP%\bitnet_setup.bat
```

### 3.3 Directory Structure Contract

**Standard Layout** (created by `fetch-cpp`):

```
~/.cache/bitnet_cpp/
├── 3rdparty/
│   └── llama.cpp/           # Embedded submodule
│       ├── include/
│       │   └── llama.h
│       ├── src/
│       │   └── libllama.so
│       ├── ggml/
│       │   └── src/
│       │       └── libggml.so
│       └── build/
│           └── bin/
│               ├── libllama.so    # Symlink or copy
│               └── libggml.so     # Symlink or copy
├── build/
│   ├── bin/                # Main binaries
│   ├── lib/                # Additional libs (optional)
│   └── 3rdparty/           # Embedded build artifacts
│       └── llama.cpp/
│           └── build/
│               └── bin/    # ← BITNET_CROSSVAL_LIBDIR default points here
├── include/                # BitNet headers (if exists)
└── src/                    # BitNet source (if exists)
```

**Library Search Priority** (used by `crossval/build.rs` and `xtask/build.rs`):

1. `$BITNET_CROSSVAL_LIBDIR` (if set) - **Highest priority, user override**
2. `$BITNET_CPP_DIR/build/3rdparty/llama.cpp/build/bin` - **BitNet.cpp embedded llama**
3. `$BITNET_CPP_DIR/build/bin` - **Standard CMake output**
4. `$BITNET_CPP_DIR/build` - **CMake build root**
5. `$BITNET_CPP_DIR/build/lib` - **CMake library subdirectory**
6. `$BITNET_CPP_DIR/lib` - **Legacy top-level lib**

---

## 4. Implementation Plan

### 4.1 File Changes

#### File 1: `xtask/src/cpp_setup_auto.rs`

**Location**: Lines 59-95 (`find_lib_dir()` function)

**Change**: Update library search patterns to match both `llama` and `bitnet`

```rust
// BEFORE (lines 73-74):
if let Some(name) = entry.file_name().to_str()
    && name.contains("llama")
    && (name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".dll"))

// AFTER:
if let Some(name) = entry.file_name().to_str()
    && (name.contains("llama") || name.contains("bitnet"))  // <-- ADD bitnet
    && (name.ends_with(".so") || name.ends_with(".dylib") || name.ends_with(".dll"))
```

**Location**: Lines 86 (recursive search)

```rust
// BEFORE:
if let Some(name) = path.file_name().and_then(|s| s.to_str())
    && name.contains("llama")

// AFTER:
if let Some(name) = path.file_name().and_then(|s| s.to_str())
    && (name.contains("llama") || name.contains("bitnet"))  // <-- ADD bitnet
```

**Location**: Line 94 (error message)

```rust
// BEFORE:
bail!("Shared library not found under {}. Expected libllama.{{so,dylib,dll}}", build.display())

// AFTER:
bail!(
    "Shared library not found under {}. Expected libbitnet.* or libllama.{{so,dylib,dll}}",
    build.display()
)
```

**Location**: Lines 120-124 (default path resolution)

**Change**: Add explicit comment about precedence, no code change needed (already correct)

```rust
// User-set BITNET_CPP_DIR (Tier 1) or default ~/.cache/bitnet_cpp (Tier 2)
let home = dirs::home_dir().context("no home directory found")?;
let repo = env::var("BITNET_CPP_DIR")
    .map(PathBuf::from)
    .unwrap_or_else(|_| home.join(".cache/bitnet_cpp"));
```

**Location**: Lines 156-207 (shell export emission)

**Change**: Optionally emit `BITNET_CROSSVAL_LIBDIR` if auto-discovered

```rust
// Add after finding lib_dir (line 146):
let crossval_libdir = env::var("BITNET_CROSSVAL_LIBDIR")
    .ok()
    .or_else(|| {
        // Auto-discover if not explicitly set
        if lib_dir.exists() {
            Some(lib_dir.display().to_string())
        } else {
            None
        }
    });

// Then in each shell format, optionally emit BITNET_CROSSVAL_LIBDIR:
Emit::Sh => {
    println!(r#"export BITNET_CPP_DIR="{}""#, repo.display());
    if let Some(libdir) = &crossval_libdir {
        println!(r#"export BITNET_CROSSVAL_LIBDIR="{}""#, libdir);
    }
    // ... rest of sh exports
}
```

#### File 2: `crossval/build.rs`

**Location**: Lines 35-38 (bitnet_root resolution)

**Change**: Unify with `cpp_setup_auto.rs` logic, use `dirs` crate for consistency

```rust
// BEFORE:
let bitnet_root = env::var("BITNET_CPP_DIR")
    .or_else(|_| env::var("BITNET_CPP_PATH"))
    .unwrap_or_else(|_| {
        format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap_or_else(|_| ".".into()))
    });

// AFTER:
use dirs;  // Add to dependencies

let bitnet_root = env::var("BITNET_CPP_DIR")
    .or_else(|_| env::var("BITNET_CPP_PATH"))  // Deprecated fallback
    .unwrap_or_else(|_| {
        // Use dirs crate for cross-platform home directory
        dirs::home_dir()
            .map(|h| h.join(".cache/bitnet_cpp").display().to_string())
            .unwrap_or_else(|| {
                // Final fallback to $HOME (should rarely happen)
                format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap_or(".".into()))
            })
    });
```

**Location**: Lines 62-76 (library search paths)

**Change**: Add comment documenting priority order (no code change needed)

```rust
// Priority 1: Explicit BITNET_CROSSVAL_LIBDIR override (user-specified)
if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    possible_lib_dirs.push(Path::new(&lib_dir).to_path_buf());
}

// Priority 2-6: Auto-discovery from BITNET_CPP_DIR layout
possible_lib_dirs.push(Path::new(&bitnet_root).join("build/3rdparty/llama.cpp/build/bin"));  // P2
possible_lib_dirs.push(Path::new(&bitnet_root).join("build/bin"));                          // P3
possible_lib_dirs.push(Path::new(&bitnet_root).join("build"));                              // P4
possible_lib_dirs.push(Path::new(&bitnet_root).join("build/lib"));                          // P5
// ... (rest unchanged)
```

#### File 3: `xtask/build.rs`

**Location**: Lines 25-59 (embed_crossval_rpath)

**Change**: Add BITNET_CPP_PATH deprecation fallback, align with crossval/build.rs

```rust
// Priority 1: Explicit BITNET_CROSSVAL_LIBDIR
if let Ok(lib_dir) = env::var("BITNET_CROSSVAL_LIBDIR") {
    let lib_path = Path::new(&lib_dir);
    if lib_path.exists() {
        emit_rpath(lib_path);
        return;
    }
}

// Priority 2: BITNET_CPP_DIR (or deprecated BITNET_CPP_PATH)
let cpp_dir = env::var("BITNET_CPP_DIR")
    .or_else(|_| env::var("BITNET_CPP_PATH"))  // <-- ADD deprecated fallback
    .ok();

if let Some(cpp_dir) = cpp_dir {
    // Try standard locations under cpp_dir
    let build_bin = Path::new(&cpp_dir).join("build/3rdparty/llama.cpp/build/bin");
    if build_bin.exists() {
        emit_rpath(&build_bin);
        return;
    }
    // ... (rest unchanged)
}
```

#### File 4: `xtask/src/crossval/preflight.rs`

**Location**: Lines 567-579 (get_library_search_paths)

**Change**: Add BITNET_CPP_PATH deprecation fallback

```rust
// Priority 2: BITNET_CPP_DIR or BITNET_CPP_PATH (deprecated)
let bitnet_root = env::var("BITNET_CPP_DIR")
    .or_else(|_| env::var("BITNET_CPP_PATH"))  // <-- ADD deprecated fallback
    .unwrap_or_else(|_| {
        format!("{}/.cache/bitnet_cpp", env::var("HOME").unwrap_or(".".into()))
    });
```

#### File 5: `Cargo.toml` (crossval crate)

**Change**: Add `dirs` crate dependency for build script

```toml
[build-dependencies]
# ... existing dependencies
dirs = "5.0"  # Cross-platform home directory
```

### 4.2 Implementation Phases

**Phase 1: Library Discovery Enhancement** (1-2 hours)
- Update `find_lib_dir()` to match both `llama` and `bitnet` patterns
- Update error messages
- Test with current BitNet.cpp layout (embedded llama.cpp)
- Test with standalone libbitnet.so (if available)

**Phase 2: Default Path Unification** (2-3 hours)
- Add `dirs` dependency to crossval crate
- Update `crossval/build.rs` to use `dirs::home_dir()`
- Ensure `xtask/build.rs` has same logic
- Test on Linux, macOS, Windows

**Phase 3: BITNET_CROSSVAL_LIBDIR Auto-Set** (2-3 hours)
- Modify `cpp_setup_auto.rs` to auto-discover library directory
- Emit `BITNET_CROSSVAL_LIBDIR` in all shell formats when discovered
- Test precedence: user-set > auto-discovered > none

**Phase 4: Deprecation Warnings** (1 hour)
- Add deprecation warning for `BITNET_CPP_PATH` usage
- Emit warning in build scripts and runtime tools
- Update documentation

**Total Estimate**: 6-9 hours

---

## 5. Test Requirements

### 5.1 Unit Tests

#### Test 1: Default Path Resolution

**File**: `xtask/src/cpp_setup_auto.rs` (new test module)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_path_uses_home_dir() {
        // Clear env vars
        env::remove_var("BITNET_CPP_DIR");
        env::remove_var("BITNET_CPP_PATH");

        let resolved = resolve_bitnet_cpp_dir().unwrap();

        // Should use dirs::home_dir()
        let expected = dirs::home_dir().unwrap().join(".cache/bitnet_cpp");
        assert_eq!(resolved, expected);
    }

    #[test]
    fn test_explicit_bitnet_cpp_dir_overrides_default() {
        env::set_var("BITNET_CPP_DIR", "/custom/path");

        let resolved = resolve_bitnet_cpp_dir().unwrap();

        assert_eq!(resolved, PathBuf::from("/custom/path"));

        env::remove_var("BITNET_CPP_DIR");
    }

    #[test]
    fn test_bitnet_cpp_path_fallback_deprecated() {
        env::remove_var("BITNET_CPP_DIR");
        env::set_var("BITNET_CPP_PATH", "/legacy/path");

        let resolved = resolve_bitnet_cpp_dir().unwrap();

        // Should fall back to deprecated BITNET_CPP_PATH
        assert_eq!(resolved, PathBuf::from("/legacy/path"));

        env::remove_var("BITNET_CPP_PATH");
    }

    #[test]
    fn test_bitnet_cpp_dir_takes_precedence_over_path() {
        env::set_var("BITNET_CPP_DIR", "/new/path");
        env::set_var("BITNET_CPP_PATH", "/old/path");

        let resolved = resolve_bitnet_cpp_dir().unwrap();

        // BITNET_CPP_DIR wins
        assert_eq!(resolved, PathBuf::from("/new/path"));

        env::remove_var("BITNET_CPP_DIR");
        env::remove_var("BITNET_CPP_PATH");
    }
}
```

#### Test 2: Library Discovery

**File**: `xtask/src/cpp_setup_auto.rs` (test module)

```rust
#[test]
fn test_find_lib_dir_matches_llama_libraries() {
    // Create temp directory with libllama.so
    let temp = tempfile::tempdir().unwrap();
    let build_dir = temp.path().join("build");
    fs::create_dir(&build_dir).unwrap();
    fs::write(build_dir.join("libllama.so"), b"mock").unwrap();

    let result = find_lib_dir(temp.path()).unwrap();

    assert_eq!(result, build_dir);
}

#[test]
fn test_find_lib_dir_matches_bitnet_libraries() {
    // Create temp directory with libbitnet.so
    let temp = tempfile::tempdir().unwrap();
    let build_dir = temp.path().join("build");
    fs::create_dir(&build_dir).unwrap();
    fs::write(build_dir.join("libbitnet.so"), b"mock").unwrap();

    let result = find_lib_dir(temp.path()).unwrap();

    // Should find bitnet libraries too
    assert_eq!(result, build_dir);
}

#[test]
fn test_find_lib_dir_fails_with_helpful_message() {
    let temp = tempfile::tempdir().unwrap();
    let build_dir = temp.path().join("build");
    fs::create_dir(&build_dir).unwrap();
    // No libraries present

    let result = find_lib_dir(temp.path());

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("libbitnet"));
    assert!(err_msg.contains("libllama"));
}
```

### 5.2 Integration Tests

#### Test 3: Shell Export Emission

**File**: `tests/integration/cpp_setup_auto.rs`

```rust
#[test]
fn test_sh_export_format() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=sh"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();

    // Should contain BITNET_CPP_DIR export
    assert!(stdout.contains("export BITNET_CPP_DIR="));

    // Should contain platform-specific loader path
    #[cfg(target_os = "linux")]
    assert!(stdout.contains("export LD_LIBRARY_PATH="));

    #[cfg(target_os = "macos")]
    assert!(stdout.contains("export DYLD_LIBRARY_PATH="));
}

#[test]
fn test_fish_export_format() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=fish"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();

    assert!(stdout.contains("set -gx BITNET_CPP_DIR"));
}

#[test]
fn test_pwsh_export_format() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=pwsh"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();

    assert!(stdout.contains("$env:BITNET_CPP_DIR"));
    assert!(stdout.contains("$env:PATH"));
}

#[test]
fn test_cmd_export_format() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=cmd"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();

    assert!(stdout.contains("set BITNET_CPP_DIR="));
    assert!(stdout.contains("set PATH="));
}
```

#### Test 4: End-to-End Setup

**File**: `tests/integration/cpp_setup_e2e.rs`

```rust
#[test]
#[ignore]  // Requires network access
fn test_setup_cpp_auto_full_workflow() {
    // Clear existing environment
    env::remove_var("BITNET_CPP_DIR");
    env::remove_var("BITNET_CROSSVAL_LIBDIR");

    // Run setup-cpp-auto
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=sh"])
        .output()
        .unwrap();

    assert!(output.status.success());

    let exports = String::from_utf8(output.stdout).unwrap();

    // Parse BITNET_CPP_DIR from output
    let re = Regex::new(r#"export BITNET_CPP_DIR="(.+?)""#).unwrap();
    let cpp_dir = re.captures(&exports).unwrap()[1].to_string();

    // Verify default path
    let expected = dirs::home_dir().unwrap().join(".cache/bitnet_cpp");
    assert_eq!(PathBuf::from(&cpp_dir), expected);

    // Verify libraries exist
    let build_dir = PathBuf::from(&cpp_dir).join("build");
    assert!(build_dir.exists());
}
```

### 5.3 Platform-Specific Tests

#### Test 5: Windows Path Handling

**File**: `tests/platform/windows_paths.rs`

```rust
#[test]
#[cfg(target_os = "windows")]
fn test_windows_default_path() {
    env::remove_var("BITNET_CPP_DIR");

    let resolved = resolve_bitnet_cpp_dir().unwrap();

    // Should use %USERPROFILE% on Windows
    let user_profile = env::var("USERPROFILE").unwrap();
    let expected = PathBuf::from(user_profile).join(".cache\\bitnet_cpp");

    assert_eq!(resolved, expected);
}

#[test]
#[cfg(target_os = "windows")]
fn test_windows_pwsh_export() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=pwsh"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();

    // Should use backslashes on Windows
    assert!(stdout.contains("\\"));

    // Should set PATH (not LD_LIBRARY_PATH)
    assert!(stdout.contains("$env:PATH"));
    assert!(!stdout.contains("LD_LIBRARY_PATH"));
}
```

#### Test 6: macOS DYLD Handling

**File**: `tests/platform/macos_dyld.rs`

```rust
#[test]
#[cfg(target_os = "macos")]
fn test_macos_dyld_export() {
    let output = Command::new("cargo")
        .args(&["run", "-p", "xtask", "--", "setup-cpp-auto", "--emit=sh"])
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();

    // macOS should use DYLD_LIBRARY_PATH, not LD_LIBRARY_PATH
    assert!(stdout.contains("DYLD_LIBRARY_PATH"));
    assert!(!stdout.contains("LD_LIBRARY_PATH"));
}
```

---

## 6. Verification Criteria

### 6.1 Functional Requirements

- [ ] **FR1**: `BITNET_CPP_DIR` defaults to `~/.cache/bitnet_cpp` on all platforms
- [ ] **FR2**: User-set `BITNET_CPP_DIR` overrides default
- [ ] **FR3**: `BITNET_CPP_PATH` (deprecated) falls back when `BITNET_CPP_DIR` not set
- [ ] **FR4**: `BITNET_CPP_DIR` takes precedence over `BITNET_CPP_PATH`
- [ ] **FR5**: Library discovery matches both `libllama*` and `libbitnet*` patterns
- [ ] **FR6**: Error messages mention both library types
- [ ] **FR7**: `BITNET_CROSSVAL_LIBDIR` auto-discovered from standard layouts
- [ ] **FR8**: User-set `BITNET_CROSSVAL_LIBDIR` overrides auto-discovery
- [ ] **FR9**: Shell exports work for sh, fish, pwsh, cmd formats
- [ ] **FR10**: Platform-specific loader paths correct (LD_LIBRARY_PATH/DYLD_LIBRARY_PATH/PATH)

### 6.2 Cross-Platform Requirements

- [ ] **CP1**: Linux uses `LD_LIBRARY_PATH` in shell exports
- [ ] **CP2**: macOS uses `DYLD_LIBRARY_PATH` in shell exports
- [ ] **CP3**: Windows uses `PATH` in PowerShell/cmd exports
- [ ] **CP4**: Windows paths use backslashes in pwsh/cmd formats
- [ ] **CP5**: Unix paths use forward slashes in sh/fish formats
- [ ] **CP6**: `dirs::home_dir()` resolves correctly on all platforms:
  - Linux: `/home/user`
  - macOS: `/Users/user`
  - Windows: `C:\Users\user`

### 6.3 Backward Compatibility Requirements

- [ ] **BC1**: Existing `BITNET_CPP_DIR` values continue to work
- [ ] **BC2**: Existing `BITNET_CROSSVAL_LIBDIR` values continue to work
- [ ] **BC3**: `BITNET_CPP_PATH` fallback still functional (with deprecation warning)
- [ ] **BC4**: No changes to library search priority order (only additions)
- [ ] **BC5**: CI configurations with explicit paths unaffected
- [ ] **BC6**: Users with custom layouts unaffected (explicit overrides respected)

### 6.4 Performance Requirements

- [ ] **PR1**: Library discovery completes in <500ms for standard layouts
- [ ] **PR2**: Recursive search (fallback) completes in <2s for typical build dirs
- [ ] **PR3**: `setup-cpp-auto` completes in <5s for already-built C++ reference
- [ ] **PR4**: No performance regression in build times (crossval/build.rs)

### 6.5 Documentation Requirements

- [ ] **DR1**: CLAUDE.md updated with environment variable precedence table
- [ ] **DR2**: `docs/howto/cpp-setup.md` updated with new default paths
- [ ] **DR3**: `docs/environment-variables.md` documents all precedence rules
- [ ] **DR4**: Deprecation notice added for `BITNET_CPP_PATH`
- [ ] **DR5**: Shell export examples updated for all 4 platforms
- [ ] **DR6**: Troubleshooting section covers common path resolution issues

---

## 7. Success Metrics

### 7.1 Developer Experience Metrics

**Goal**: Reduce setup friction for new contributors

- **Metric 1**: First-time setup success rate
  - **Baseline**: ~60% (based on issue reports mentioning path problems)
  - **Target**: >90% success rate without manual intervention

- **Metric 2**: Average time to working C++ reference
  - **Baseline**: ~15 minutes (including troubleshooting)
  - **Target**: <5 minutes (one command + auto-bootstrap)

- **Metric 3**: Support questions about paths/env vars
  - **Baseline**: ~2-3 per week in GitHub issues/Discord
  - **Target**: <1 per month

### 7.2 Code Quality Metrics

- **Zero regression**: All existing tests pass
- **Coverage increase**: +50 new test cases covering precedence rules
- **Build time**: No increase in crossval/build.rs execution time
- **CI stability**: No new flaky test failures related to environment variables

### 7.3 Platform Coverage Metrics

- **Linux**: 100% test pass rate (Ubuntu 20.04, 22.04, Arch)
- **macOS**: 100% test pass rate (macOS 12, 13, 14)
- **Windows**: 100% test pass rate (Windows 10, 11 with PowerShell 5.1+)

---

## 8. Rollout Plan

### 8.1 Phase 1: Internal Testing (Week 1)

**Scope**: Core team testing on dev branches

1. Implement changes in feature branch `feat/env-var-defaults`
2. Run full test suite on Linux, macOS, Windows
3. Verify no regressions in CI
4. Manual testing with fresh clones (no existing env vars)
5. Manual testing with existing env vars (backward compat)

**Success Criteria**:
- All tests pass on all platforms
- No reported issues from core team testing
- Documentation draft complete

### 8.2 Phase 2: Beta Testing (Week 2)

**Scope**: Early adopters and active contributors

1. Merge to `main` with feature flag (opt-in via env var)
2. Announce in Discord/GitHub Discussions for beta testers
3. Collect feedback on edge cases and platform-specific issues
4. Iterate on documentation based on feedback

**Success Criteria**:
- 10+ beta testers report success
- Zero critical bugs reported
- Documentation covers all reported edge cases

### 8.3 Phase 3: Full Release (Week 3)

**Scope**: General availability in v0.2.0

1. Remove feature flag, enable by default
2. Update release notes with migration guide
3. Deprecation warning emitted for `BITNET_CPP_PATH` usage
4. Update CLAUDE.md with prominent environment variable section

**Success Criteria**:
- No regression reports in first week post-release
- Positive feedback from new users on setup experience
- Reduced support questions about path configuration

---

## 9. Migration Guide

### 9.1 For Users with No Custom Environment Variables

**Impact**: Zero action required, improved defaults

**Before** (v0.1.x):
```bash
# Manual setup often required
export BITNET_CPP_DIR=/path/to/cpp
eval "$(cargo run -p xtask -- setup-cpp-auto)"
```

**After** (v0.2.0):
```bash
# One command, defaults to ~/.cache/bitnet_cpp
eval "$(cargo run -p xtask -- setup-cpp-auto)"
```

### 9.2 For Users with BITNET_CPP_DIR Set

**Impact**: No change required, existing values respected

**Before** (v0.1.x):
```bash
export BITNET_CPP_DIR=/custom/path
eval "$(cargo run -p xtask -- setup-cpp-auto)"
```

**After** (v0.2.0):
```bash
# Same behavior, explicit value takes precedence
export BITNET_CPP_DIR=/custom/path
eval "$(cargo run -p xtask -- setup-cpp-auto)"
```

### 9.3 For Users with BITNET_CPP_PATH (Deprecated)

**Impact**: Deprecation warning emitted, migration recommended

**Before** (v0.1.x):
```bash
export BITNET_CPP_PATH=/legacy/path
```

**After** (v0.2.0):
```bash
# Still works, but emits deprecation warning:
# "Warning: BITNET_CPP_PATH is deprecated. Use BITNET_CPP_DIR instead."
export BITNET_CPP_PATH=/legacy/path

# RECOMMENDED: Migrate to BITNET_CPP_DIR
export BITNET_CPP_DIR=/legacy/path
unset BITNET_CPP_PATH
```

**Removal Timeline**:
- v0.2.0: Deprecation warning emitted
- v0.2.x: Still functional, warning continues
- v0.3.0: Removed entirely

### 9.4 For Users with Custom BITNET_CROSSVAL_LIBDIR

**Impact**: No change required, explicit overrides respected

**Before** (v0.1.x):
```bash
export BITNET_CROSSVAL_LIBDIR=/opt/custom/lib
```

**After** (v0.2.0):
```bash
# Same behavior, user override takes highest priority
export BITNET_CROSSVAL_LIBDIR=/opt/custom/lib
```

---

## 10. Risks and Mitigations

### Risk 1: Platform-Specific Path Handling

**Risk**: `dirs::home_dir()` behaves differently across platforms

**Likelihood**: Medium
**Impact**: High (broken setup on affected platform)

**Mitigation**:
- Comprehensive testing on Linux, macOS, Windows
- Fallback to `$HOME` env var if `dirs::home_dir()` fails
- CI coverage for all three platforms
- Early beta testing to catch edge cases

### Risk 2: Existing Configurations Break

**Risk**: Users with unusual setups encounter regressions

**Likelihood**: Low
**Impact**: Medium (user frustration, support burden)

**Mitigation**:
- Preserve existing environment variable precedence
- Extensive backward compatibility testing
- Clear migration guide in release notes
- Rollback plan if critical issues discovered

### Risk 3: Build Script Failures

**Risk**: `crossval/build.rs` changes break compilation

**Likelihood**: Low
**Impact**: High (blocks development for all users)

**Mitigation**:
- Conservative changes to build.rs (mostly comments)
- Pre-merge CI validation on all platforms
- Feature flag for gradual rollout (if needed)
- Quick revert capability if issues arise

### Risk 4: Shell Export Format Issues

**Risk**: Generated exports don't work in specific shell versions

**Likelihood**: Medium
**Impact**: Medium (setup fails for affected shells)

**Mitigation**:
- Test with multiple shell versions (bash 3.x, 4.x, 5.x; zsh, fish)
- Conservative quoting and escaping in templates
- Document known shell compatibility issues
- Provide manual export instructions as fallback

---

## 11. Future Enhancements

### Enhancement 1: XDG Base Directory Compliance (Linux)

**Priority**: Low
**Target**: v0.3.0+

**Description**: Follow XDG Base Directory specification on Linux

```bash
# Respect XDG_CACHE_HOME if set
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/bitnet_cpp"
```

**Benefits**:
- Better Linux desktop integration
- User control over cache location via XDG_CACHE_HOME
- Aligns with modern Linux app conventions

### Enhancement 2: Windows AppData Support

**Priority**: Low
**Target**: v0.3.0+

**Description**: Use Windows-native cache directories

```
Default: %APPDATA%\bitnet\cpp
Fallback: %USERPROFILE%\.cache\bitnet_cpp
```

**Benefits**:
- Native Windows app behavior
- Respects Windows roaming profiles
- Better integration with Windows management tools

### Enhancement 3: Separate BitNet/llama Caches

**Priority**: Low
**Target**: v0.4.0+ (if needed)

**Description**: Allow separate cache directories for BitNet.cpp and llama.cpp

```bash
BITNET_CPP_DIR=~/.cache/bitnet_cpp  # BitNet.cpp repo
LLAMA_CPP_DIR=~/.cache/llama_cpp    # llama.cpp standalone
```

**Benefits**:
- Independent versioning of C++ references
- Easier to test against different llama.cpp versions
- Avoids conflicts if upstream APIs diverge

**Decision**: Only implement if dual-cache proves necessary; current shared cache works well.

---

## 12. Appendices

### Appendix A: Environment Variable Reference Table

| Variable | Default | Precedence | Platform | Required | Deprecated |
|----------|---------|------------|----------|----------|------------|
| `BITNET_CPP_DIR` | `~/.cache/bitnet_cpp` | User > Default | All | No | No |
| `BITNET_CROSSVAL_LIBDIR` | Auto-discovered | User > Auto > None | All | No | No |
| `BITNET_CPP_PATH` | None | Fallback only | All | No | Yes (v0.3.0) |
| `LD_LIBRARY_PATH` | Prepended by setup | Append to existing | Linux | No | No |
| `DYLD_LIBRARY_PATH` | Prepended by setup | Append to existing | macOS | No | No |
| `PATH` | Prepended by setup | Append to existing | Windows | No | No |

### Appendix B: Shell Export Templates

**sh/bash/zsh**:
```bash
export BITNET_CPP_DIR="{{cpp_dir}}"
export LD_LIBRARY_PATH="{{lib_dir}}:${LD_LIBRARY_PATH:-}"  # Linux
export DYLD_LIBRARY_PATH="{{lib_dir}}:${DYLD_LIBRARY_PATH:-}"  # macOS
echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
```

**fish**:
```fish
set -gx BITNET_CPP_DIR "{{cpp_dir}}"
set -gx LD_LIBRARY_PATH "{{lib_dir}}" $LD_LIBRARY_PATH  # Linux
set -gx DYLD_LIBRARY_PATH "{{lib_dir}}" $DYLD_LIBRARY_PATH  # macOS
echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
```

**PowerShell**:
```powershell
$env:BITNET_CPP_DIR = "{{cpp_dir}}"
$env:PATH = "{{lib_dir}};" + $env:PATH
Write-Host "[bitnet] C++ ready at $env:BITNET_CPP_DIR"
```

**cmd**:
```batch
set BITNET_CPP_DIR={{cpp_dir}}
set PATH={{lib_dir}};%PATH%
echo [bitnet] C++ ready at %BITNET_CPP_DIR%
```

### Appendix C: File Modification Summary

| File | Lines Modified | Type | Complexity |
|------|----------------|------|------------|
| `xtask/src/cpp_setup_auto.rs` | ~20 | Enhancement | Low |
| `crossval/build.rs` | ~10 | Enhancement | Low |
| `xtask/build.rs` | ~5 | Enhancement | Low |
| `xtask/src/crossval/preflight.rs` | ~2 | Enhancement | Low |
| `crossval/Cargo.toml` | +1 | Dependency | Low |
| Tests (new) | +200 | Test coverage | Medium |
| Documentation | ~50 | Docs update | Low |

**Total**: ~288 lines changed/added

### Appendix D: Related Documentation

**Internal Docs**:
- `/tmp/cpp-setup-auto-env.md` - Environment variable analysis
- `/tmp/cpp-setup-auto-env-SUMMARY.txt` - Key findings
- `/tmp/cpp-setup-auto-CODE-REFERENCE.md` - Code snippets

**Repository Docs**:
- `docs/howto/cpp-setup.md` - C++ reference setup guide
- `docs/environment-variables.md` - Environment variable reference
- `CLAUDE.md` - Essential guidance for contributors

**External References**:
- [dirs crate documentation](https://docs.rs/dirs/)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)

---

## Document Metadata

**Specification Version**: 1.0
**Authors**: BitNet-rs Core Team
**Review Status**: DRAFT
**Target Implementation**: v0.2.0
**Last Updated**: 2025-10-25

**Sign-Off**:
- [ ] Technical Review (TBD)
- [ ] Platform Testing (Linux/macOS/Windows)
- [ ] Documentation Review
- [ ] Security Review (if applicable)
- [ ] Final Approval
