# Quick Reference: Environment Export → build.rs Gap

## The Problem in 30 Seconds

```
┌─ setup-cpp-auto               ┌─ rebuild_xtask()
│  (installs libs)              │  (runs cargo build)
│  outputs: export VAR="..."    │
│           export PATH="..."   │
│                               │
└─→ [SUBPROCESS STDOUT]        ├─ Inherits PARENT env (stale!)
   [OUTPUT DISCARDED]           │
                                ├─ Does NOT see new BITNET_CPP_DIR
                                │
                                └─→ build.rs can't find libraries
                                    HAS_BITNET stays false
```

## The Three Gaps

| Gap | Location | Problem | Impact |
|-----|----------|---------|--------|
| **Gap 1** | `preflight.rs:1970` | `setup-cpp-auto` stdout captured but never read | Env exports lost |
| **Gap 2** | `preflight.rs:1404` | `rebuild_xtask()` doesn't pass env to child cargo | Child doesn't see BITNET_CPP_DIR |
| **Gap 3** | `preflight.rs` | No parsing function exists | Can't extract `export VAR=value` lines |

## Solution: 4-Step Fix

### Step 1: Parse Shell Exports (NEW FUNCTION)
```rust
fn parse_sh_exports(output: &str) -> HashMap<String, String> {
    // Extracts: export BITNET_CPP_DIR="/path"
    //           export LD_LIBRARY_PATH="/path:..."
    //           set -gx VAR "value" (fish)
    //           $env:VAR = "value" (pwsh)
}
```

### Step 2: Apply to Environment (NEW FUNCTION)
```rust
fn apply_env_exports(exports: &HashMap<String, String>) {
    for (key, value) in exports {
        unsafe { env::set_var(key, value) }
    }
}
```

### Step 3: Modify Repair Flow (CHANGE EXISTING)
```rust
// Line 1393-1404 in preflight_with_auto_repair()
let setup_output = attempt_repair_with_retry(backend, verbose)?;
let exports = parse_sh_exports(&setup_output)?;
apply_env_exports(&exports)?;
rebuild_xtask_with_env(verbose, &exports)?;
```

### Step 4: Pass Env to Child Cargo (CHANGE EXISTING)
```rust
fn rebuild_xtask_with_env(verbose: bool, exports: &HashMap<String, String>) {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "-p", "xtask", "--features", "crossval-all"]);
    for (key, value) in exports {
        cmd.env(key, value);  // ← Pass to child
    }
    cmd.status()?;
}
```

## Expected Output After Fix

### Before (BROKEN)
```
[repair] Step 2/3: Rebuilding xtask binary...
cargo:warning=crossval: ✗ BITNET_STUB mode: No C++ libraries found
cargo:warning=crossval: Set BITNET_CPP_DIR to enable C++ backend integration
```

### After (FIXED)
```
[repair] Step 2/3: Rebuilding xtask binary...
cargo:warning=crossval: ✓ BITNET_FULL: BitNet.cpp and llama.cpp libraries found
cargo:warning=crossval: Backend: full
cargo:warning=crossval: Linked libraries: bitnet, llama, ggml
```

## Key Files to Modify

| File | Lines | Change | Reason |
|------|-------|--------|--------|
| `xtask/src/crossval/preflight.rs` | (new) | Add `parse_sh_exports()` | Parse stdout from setup-cpp-auto |
| `xtask/src/crossval/preflight.rs` | (new) | Add `apply_env_exports()` | Set vars in current process |
| `xtask/src/crossval/preflight.rs` | 1393-1407 | Capture + parse + apply | Integrate parsing into repair |
| `xtask/src/crossval/preflight.rs` | 1617 | Add `rebuild_xtask_with_env()` | Pass env to child cargo |
| `xtask/src/crossval/preflight.rs` | 1970-1976 | Return stdout from setup | Capture output for parsing |

## Testing Checklist

- [ ] `parse_sh_exports()` extracts sh format correctly
- [ ] `parse_sh_exports()` extracts fish format correctly
- [ ] `parse_sh_exports()` extracts PowerShell format correctly
- [ ] `apply_env_exports()` sets vars in current process
- [ ] `rebuild_xtask_with_env()` passes vars to child cargo
- [ ] Integration test: full repair → rebuild → detection succeeds
- [ ] Verify HAS_BITNET = true after re-exec

## References

- **Current Gap Analysis**: `docs/analysis/env-export-build-gap-analysis.md` (full details)
- **setup-cpp-auto**: `xtask/src/cpp_setup_auto.rs:707-866`
- **Repair Flow**: `xtask/src/crossval/preflight.rs:1326-1936`
- **Build Detection**: `crossval/build.rs:131-189`
- **Constants**: `crossval/build.rs:340-344` (HAS_BITNET/HAS_LLAMA emission)

## Acceptance Criteria

1. **AC1**: parse_sh_exports() correctly parses all shell formats
2. **AC2**: apply_env_exports() propagates vars to child processes
3. **AC3**: rebuild_xtask_with_env() applies env before cargo build
4. **AC4**: After repair + rebuild + re-exec, HAS_BITNET = true (no BITNET_STUB)

## Implementation Estimate

- **Parsing Functions**: 2-3 hours (with unit tests)
- **Integration**: 1-2 hours (modify repair flow)
- **Testing**: 2-3 hours (unit + integration + e2e)
- **Total**: ~5-8 hours

## Risk Level: MEDIUM

**Mitigation**:
- Keep `rebuild_xtask()` as standalone fallback
- Comprehensive unit tests for parsing
- Graceful error handling (fallback to default paths)
- Verbose logging of env vars passed to cargo
