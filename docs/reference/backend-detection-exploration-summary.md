# Backend Detection & Device Selection Exploration Summary

## Exploration Completed: Medium Thoroughness

This exploration documents the backend detection and device selection patterns used throughout BitNet-rs, with a focus on identifying reusable patterns for adding C++ backend support.

## Key Discoveries

### 1. **Two-Tier Architecture** (Foundation Pattern)
BitNet-rs implements a universal two-tier backend detection approach:
- **Compile-time**: Feature gates (`--features gpu`, `--features cuda`)
- **Runtime**: System checks + environment overrides

**Files**: 
- `/crates/bitnet-kernels/src/device_features.rs` (Master detection API)
- `/crates/bitnet-kernels/src/gpu_utils.rs` (Runtime detection)

### 2. **Environment Variable Hierarchy** (Configuration Pattern)
Three-level precedence for environment-driven selection:

```
BITNET_GPU_FAKE (testing override)
  ↓ (unless BITNET_STRICT_MODE=1)
BITNET_STRICT_MODE (safety override)
  ↓ (if not set)
Real detection (nvidia-smi, rocm-smi, etc.)
```

This pattern is immediately reusable for C++ backend with `BITNET_CPP_FAKE` and `BITNET_CPP_DIR`.

### 3. **Provider Registry Pattern** (Selection Architecture)
Kernel manager uses ordered provider list with lazy caching:

```rust
providers = [GPU, AVX512, AVX2, NEON, FFI, Fallback]
selected = OnceLock<usize>  // Single cached selection
```

**Key insight**: Easy to add `CppBackend` as a new provider without changing selection logic.

### 4. **Graceful Fallback Chain** (Resilience Pattern)
Every backend selection has guaranteed fallback:
- GPU → fails? → CPU (with version selection)
- CPU AVX2 → fails? → CPU scalar
- C++ → fails? → GPU or CPU

No single point of failure.

### 5. **Dual-Provider Model** (Quantization)
DeviceAwareQuantizer uses primary+fallback:
```rust
primary_provider: Option<CudaKernel>  // Preferred
fallback_provider: CpuKernel         // Always available
```

Perfect pattern for C++ backend selection (try C++, fallback to GPU/CPU).

## Critical Files Analyzed

| File | Lines | Purpose | Key Pattern |
|------|-------|---------|------------|
| `bitnet-kernels/src/device_features.rs` | 165 | Device detection API | Two-tier checks |
| `bitnet-kernels/src/gpu_utils.rs` | 217 | GPU runtime detection | Command-based detection |
| `bitnet-inference/src/backends.rs` | 470 | Backend abstraction | Trait-based selection |
| `bitnet-kernels/src/lib.rs` | 214 | Kernel manager | Provider registry |
| `bitnet-kernels/src/device_aware.rs` | 150+ | Device-aware selection | Dual-provider fallback |
| `xtask/tests/preflight.rs` | 231 | Diagnostic tests | Fake override validation |
| `bitnet-cli/src/config.rs` | 100+ | CLI configuration | Config structures |

## Reusable Patterns for C++ Backend

### Pattern 1: Compile-Time Check
```rust
#[inline]
pub fn cpp_compiled() -> bool {
    cfg!(feature = "cpp")
}
```

### Pattern 2: Runtime Detection
```rust
pub fn cpp_available_runtime() -> bool {
    // 1. Check BITNET_CPP_FAKE (testing)
    if let Ok(v) = env::var("BITNET_CPP_FAKE") {
        return v == "yes" || v == "1";
    }
    
    // 2. Strict mode check
    if env::var("BITNET_STRICT_MODE").is_ok() {
        return real_cpp_detection();
    }
    
    // 3. Real detection
    check_cpp_dir() || check_standard_paths() || Command::new("bitnet-ref").output().is_ok()
}
```

### Pattern 3: Backend Selection
```rust
pub fn select_cpp_backend() -> Result<Box<dyn Backend>> {
    if !cpp_compiled() { return Err("not compiled"); }
    if !cpp_available_runtime() { return fallback(); }
    
    // Try BITNET_CPP_DIR
    // Try standard paths
    // Try LD_LIBRARY_PATH
    
    Box::new(CppBackendImpl::new()?)
}
```

### Pattern 4: Preflight Diagnostic
```rust
pub fn cpp_capability_summary() -> String {
    format!(
        "C++ Backend:\n\
         Compiled: {}\n\
         Runtime: {}\n\
         Version: {}\n\
         Path: {}",
        if cpp_compiled() { "✓" } else { "✗" },
        if cpp_available_runtime() { "✓" } else { "✗" },
        cpp_version().unwrap_or("unknown"),
        cpp_path().unwrap_or_default()
    )
}
```

## Environment Variables (Current + Proposed)

| Variable | Type | Purpose | Precedence |
|----------|------|---------|-----------|
| `BITNET_GPU_FAKE` | Test | Fake GPU availability | Highest |
| `BITNET_STRICT_MODE` | Safety | Force real detection | Medium |
| `BITNET_CPP_FAKE` | Test | Fake C++ availability | Highest (proposed) |
| `BITNET_CPP_DIR` | Config | C++ installation path | Medium (proposed) |
| `BITNET_STRICT_NO_FAKE_GPU` | Safety | Panic on fake | Safety gate |
| `BITNET_STRICT_NO_FAKE_CPP` | Safety | Panic on fake C++ | Safety gate (proposed) |

## Testing Patterns

The codebase uses these testing patterns that should be replicated for C++:

1. **Feature gate tests**: Validate compile-time detection
2. **Fake override tests**: Validate environment variable masking
3. **Strict mode tests**: Validate safety enforcement
4. **Preflight tests**: Validate output formatting
5. **Edge case tests**: Invalid values, missing tools, etc.

**Test location**: `xtask/tests/preflight.rs` (231 lines, comprehensive)

## Implementation Roadmap (for C++ Backend)

### Phase 1: Detection Infrastructure
- [ ] Add `cpp_compiled()` in new module
- [ ] Add `cpp_available_runtime()` with BITNET_CPP_FAKE support
- [ ] Create `CppInfo` struct (path, version, features)
- [ ] Add `cpp_capability_summary()` for diagnostics

### Phase 2: Backend Implementation
- [ ] Create `CppBackend` struct implementing trait
- [ ] Add to provider registry in kernel manager
- [ ] Integrate with device selection logic
- [ ] Add fallback chain handling

### Phase 3: Configuration & Testing
- [ ] Add `--features cpp` to Cargo.toml
- [ ] Add preflight tests (8-10 test cases)
- [ ] Add integration tests for selection logic
- [ ] Document in CLAUDE.md

### Phase 4: Diagnostics
- [ ] Add xtask preflight support for C++
- [ ] Add `inspect --cpp` command
- [ ] Update health endpoints
- [ ] Create troubleshooting guide

## Key Design Decisions

1. **No defaults for GPU features** → Apply to C++ backend too
2. **Unified feature predicate** (`any(feature = "gpu", feature = "cuda")`) → Use `any(feature = "cpp")` 
3. **Environment overrides for testing** → BITNET_CPP_FAKE for reproducible tests
4. **Strict mode for production** → BITNET_STRICT_MODE prevents testing overrides
5. **Command-based detection** → Use `Command::new()` to probe for tools
6. **Lazy caching of selection** → OnceLock for single initialization
7. **Graceful fallback always** → CPU as ultimate safety net

## Deliverable

**File**: `/docs/explanation/backend-detection-and-device-selection-patterns.md` (693 lines)

Contains:
- Complete architecture documentation
- Code patterns with examples
- Reusable templates
- Implementation checklist for C++ backend
- Environment variable reference
- Testing patterns
- File reference guide

## Conclusion

BitNet-rs has a well-architected, proven backend detection system. The patterns are:
- **Consistent** across GPU, CPU, FFI backends
- **Extensible** for new backends (C++, etc.)
- **Testable** with environment overrides
- **Resilient** with guaranteed fallbacks
- **Observable** with diagnostic commands

Adding C++ backend detection requires only implementing 4-5 functions following exact patterns already proven in production. No new architectural decisions needed.

---
**Exploration Level**: Medium
**Files Analyzed**: 6 primary + 10 supporting
**Patterns Identified**: 9 reusable patterns
**Coverage**: Device detection, runtime selection, environment configuration, testing, diagnostics
