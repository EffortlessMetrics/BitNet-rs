# Error Handling Test Coverage

**Specification:** `docs/specs/cpp-wrapper-error-handling.md`

This document provides traceability between test functions and specification requirements.

## Overview

- **Total Tests:** 44
- **Priority 1 (Must-Have):** 12 tests
- **Priority 2 (Should-Have):** 8 tests
- **Priority 3 (Nice-to-Have):** 24 tests
- **Total LOC:** 1,782 lines

## Test Files

1. **crossval/tests/error_handling_tests.rs** (835 lines, 20 tests)
   - Priority 1 and Priority 2 tests
   - Covers timeout mechanism, cleanup validation, C++ logging, error enum expansion

2. **crossval/tests/error_path_comprehensive_tests.rs** (947 lines, 24 tests)
   - Priority 3 tests
   - Covers comprehensive error paths across 12 categories

## Priority 1: Critical Safety (12 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_timeout_mechanism_default_30s` | §5.1 Timeout Mechanism | Default 30s timeout validation |
| `test_timeout_mechanism_configurable` | §5.1 Timeout Mechanism | Custom timeout duration |
| `test_timeout_prevents_hangs` | §5.1 Timeout Mechanism | Hang prevention verification |
| `test_cleanup_validation_context_counter` | §6.1 Cleanup Validation | AtomicUsize context counter |
| `test_cleanup_validation_valgrind` | §6.1 Cleanup Validation | Valgrind memory leak detection |
| `test_cleanup_on_error_paths` | §6.1 Cleanup Validation | RAII Drop enforcement |
| `test_cpp_logging_macros` | §4.1 C++ Logging | ERROR/WARN/INFO/DEBUG macros |
| `test_cpp_log_capture_in_rust` | §4.2 FFI Error Trace | C++ → Rust log propagation |
| `test_error_context_chaining` | §4.3 Error Context Chaining | anyhow context chains |
| `test_library_not_found_error` | §2.2 Error Enum | LibraryNotFound variant |
| `test_symbol_not_found_error` | §2.2 Error Enum | SymbolNotFound variant |
| `test_out_of_memory_error` | §2.2 Error Enum | OutOfMemory variant |

## Priority 2: Error Completeness (8 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_error_enum_expansion` | §2.2 Error Enum | All 8+ new variants |
| `test_tokenization_phase_error` | §3.2.2 Pass-Phase | Query vs Fill phase (tokenize) |
| `test_inference_phase_error` | §3.2.2 Pass-Phase | Query vs Fill phase (evaluate) |
| `test_context_overflow_error` | §2.2 Error Enum | ContextOverflow variant |
| `test_thread_safety_error` | §2.2 Error Enum | ThreadSafetyError variant |
| `test_cleanup_failed_error` | §2.2 Error Enum | CleanupFailed variant |
| `test_operation_timeout_error` | §2.2 Error Enum | OperationTimeout variant |
| `test_error_message_actionable_guidance` | §2.3 Error Message Guidelines | What/Why/How/Where format |

## Priority 3: Comprehensive Error Paths (24 tests)

### Category 1: Library Availability (3 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_bitnet_lib_not_found` | §1.2 Gap 1 | BitNet.cpp library missing |
| `test_llama_lib_not_found` | §1.2 Gap 1 | llama.cpp library missing |
| `test_both_libs_not_found` | §1.2 Gap 1 | No fallback available |

### Category 2: Symbol Resolution (3 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_missing_required_symbol` | §1.2 Gap 1 | Required symbol unavailable |
| `test_missing_optional_symbol_fallback` | §1.2 Gap 1 | Optional symbol with fallback |
| `test_version_mismatch_symbol` | §1.2 Gap 8 | Version incompatibility |

### Category 3: Model Loading (3 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_invalid_gguf_format` | §8.1 Category 3 | Invalid magic header |
| `test_corrupted_model_file` | §8.1 Category 3 | Mid-file corruption |
| `test_unsupported_quantization` | §8.1 Category 3 | Unsupported quant type |

### Category 4: Inference Operations (3 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_context_size_exceeded` | §8.1 Category 4 | Token count > context size |
| `test_oom_during_inference` | §8.1 Category 4 | malloc failure |
| `test_numerical_instability` | §8.1 Category 4 | NaN/Inf detection |

### Category 5: Buffer Negotiation (2 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_buffer_too_small` | §8.1 Category 5 | Insufficient buffer |
| `test_buffer_overflow_prevention` | §8.1 Category 5 | Safe truncation |

### Category 6: Cleanup on Error (3 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_model_cleanup_on_load_failure` | §8.1 Category 6 | Model load failure |
| `test_context_cleanup_on_init_failure` | §8.1 Category 6 | Context init failure |
| `test_resource_leak_detection` | §8.1 Category 6 | Valgrind 100+ scenarios |

### Category 7: Error Message Quality (2 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_actionable_error_messages` | §8.1 Category 7 | What/Why/How/Where format |
| `test_error_message_consistency` | §8.1 Category 7 | C++ → Rust consistency |

### Category 8-12: Fallback Tests (5 tests)

| Test Function | Spec Section | Coverage |
|---------------|--------------|----------|
| `test_tokenization_fallback_bitnet_to_llama` | §8.1 Category 8 | Tokenization fallback |
| `test_inference_fallback_chain` | §8.1 Category 9 | BitNet → llama → error |
| `test_fallback_performance_overhead` | §8.1 Category 11 | < 5% overhead |
| `test_fallback_consistency_with_direct` | §8.1 Category 12 | Identical results |
| `test_fallback_diagnostics_in_preflight` | §8.1 Category 12 | Preflight diagnostics |

## Feature Gates

All tests use `#[cfg(feature = "crossval-all")]` to ensure proper compilation isolation:

```rust
#![cfg(feature = "crossval-all")]
```

Debug-only tests use additional guards:

```rust
#[test]
#[cfg(debug_assertions)]
fn test_cleanup_validation_context_counter() { ... }
```

## Test Patterns

### Environment Isolation

Tests use `#[serial(bitnet_env)]` for environment variable safety:

```rust
#[test]
#[serial(bitnet_env)]
fn test_timeout_mechanism_default_30s() { ... }
```

### TDD Approach

All tests initially FAIL with `todo!()` macros:

```rust
#[test]
#[ignore] // P1: Must-have - implement timeout infrastructure
fn test_timeout_mechanism_default_30s() {
    todo!("P1: Implement timeout mechanism with default 30s timeout");
}
```

### Specification Traceability

Each test includes specification references:

```rust
/// Tests feature spec: cpp-wrapper-error-handling.md#timeout-mechanism
///
/// **Purpose:** P1: Validate timeout mechanism prevents C++ hangs
/// **Priority:** P1 (Must-Have)
/// **Expected:** Returns OperationTimeout error after 30s default timeout
```

## Acceptance Criteria Coverage

### AC1: Timeout Mechanism (3 tests)

- ✅ Default 30s timeout
- ✅ Configurable timeout
- ✅ Hang prevention
- ✅ Thread leak acceptable

### AC2: Cleanup Validation (3 tests)

- ✅ Context counter (debug builds)
- ✅ Valgrind leak detection
- ✅ RAII Drop enforcement

### AC3: C++ Error Logging (3 tests)

- ✅ Structured logging (DEBUG/INFO/WARN/ERROR)
- ✅ FFI log propagation
- ✅ Error context chaining

### AC4: Error Enum Expansion (8 tests)

- ✅ LibraryNotFound
- ✅ SymbolNotFound
- ✅ OutOfMemory
- ✅ ContextOverflow
- ✅ ThreadSafetyError
- ✅ CleanupFailed
- ✅ OperationTimeout
- ✅ OptionalSymbolMissing

### AC5: Pass-Phase Distinction (2 tests)

- ✅ Tokenization Query vs Fill
- ✅ Inference Query vs Fill

### AC6: Error Context (1 test)

- ✅ anyhow context chaining

### AC7: Diagnostics Flag (1 test)

- ✅ Preflight diagnostics

### AC8: Error Tests (24 tests)

- ✅ Library availability (3 tests)
- ✅ Symbol resolution (3 tests)
- ✅ Model loading (3 tests)
- ✅ Inference operations (3 tests)
- ✅ Buffer negotiation (2 tests)
- ✅ Cleanup on error (3 tests)
- ✅ Error message quality (2 tests)
- ✅ Fallback tests (5 tests)

## Next Steps

1. **Implement Priority 1 Tests** (4-6 hours)
   - Timeout mechanism
   - Cleanup validation
   - C++ error logging

2. **Implement Priority 2 Tests** (5-7 hours)
   - Error enum expansion
   - Pass-phase distinction
   - Error context

3. **Implement Priority 3 Tests** (13-17 hours)
   - Comprehensive error paths
   - Fallback validation
   - Diagnostics integration

4. **Validation**
   - Run tests: `cargo test --features crossval-all`
   - Memory leak check: `valgrind cargo test`
   - Error message quality audit

## References

- **Specification:** `docs/specs/cpp-wrapper-error-handling.md`
- **Analysis:** `/tmp/error_handling_analysis.md`
- **FFI Architecture:** `docs/specs/bitnet-cpp-ffi-sockets.md`
- **Build Detection:** `docs/specs/bitnet-buildrs-detection-enhancement.md`
