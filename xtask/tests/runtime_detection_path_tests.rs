//! Runtime Detection Path Tests
//!
//! Tests for `detect_backend_runtime()` precedence rules and dual-lib requirements.
//!
//! **Specification**: `/tmp/phase2_test_flip_specification.md` (Category 3, lines 1131-1529)
//!
//! Priority order:
//! 1. BITNET_CROSSVAL_LIBDIR (explicit override)
//! 2. CROSSVAL_RPATH_BITNET / CROSSVAL_RPATH_LLAMA (backend-specific)
//! 3. BITNET_CPP_DIR/build, LLAMA_CPP_DIR/build + subdirs (home dir + subdir search)
//!
//! **Test Coverage (12 tests)**:
//! - Priority 1: BITNET_CROSSVAL_LIBDIR tests (4 tests)
//! - Priority 2: CROSSVAL_RPATH_* tests (4 tests)
//! - Priority 3: *_CPP_DIR + subdir tests (2 tests)
//! - Dual-lib requirements (2 tests)
//!
//! **Critical Requirements**:
//! - All tests use `#[serial(bitnet_env)]` for environment isolation
//! - All tests use `temp_env::with_var()` for safe env mutations
//! - No EnvGuard usage (use temp_env only per spec)
//! - Test both bitnet and llama backends
//! - Verify subdirectory search: build, build/bin, build/lib
//! - Enforce dual-lib requirement for llama only

// Note: Tests work with both `inference` and `crossval-all` features
// Using `inference` feature for now to avoid C++ FFI build dependency (Issue #469)
// The detect_backend_runtime() function is available via bitnet-crossval which is included
// in the inference feature. This allows tests to compile even when C++ backend has issues.
//
// To run tests:
//   cargo test -p xtask --test runtime_detection_path_tests --features inference
//   cargo test -p xtask --test runtime_detection_path_tests --features crossval-all (when FFI fixed)
#![cfg(any(feature = "inference", feature = "crossval-all"))]

use serial_test::serial;
use std::fs;
use temp_env::with_var;
use tempfile::TempDir;
use xtask::crossval::backend::{CppBackend, detect_backend_runtime};

// ============================================================================
// Priority 1 Tests: BITNET_CROSSVAL_LIBDIR (Highest Priority)
// ============================================================================

/// Tests feature spec: phase2_test_flip_specification.md#test_priority_1_bitnet_crossval_libdir_overrides_all
/// AC:Priority-1 - BITNET_CROSSVAL_LIBDIR overrides all other environment variables
///
/// **Given**: BITNET_CROSSVAL_LIBDIR set with valid libs
/// **And**: CROSSVAL_RPATH_BITNET also set with different path
/// **When**: detect_backend_runtime(BitNet) is called
/// **Then**: Should return BITNET_CROSSVAL_LIBDIR path (priority 1)
#[test]
#[serial(bitnet_env)]
fn test_priority_1_bitnet_crossval_libdir_overrides_all() {
    // Create two temp directories
    let priority1_dir = TempDir::new().unwrap();
    let priority2_dir = TempDir::new().unwrap();

    // Create mock lib in priority 1 dir
    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(priority1_dir.path()).unwrap();
        fs::write(priority1_dir.path().join("libbitnet.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(priority1_dir.path()).unwrap();
        fs::write(priority1_dir.path().join("libbitnet.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(priority1_dir.path()).unwrap();
        fs::write(priority1_dir.path().join("bitnet.dll"), "").unwrap();
    }

    // Create different mock lib in priority 2 dir
    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(priority2_dir.path()).unwrap();
        fs::write(priority2_dir.path().join("libbitnet.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(priority2_dir.path()).unwrap();
        fs::write(priority2_dir.path().join("libbitnet.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(priority2_dir.path()).unwrap();
        fs::write(priority2_dir.path().join("bitnet.dll"), "").unwrap();
    }

    // Test: Priority 1 (BITNET_CROSSVAL_LIBDIR) overrides Priority 2 (CROSSVAL_RPATH_BITNET)
    with_var("BITNET_CROSSVAL_LIBDIR", Some(priority1_dir.path().to_str().unwrap()), || {
        with_var("CROSSVAL_RPATH_BITNET", Some(priority2_dir.path().to_str().unwrap()), || {
            with_var("BITNET_CPP_DIR", None::<&str>, || {
                let (found, path) =
                    detect_backend_runtime(CppBackend::BitNet).expect("detection should succeed");

                assert!(found, "Backend should be detected");
                assert_eq!(
                    path.unwrap().canonicalize().unwrap(),
                    priority1_dir.path().canonicalize().unwrap(),
                    "Matched path should equal BITNET_CROSSVAL_LIBDIR (priority 1), not CROSSVAL_RPATH_BITNET (priority 2)"
                );
            });
        });
    });
}

/// Tests feature spec: phase2_test_flip_specification.md#test_priority_1_bitnet_crossval_libdir_overrides_all
/// AC:Priority-1 - BITNET_CROSSVAL_LIBDIR overrides all (llama backend variant)
///
/// **Given**: BITNET_CROSSVAL_LIBDIR set with valid llama libs (dual-lib)
/// **And**: CROSSVAL_RPATH_LLAMA also set with different path
/// **When**: detect_backend_runtime(Llama) is called
/// **Then**: Should return BITNET_CROSSVAL_LIBDIR path (priority 1)
#[test]
#[serial(bitnet_env)]
fn test_priority_1_llama_crossval_libdir_overrides_all() {
    let priority1_dir = TempDir::new().unwrap();
    let priority2_dir = TempDir::new().unwrap();

    // Create mock libs in priority 1 dir (llama requires both libllama and libggml)
    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(priority1_dir.path()).unwrap();
        fs::write(priority1_dir.path().join("libllama.so"), "").unwrap();
        fs::write(priority1_dir.path().join("libggml.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(priority1_dir.path()).unwrap();
        fs::write(priority1_dir.path().join("libllama.dylib"), "").unwrap();
        fs::write(priority1_dir.path().join("libggml.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(priority1_dir.path()).unwrap();
        fs::write(priority1_dir.path().join("llama.dll"), "").unwrap();
        fs::write(priority1_dir.path().join("ggml.dll"), "").unwrap();
    }

    // Create different mock libs in priority 2 dir
    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(priority2_dir.path()).unwrap();
        fs::write(priority2_dir.path().join("libllama.so"), "").unwrap();
        fs::write(priority2_dir.path().join("libggml.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(priority2_dir.path()).unwrap();
        fs::write(priority2_dir.path().join("libllama.dylib"), "").unwrap();
        fs::write(priority2_dir.path().join("libggml.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(priority2_dir.path()).unwrap();
        fs::write(priority2_dir.path().join("llama.dll"), "").unwrap();
        fs::write(priority2_dir.path().join("ggml.dll"), "").unwrap();
    }

    // Test: Priority 1 overrides Priority 2
    with_var("BITNET_CROSSVAL_LIBDIR", Some(priority1_dir.path().to_str().unwrap()), || {
        with_var("CROSSVAL_RPATH_LLAMA", Some(priority2_dir.path().to_str().unwrap()), || {
            with_var("LLAMA_CPP_DIR", None::<&str>, || {
                let (found, path) =
                    detect_backend_runtime(CppBackend::Llama).expect("detection should succeed");

                assert!(found, "Llama backend should be detected");
                assert_eq!(
                    path.unwrap().canonicalize().unwrap(),
                    priority1_dir.path().canonicalize().unwrap(),
                    "Matched path should equal BITNET_CROSSVAL_LIBDIR (priority 1), not CROSSVAL_RPATH_LLAMA (priority 2)"
                );
            });
        });
    });
}

/// Tests feature spec: phase2_test_flip_specification.md#test_priority_1_bitnet_crossval_libdir_missing_libs
/// AC:Priority-1 - BITNET_CROSSVAL_LIBDIR with missing libs returns None
///
/// **Given**: BITNET_CROSSVAL_LIBDIR set to directory without libs
/// **When**: detect_backend_runtime(BitNet) is called
/// **Then**: Should return (false, None) - no detection
#[test]
#[serial(bitnet_env)]
fn test_priority_1_crossval_libdir_missing_libs_returns_none() {
    let empty_dir = TempDir::new().unwrap();
    fs::create_dir_all(empty_dir.path()).unwrap();

    // Directory exists but no libs
    with_var("BITNET_CROSSVAL_LIBDIR", Some(empty_dir.path().to_str().unwrap()), || {
        with_var("BITNET_CPP_DIR", None::<&str>, || {
            with_var("CROSSVAL_RPATH_BITNET", None::<&str>, || {
                let (found, path) =
                    detect_backend_runtime(CppBackend::BitNet).expect("detection should succeed");

                assert!(!found, "Backend should NOT be detected (missing libs)");
                assert!(path.is_none(), "Matched path should be None (missing libs)");
            });
        });
    });
}

/// Tests feature spec: phase2_test_flip_specification.md#test_priority_1_crossval_libdir_subdirs
/// AC:Priority-1 - BITNET_CROSSVAL_LIBDIR with subdirs (build/bin, build/lib)
///
/// **Given**: BITNET_CROSSVAL_LIBDIR set to root, libs in build/bin subdir
/// **When**: detect_backend_runtime(BitNet) is called
/// **Then**: Should NOT search subdirs (exact path only for Priority 1)
#[test]
#[serial(bitnet_env)]
fn test_priority_1_crossval_libdir_subdirs_not_searched() {
    let root_dir = TempDir::new().unwrap();
    let build_bin_dir = root_dir.path().join("build").join("bin");
    fs::create_dir_all(&build_bin_dir).unwrap();

    // Create libs in subdir
    #[cfg(target_os = "linux")]
    {
        fs::write(build_bin_dir.join("libbitnet.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::write(build_bin_dir.join("libbitnet.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::write(build_bin_dir.join("bitnet.dll"), "").unwrap();
    }

    // Priority 1 points to root (not subdir) - should NOT find libs
    with_var("BITNET_CROSSVAL_LIBDIR", Some(root_dir.path().to_str().unwrap()), || {
        with_var("BITNET_CPP_DIR", None::<&str>, || {
            with_var("CROSSVAL_RPATH_BITNET", None::<&str>, || {
                let (found, _path) =
                    detect_backend_runtime(CppBackend::BitNet).expect("detection should succeed");

                assert!(
                    !found,
                    "Backend should NOT be detected (libs in subdir, Priority 1 is exact path only)"
                );
            });
        });
    });
}

// ============================================================================
// Priority 2 Tests: CROSSVAL_RPATH_* (Backend-Specific)
// ============================================================================

/// Tests feature spec: phase2_test_flip_specification.md#test_priority_2_crossval_rpath_bitnet
/// AC:Priority-2 - CROSSVAL_RPATH_BITNET takes precedence over BITNET_CPP_DIR
///
/// **Given**: CROSSVAL_RPATH_BITNET set with valid libs
/// **And**: BITNET_CPP_DIR also set with different path
/// **When**: detect_backend_runtime(BitNet) is called
/// **Then**: Should return CROSSVAL_RPATH_BITNET path (priority 2 over priority 3)
#[test]
#[serial(bitnet_env)]
fn test_priority_2_crossval_rpath_bitnet() {
    let rpath_dir = TempDir::new().unwrap();
    let cpp_dir_root = TempDir::new().unwrap();
    let cpp_dir_build = cpp_dir_root.path().join("build");

    // Create libs in RPATH dir
    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(rpath_dir.path()).unwrap();
        fs::write(rpath_dir.path().join("libbitnet.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(rpath_dir.path()).unwrap();
        fs::write(rpath_dir.path().join("libbitnet.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(rpath_dir.path()).unwrap();
        fs::write(rpath_dir.path().join("bitnet.dll"), "").unwrap();
    }

    // Create different libs in CPP_DIR/build
    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(&cpp_dir_build).unwrap();
        fs::write(cpp_dir_build.join("libbitnet.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(&cpp_dir_build).unwrap();
        fs::write(cpp_dir_build.join("libbitnet.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(&cpp_dir_build).unwrap();
        fs::write(cpp_dir_build.join("bitnet.dll"), "").unwrap();
    }

    // Test: CROSSVAL_RPATH_BITNET (priority 2) used when priority 1 unset
    with_var("BITNET_CROSSVAL_LIBDIR", None::<&str>, || {
        with_var("CROSSVAL_RPATH_BITNET", Some(rpath_dir.path().to_str().unwrap()), || {
            with_var("BITNET_CPP_DIR", Some(cpp_dir_root.path().to_str().unwrap()), || {
                let (found, path) =
                    detect_backend_runtime(CppBackend::BitNet).expect("detection should succeed");

                assert!(found, "Backend should be detected");
                assert_eq!(
                    path.unwrap().canonicalize().unwrap(),
                    rpath_dir.path().canonicalize().unwrap(),
                    "Matched path should equal CROSSVAL_RPATH_BITNET (priority 2), not BITNET_CPP_DIR/build (priority 3)"
                );
            });
        });
    });
}

/// Tests feature spec: phase2_test_flip_specification.md#test_priority_2_crossval_rpath_llama
/// AC:Priority-2 - CROSSVAL_RPATH_LLAMA takes precedence over LLAMA_CPP_DIR
///
/// **Given**: CROSSVAL_RPATH_LLAMA set with valid dual libs
/// **And**: LLAMA_CPP_DIR also set with different path
/// **When**: detect_backend_runtime(Llama) is called
/// **Then**: Should return CROSSVAL_RPATH_LLAMA path (priority 2)
#[test]
#[serial(bitnet_env)]
fn test_priority_2_crossval_rpath_llama() {
    let rpath_dir = TempDir::new().unwrap();
    let cpp_dir_root = TempDir::new().unwrap();
    let cpp_dir_build = cpp_dir_root.path().join("build");

    // Create dual libs in RPATH dir
    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(rpath_dir.path()).unwrap();
        fs::write(rpath_dir.path().join("libllama.so"), "").unwrap();
        fs::write(rpath_dir.path().join("libggml.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(rpath_dir.path()).unwrap();
        fs::write(rpath_dir.path().join("libllama.dylib"), "").unwrap();
        fs::write(rpath_dir.path().join("libggml.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(rpath_dir.path()).unwrap();
        fs::write(rpath_dir.path().join("llama.dll"), "").unwrap();
        fs::write(rpath_dir.path().join("ggml.dll"), "").unwrap();
    }

    // Create different libs in CPP_DIR/build
    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(&cpp_dir_build).unwrap();
        fs::write(cpp_dir_build.join("libllama.so"), "").unwrap();
        fs::write(cpp_dir_build.join("libggml.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(&cpp_dir_build).unwrap();
        fs::write(cpp_dir_build.join("libllama.dylib"), "").unwrap();
        fs::write(cpp_dir_build.join("libggml.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(&cpp_dir_build).unwrap();
        fs::write(cpp_dir_build.join("llama.dll"), "").unwrap();
        fs::write(cpp_dir_build.join("ggml.dll"), "").unwrap();
    }

    // Test: CROSSVAL_RPATH_LLAMA (priority 2, backend-specific)
    with_var("BITNET_CROSSVAL_LIBDIR", None::<&str>, || {
        with_var("CROSSVAL_RPATH_LLAMA", Some(rpath_dir.path().to_str().unwrap()), || {
            with_var("LLAMA_CPP_DIR", Some(cpp_dir_root.path().to_str().unwrap()), || {
                let (found, path) =
                    detect_backend_runtime(CppBackend::Llama).expect("detection should succeed");

                assert!(found, "Llama should be detected with both libllama and libggml");
                assert_eq!(
                    path.unwrap().canonicalize().unwrap(),
                    rpath_dir.path().canonicalize().unwrap(),
                    "Matched path should equal CROSSVAL_RPATH_LLAMA (priority 2), not LLAMA_CPP_DIR/build (priority 3)"
                );
            });
        });
    });
}

/// Tests feature spec: phase2_test_flip_specification.md#test_priority_2_crossval_rpath_subdirs
/// AC:Priority-2 - CROSSVAL_RPATH_* with subdirs (build, build/bin, build/lib)
///
/// **Given**: CROSSVAL_RPATH_BITNET set to root, libs in build/bin subdir
/// **When**: detect_backend_runtime(BitNet) is called
/// **Then**: Should NOT search subdirs (exact path only for Priority 2)
#[test]
#[serial(bitnet_env)]
fn test_priority_2_crossval_rpath_subdirs_not_searched() {
    let root_dir = TempDir::new().unwrap();
    let build_bin_dir = root_dir.path().join("build").join("bin");
    fs::create_dir_all(&build_bin_dir).unwrap();

    // Create libs in subdir
    #[cfg(target_os = "linux")]
    {
        fs::write(build_bin_dir.join("libbitnet.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::write(build_bin_dir.join("libbitnet.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::write(build_bin_dir.join("bitnet.dll"), "").unwrap();
    }

    // Priority 2 points to root (not subdir) - should NOT find libs
    with_var("BITNET_CROSSVAL_LIBDIR", None::<&str>, || {
        with_var("CROSSVAL_RPATH_BITNET", Some(root_dir.path().to_str().unwrap()), || {
            with_var("BITNET_CPP_DIR", None::<&str>, || {
                let (found, _path) =
                    detect_backend_runtime(CppBackend::BitNet).expect("detection should succeed");

                assert!(
                    !found,
                    "Backend should NOT be detected (libs in subdir, Priority 2 is exact path only)"
                );
            });
        });
    });
}

/// Tests feature spec: phase2_test_flip_specification.md#test_mixed_precedence_p1_takes_precedence
/// AC:Priority-Mixed - P1 env set (takes precedence), P2 env also set (ignored)
///
/// **Given**: Both BITNET_CROSSVAL_LIBDIR (P1) and CROSSVAL_RPATH_BITNET (P2) set
/// **When**: detect_backend_runtime(BitNet) is called
/// **Then**: Should return P1 path (P1 > P2)
#[test]
#[serial(bitnet_env)]
fn test_mixed_precedence_p1_takes_precedence_over_p2() {
    let p1_dir = TempDir::new().unwrap();
    let p2_dir = TempDir::new().unwrap();

    // Create libs in both dirs
    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(p1_dir.path()).unwrap();
        fs::write(p1_dir.path().join("libbitnet.so"), "").unwrap();
        fs::create_dir_all(p2_dir.path()).unwrap();
        fs::write(p2_dir.path().join("libbitnet.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(p1_dir.path()).unwrap();
        fs::write(p1_dir.path().join("libbitnet.dylib"), "").unwrap();
        fs::create_dir_all(p2_dir.path()).unwrap();
        fs::write(p2_dir.path().join("libbitnet.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(p1_dir.path()).unwrap();
        fs::write(p1_dir.path().join("bitnet.dll"), "").unwrap();
        fs::create_dir_all(p2_dir.path()).unwrap();
        fs::write(p2_dir.path().join("bitnet.dll"), "").unwrap();
    }

    // Test: P1 overrides P2
    with_var("BITNET_CROSSVAL_LIBDIR", Some(p1_dir.path().to_str().unwrap()), || {
        with_var("CROSSVAL_RPATH_BITNET", Some(p2_dir.path().to_str().unwrap()), || {
            with_var("BITNET_CPP_DIR", None::<&str>, || {
                let (found, path) =
                    detect_backend_runtime(CppBackend::BitNet).expect("detection should succeed");

                assert!(found, "Backend should be detected");
                assert_eq!(
                    path.unwrap().canonicalize().unwrap(),
                    p1_dir.path().canonicalize().unwrap(),
                    "P1 (BITNET_CROSSVAL_LIBDIR) should take precedence over P2 (CROSSVAL_RPATH_BITNET)"
                );
            });
        });
    });
}

// ============================================================================
// Priority 3 Tests: *_CPP_DIR + Subdirectory Search
// ============================================================================

/// Tests feature spec: phase2_test_flip_specification.md#test_priority_3_cpp_dir_fallback_bitnet
/// AC:Priority-3 - BITNET_CPP_DIR fallback when P1/P2 not set
///
/// **Given**: BITNET_CPP_DIR set with libs in build/ subdir
/// **And**: P1 and P2 env vars unset
/// **When**: detect_backend_runtime(BitNet) is called
/// **Then**: Should return BITNET_CPP_DIR/build path (priority 3 fallback)
#[test]
#[serial(bitnet_env)]
fn test_priority_3_cpp_dir_fallback_bitnet() {
    let root = TempDir::new().unwrap();
    let build_dir = root.path().join("build");

    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(&build_dir).unwrap();
        fs::write(build_dir.join("libbitnet.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(&build_dir).unwrap();
        fs::write(build_dir.join("libbitnet.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(&build_dir).unwrap();
        fs::write(build_dir.join("bitnet.dll"), "").unwrap();
    }

    // Test: BITNET_CPP_DIR/build (first subdir match)
    with_var("BITNET_CROSSVAL_LIBDIR", None::<&str>, || {
        with_var("CROSSVAL_RPATH_BITNET", None::<&str>, || {
            with_var("BITNET_CPP_DIR", Some(root.path().to_str().unwrap()), || {
                let (found, path) =
                    detect_backend_runtime(CppBackend::BitNet).expect("detection should succeed");

                assert!(found, "Backend should be detected in BITNET_CPP_DIR/build");
                assert_eq!(
                    path.unwrap().canonicalize().unwrap(),
                    build_dir.canonicalize().unwrap(),
                    "Should match {{root}}/build (first subdir)"
                );
            });
        });
    });
}

/// Tests feature spec: phase2_test_flip_specification.md#test_priority_3_cpp_dir_fallback_llama
/// AC:Priority-3 - LLAMA_CPP_DIR fallback when P1/P2 not set
///
/// **Given**: LLAMA_CPP_DIR set with dual libs in build/ subdir
/// **And**: P1 and P2 env vars unset
/// **When**: detect_backend_runtime(Llama) is called
/// **Then**: Should return LLAMA_CPP_DIR/build path (priority 3 fallback)
#[test]
#[serial(bitnet_env)]
fn test_priority_3_cpp_dir_fallback_llama() {
    let root = TempDir::new().unwrap();
    let build_dir = root.path().join("build");

    #[cfg(target_os = "linux")]
    {
        fs::create_dir_all(&build_dir).unwrap();
        fs::write(build_dir.join("libllama.so"), "").unwrap();
        fs::write(build_dir.join("libggml.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::create_dir_all(&build_dir).unwrap();
        fs::write(build_dir.join("libllama.dylib"), "").unwrap();
        fs::write(build_dir.join("libggml.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::create_dir_all(&build_dir).unwrap();
        fs::write(build_dir.join("llama.dll"), "").unwrap();
        fs::write(build_dir.join("ggml.dll"), "").unwrap();
    }

    // Test: LLAMA_CPP_DIR/build (dual-lib fallback)
    with_var("BITNET_CROSSVAL_LIBDIR", None::<&str>, || {
        with_var("CROSSVAL_RPATH_LLAMA", None::<&str>, || {
            with_var("LLAMA_CPP_DIR", Some(root.path().to_str().unwrap()), || {
                let (found, path) =
                    detect_backend_runtime(CppBackend::Llama).expect("detection should succeed");

                assert!(found, "Llama should be detected in LLAMA_CPP_DIR/build");
                assert_eq!(
                    path.unwrap().canonicalize().unwrap(),
                    build_dir.canonicalize().unwrap(),
                    "Should match {{root}}/build with both libllama and libggml"
                );
            });
        });
    });
}

// ============================================================================
// Dual-Lib Requirement Tests (Llama-Specific)
// ============================================================================

/// Tests feature spec: phase2_test_flip_specification.md#test_llama_requires_both_libllama_and_libggml
/// AC:Dual-Lib - Llama requires both libllama and libggml (fail if only one present)
///
/// **Given**: Directory with only libllama.so (missing libggml.so)
/// **When**: detect_backend_runtime(Llama) is called
/// **Then**: Should return (false, None) - dual-lib requirement not met
///
/// **Given**: Directory with both libllama.so and libggml.so
/// **When**: detect_backend_runtime(Llama) is called
/// **Then**: Should return (true, Some(path)) - dual-lib requirement met
#[test]
#[serial(bitnet_env)]
fn test_llama_requires_both_libllama_and_libggml() {
    let temp = TempDir::new().unwrap();

    // Test case 1: Only libllama (missing libggml) → should NOT match
    #[cfg(target_os = "linux")]
    {
        fs::write(temp.path().join("libllama.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::write(temp.path().join("libllama.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::write(temp.path().join("llama.dll"), "").unwrap();
    }

    with_var("BITNET_CROSSVAL_LIBDIR", Some(temp.path().to_str().unwrap()), || {
        let (found, _) =
            detect_backend_runtime(CppBackend::Llama).expect("detection should succeed");

        assert!(
            !found,
            "Llama requires BOTH libllama AND libggml - should not match with only libllama"
        );
    });

    // Test case 2: Add libggml → should NOW match
    #[cfg(target_os = "linux")]
    {
        fs::write(temp.path().join("libggml.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::write(temp.path().join("libggml.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::write(temp.path().join("ggml.dll"), "").unwrap();
    }

    with_var("BITNET_CROSSVAL_LIBDIR", Some(temp.path().to_str().unwrap()), || {
        let (found, path) =
            detect_backend_runtime(CppBackend::Llama).expect("detection should succeed");

        assert!(found, "Llama should match with BOTH libllama AND libggml");
        assert_eq!(path.unwrap().canonicalize().unwrap(), temp.path().canonicalize().unwrap());
    });
}

/// Tests feature spec: phase2_test_flip_specification.md#test_bitnet_requires_only_libbitnet
/// AC:Dual-Lib - BitNet requires only libbitnet (single-lib, no dual requirement)
///
/// **Given**: Directory with only libbitnet.so
/// **When**: detect_backend_runtime(BitNet) is called
/// **Then**: Should return (true, Some(path)) - single-lib requirement met
#[test]
#[serial(bitnet_env)]
fn test_bitnet_requires_only_libbitnet() {
    let temp = TempDir::new().unwrap();

    #[cfg(target_os = "linux")]
    {
        fs::write(temp.path().join("libbitnet.so"), "").unwrap();
    }
    #[cfg(target_os = "macos")]
    {
        fs::write(temp.path().join("libbitnet.dylib"), "").unwrap();
    }
    #[cfg(target_os = "windows")]
    {
        fs::write(temp.path().join("bitnet.dll"), "").unwrap();
    }

    with_var("BITNET_CROSSVAL_LIBDIR", Some(temp.path().to_str().unwrap()), || {
        let (found, path) =
            detect_backend_runtime(CppBackend::BitNet).expect("detection should succeed");

        assert!(found, "BitNet should match with only libbitnet (no dual-lib requirement)");
        assert_eq!(path.unwrap().canonicalize().unwrap(), temp.path().canonicalize().unwrap());
    });
}
