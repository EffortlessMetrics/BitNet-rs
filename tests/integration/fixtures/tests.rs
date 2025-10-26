/// Integration Test Fixture Infrastructure Tests
///
/// Comprehensive tests validating DirectoryLayoutBuilder, MockLibrary, and
/// TestEnvironment functionality for BitNet.cpp auto-configuration testing.
///
/// # Test Specification
///
/// Tests feature spec: bitnet-integration-tests.md#implementation-plan-phase-1
use super::*;
use serial_test::serial;
use std::path::PathBuf;

// ============================================================================
// DirectoryLayoutBuilder Tests
// ============================================================================

/// Tests feature spec: bitnet-integration-tests.md#scenario-1-bitnet-standard-layout
#[test]
fn test_bitnet_standard_layout_creation() {
    let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
        .with_libs(true)
        .with_headers(true)
        .build()
        .expect("Failed to create BitNet standard layout");

    // Verify layout type
    assert_eq!(layout.layout_type(), LayoutType::BitNetStandard);

    // Verify root path exists
    assert!(layout.root_path().exists());

    // Verify directory structure
    let root = layout.root_path();
    assert!(root.join("include").exists(), "include/ directory missing");
    assert!(
        root.join("3rdparty/llama.cpp/include").exists(),
        "3rdparty/llama.cpp/include/ directory missing"
    );
    assert!(root.join("build/lib").exists(), "build/lib/ directory missing");
    assert!(
        root.join("build/3rdparty/llama.cpp/src").exists(),
        "build/3rdparty/llama.cpp/src/ directory missing"
    );
    assert!(
        root.join("build/3rdparty/llama.cpp/ggml/src").exists(),
        "build/3rdparty/llama.cpp/ggml/src/ directory missing"
    );
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-1-bitnet-standard-layout
#[test]
fn test_bitnet_standard_layout_headers() {
    let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
        .with_libs(true)
        .with_headers(true)
        .build()
        .expect("Failed to create BitNet standard layout");

    // Verify headers exist
    assert!(layout.has_header("ggml-bitnet.h"), "ggml-bitnet.h header missing");
    assert!(layout.has_header("llama.h"), "llama.h header missing");

    // Verify header content (mock headers)
    let bitnet_header = layout.root_path().join("include/ggml-bitnet.h");
    let content = std::fs::read_to_string(bitnet_header).expect("Failed to read header");
    assert!(content.contains("Mock BitNet header"), "Header content incorrect");
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-1-bitnet-standard-layout
#[test]
fn test_bitnet_standard_layout_libraries() {
    let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
        .with_libs(true)
        .with_headers(true)
        .build()
        .expect("Failed to create BitNet standard layout");

    // Verify libraries exist
    assert!(layout.has_library(LibraryType::BitNet), "libbitnet library missing");
    assert!(layout.has_library(LibraryType::Llama), "libllama library missing");
    assert!(layout.has_library(LibraryType::Ggml), "libggml library missing");

    // Verify library paths
    let lib_paths = layout.lib_paths();
    assert_eq!(lib_paths.len(), 3, "Expected 3 library paths");
    assert!(lib_paths.iter().any(|p| p.ends_with("build/lib")), "build/lib path missing");
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-3-standalone-llama
#[test]
fn test_llama_standalone_layout() {
    let layout = DirectoryLayoutBuilder::new(LayoutType::LlamaStandalone)
        .with_libs(true)
        .with_headers(true)
        .build()
        .expect("Failed to create llama standalone layout");

    // Verify layout type
    assert_eq!(layout.layout_type(), LayoutType::LlamaStandalone);

    // Verify directory structure
    let root = layout.root_path();
    assert!(root.join("include").exists(), "include/ directory missing");
    assert!(root.join("build/bin").exists(), "build/bin/ directory missing");

    // Verify llama header exists
    assert!(layout.has_header("llama.h"), "llama.h header missing");

    // Verify libraries exist
    assert!(layout.has_library(LibraryType::Llama), "libllama library missing");
    assert!(layout.has_library(LibraryType::Ggml), "libggml library missing");

    // Verify BitNet library does NOT exist
    assert!(
        !layout.has_library(LibraryType::BitNet),
        "libbitnet should not exist in llama standalone layout"
    );
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-2-custom-libdir
#[test]
fn test_custom_libdir_layout() {
    let layout = DirectoryLayoutBuilder::new(LayoutType::CustomLibDir)
        .with_libs(true)
        .with_headers(true)
        .build()
        .expect("Failed to create custom libdir layout");

    // Verify layout type
    assert_eq!(layout.layout_type(), LayoutType::CustomLibDir);

    // Verify custom_libs directory exists
    let root = layout.root_path();
    assert!(root.join("custom_libs").exists(), "custom_libs/ directory missing");

    // Verify all libraries in custom directory
    assert!(layout.has_library(LibraryType::BitNet), "libbitnet library missing");
    assert!(layout.has_library(LibraryType::Llama), "libllama library missing");
    assert!(layout.has_library(LibraryType::Ggml), "libggml library missing");

    // Verify lib_paths returns custom_libs
    let lib_paths = layout.lib_paths();
    assert_eq!(lib_paths.len(), 1, "Expected 1 library path");
    assert!(lib_paths[0].ends_with("custom_libs"), "custom_libs path missing");
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-4-dual-backend
#[test]
fn test_dual_backend_layout() {
    let layout = DirectoryLayoutBuilder::new(LayoutType::DualBackend)
        .with_libs(true)
        .with_headers(true)
        .build()
        .expect("Failed to create dual backend layout");

    // Verify layout type
    assert_eq!(layout.layout_type(), LayoutType::DualBackend);

    // Dual backend should have same structure as BitNetStandard
    assert!(layout.has_library(LibraryType::BitNet));
    assert!(layout.has_library(LibraryType::Llama));
    assert!(layout.has_library(LibraryType::Ggml));
    assert!(layout.has_header("ggml-bitnet.h"));
    assert!(layout.has_header("llama.h"));
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-5-missing-libs
#[test]
fn test_missing_libs_layout() {
    let layout = DirectoryLayoutBuilder::new(LayoutType::MissingLibs)
        .with_libs(false)
        .with_headers(true)
        .build()
        .expect("Failed to create missing libs layout");

    // Verify layout type
    assert_eq!(layout.layout_type(), LayoutType::MissingLibs);

    // Verify headers exist
    assert!(layout.has_header("ggml-bitnet.h"), "ggml-bitnet.h header should exist");

    // Verify NO libraries exist
    assert!(!layout.has_library(LibraryType::BitNet), "libbitnet should not exist");
    assert!(!layout.has_library(LibraryType::Llama), "libllama should not exist");
    assert!(!layout.has_library(LibraryType::Ggml), "libggml should not exist");

    // Verify lib_paths is empty
    let lib_paths = layout.lib_paths();
    assert!(lib_paths.is_empty(), "lib_paths should be empty for MissingLibs layout");

    // Verify build/ directory does NOT exist
    let root = layout.root_path();
    assert!(!root.join("build").exists(), "build/ directory should not exist");
}

/// Tests feature spec: bitnet-integration-tests.md#directory-layout-templates
#[test]
fn test_layout_builder_with_libs_disabled() {
    let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
        .with_libs(false)
        .with_headers(true)
        .build()
        .expect("Failed to create layout");

    // Verify headers exist but libraries do not
    assert!(layout.has_header("ggml-bitnet.h"));
    assert!(!layout.has_library(LibraryType::BitNet));
}

/// Tests feature spec: bitnet-integration-tests.md#directory-layout-templates
#[test]
fn test_layout_builder_with_headers_disabled() {
    let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
        .with_libs(true)
        .with_headers(false)
        .build()
        .expect("Failed to create layout");

    // Verify libraries exist but headers do not
    assert!(layout.has_library(LibraryType::BitNet));
    assert!(!layout.has_header("ggml-bitnet.h"));
}

// ============================================================================
// MockLibrary Tests
// ============================================================================

/// Tests feature spec: bitnet-integration-tests.md#scenario-17-linux-so-libraries
#[test]
#[cfg(target_os = "linux")]
fn test_mock_library_linux_elf_header() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let lib_path = temp_dir.path().to_path_buf();

    MockLibrary::new(LibraryType::Llama, Platform::Linux)
        .at_path(lib_path.clone())
        .generate()
        .expect("Failed to generate mock library");

    // Verify library file exists
    let lib_file = lib_path.join("libllama.so");
    assert!(lib_file.exists(), "libllama.so should exist");

    // Verify ELF magic bytes
    let contents = std::fs::read(&lib_file).expect("Failed to read library file");
    assert_eq!(&contents[0..4], b"\x7fELF", "ELF magic bytes incorrect");

    // Verify ELF class (64-bit)
    assert_eq!(contents[4], 2, "ELF class should be 64-bit");

    // Verify ELF data (little-endian)
    assert_eq!(contents[5], 1, "ELF data should be little-endian");
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-18-macos-dylib-libraries
#[test]
#[cfg(target_os = "macos")]
fn test_mock_library_macos_macho_header() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let lib_path = temp_dir.path().to_path_buf();

    MockLibrary::new(LibraryType::Llama, Platform::MacOS)
        .at_path(lib_path.clone())
        .generate()
        .expect("Failed to generate mock library");

    // Verify library file exists
    let lib_file = lib_path.join("libllama.dylib");
    assert!(lib_file.exists(), "libllama.dylib should exist");

    // Verify Mach-O magic bytes
    let contents = std::fs::read(&lib_file).expect("Failed to read library file");
    let magic = u32::from_le_bytes([contents[0], contents[1], contents[2], contents[3]]);
    assert_eq!(magic, 0xFEEDFACF, "Mach-O magic bytes incorrect");
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-19-windows-lib-libraries
#[test]
#[cfg(target_os = "windows")]
fn test_mock_library_windows_pe_header() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let lib_path = temp_dir.path().to_path_buf();

    MockLibrary::new(LibraryType::Llama, Platform::Windows)
        .at_path(lib_path.clone())
        .generate()
        .expect("Failed to generate mock library");

    // Verify library file exists
    let lib_file = lib_path.join("llama.lib");
    assert!(lib_file.exists(), "llama.lib should exist");

    // Verify PE/DOS magic bytes
    let contents = std::fs::read(&lib_file).expect("Failed to read library file");
    assert_eq!(&contents[0..2], b"MZ", "PE/DOS magic bytes incorrect");
}

/// Tests feature spec: bitnet-integration-tests.md#mock-library-generator
#[test]
fn test_mock_library_all_types() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let lib_path = temp_dir.path().to_path_buf();

    let platform = if cfg!(target_os = "linux") {
        Platform::Linux
    } else if cfg!(target_os = "macos") {
        Platform::MacOS
    } else {
        Platform::Windows
    };

    // Generate all library types
    for lib_type in [LibraryType::BitNet, LibraryType::Llama, LibraryType::Ggml] {
        MockLibrary::new(lib_type, platform)
            .at_path(lib_path.clone())
            .generate()
            .expect("Failed to generate mock library");

        let filename = MockLibrary::library_filename(lib_type, platform);
        let lib_file = lib_path.join(&filename);
        assert!(lib_file.exists(), "Library file {} should exist", filename);
    }
}

// ============================================================================
// TestEnvironment Tests
// ============================================================================

/// Tests feature spec: bitnet-integration-tests.md#environment-isolation
#[test]
#[serial(bitnet_env)]
fn test_env_isolation_bitnet_cpp_dir() {
    let original = std::env::var("BITNET_CPP_DIR").ok();

    {
        let mut env = TestEnvironment::new().expect("Failed to create TestEnvironment");
        env.set_bitnet_cpp_dir(PathBuf::from("/tmp/test_bitnet"));

        assert_eq!(
            std::env::var("BITNET_CPP_DIR").unwrap(),
            "/tmp/test_bitnet",
            "BITNET_CPP_DIR not set correctly"
        );
    }

    // Verify restoration after drop
    assert_eq!(
        std::env::var("BITNET_CPP_DIR").ok(),
        original,
        "BITNET_CPP_DIR not restored after drop"
    );
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-13-crossval-libdir
#[test]
#[serial(bitnet_env)]
fn test_env_isolation_crossval_libdir() {
    let original = std::env::var("BITNET_CROSSVAL_LIBDIR").ok();

    {
        let mut env = TestEnvironment::new().expect("Failed to create TestEnvironment");
        env.set_crossval_libdir(PathBuf::from("/tmp/test_libdir"));

        assert_eq!(
            std::env::var("BITNET_CROSSVAL_LIBDIR").unwrap(),
            "/tmp/test_libdir",
            "BITNET_CROSSVAL_LIBDIR not set correctly"
        );
    }

    assert_eq!(
        std::env::var("BITNET_CROSSVAL_LIBDIR").ok(),
        original,
        "BITNET_CROSSVAL_LIBDIR not restored after drop"
    );
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-15-default-cache
#[test]
#[serial(bitnet_env)]
fn test_env_isolation_home_dir() {
    let original = std::env::var("HOME").ok();

    {
        let mut env = TestEnvironment::new().expect("Failed to create TestEnvironment");
        env.set_home(PathBuf::from("/tmp/test_home"));

        assert_eq!(std::env::var("HOME").unwrap(), "/tmp/test_home", "HOME not set correctly");
    }

    assert_eq!(std::env::var("HOME").ok(), original, "HOME not restored after drop");
}

/// Tests feature spec: bitnet-integration-tests.md#environment-isolation
#[test]
#[serial(bitnet_env)]
fn test_env_multiple_vars() {
    let original_cpp = std::env::var("BITNET_CPP_DIR").ok();
    let original_lib = std::env::var("BITNET_CROSSVAL_LIBDIR").ok();

    {
        let mut env = TestEnvironment::new().expect("Failed to create TestEnvironment");
        env.set_bitnet_cpp_dir(PathBuf::from("/tmp/cpp"));
        env.set_crossval_libdir(PathBuf::from("/tmp/lib"));

        assert_eq!(std::env::var("BITNET_CPP_DIR").unwrap(), "/tmp/cpp");
        assert_eq!(std::env::var("BITNET_CROSSVAL_LIBDIR").unwrap(), "/tmp/lib");
    }

    // Verify both restored
    assert_eq!(std::env::var("BITNET_CPP_DIR").ok(), original_cpp);
    assert_eq!(std::env::var("BITNET_CROSSVAL_LIBDIR").ok(), original_lib);
}

/// Tests feature spec: bitnet-integration-tests.md#environment-isolation
#[test]
fn test_temp_dir_exists_during_lifetime() {
    let env = TestEnvironment::new().expect("Failed to create TestEnvironment");
    let temp_path = env.temp_path().to_path_buf();

    // Directory should exist while env is alive
    assert!(temp_path.exists(), "Temp directory should exist during TestEnvironment lifetime");
}

/// Tests feature spec: bitnet-integration-tests.md#environment-isolation
#[test]
fn test_temp_dir_cleanup_after_drop() {
    let temp_path: PathBuf;

    {
        let env = TestEnvironment::new().expect("Failed to create TestEnvironment");
        temp_path = env.temp_path().to_path_buf();
        assert!(temp_path.exists());
    }

    // Directory should be cleaned up after drop
    assert!(!temp_path.exists(), "Temp directory should be cleaned up after drop");
}

/// Tests feature spec: bitnet-integration-tests.md#environment-isolation
#[test]
fn test_create_subdir() {
    let env = TestEnvironment::new().expect("Failed to create TestEnvironment");
    let subdir = env.create_subdir("test_subdir").expect("Failed to create subdirectory");

    assert!(subdir.exists(), "Subdirectory should exist");
    assert!(subdir.is_dir(), "Path should be a directory");
    assert_eq!(subdir.file_name().unwrap(), "test_subdir", "Subdirectory name incorrect");

    // Subdirectory should be inside temp directory
    assert!(subdir.starts_with(env.temp_path()));
}

// ============================================================================
// Integration Tests (Combining Components)
// ============================================================================

/// Tests feature spec: bitnet-integration-tests.md#test-execution-flow
#[test]
#[serial(bitnet_env)]
fn test_complete_fixture_workflow() {
    // Create isolated environment
    let mut env = TestEnvironment::new().expect("Failed to create TestEnvironment");

    // Create BitNet standard layout
    let layout = DirectoryLayoutBuilder::new(LayoutType::BitNetStandard)
        .with_libs(true)
        .with_headers(true)
        .build()
        .expect("Failed to create layout");

    // Set environment variables
    env.set_bitnet_cpp_dir(layout.root_path().to_path_buf());

    // Verify environment is set
    assert_eq!(std::env::var("BITNET_CPP_DIR").unwrap(), layout.root_path().display().to_string());

    // Verify layout structure
    assert!(layout.has_library(LibraryType::BitNet));
    assert!(layout.has_library(LibraryType::Llama));
    assert!(layout.has_header("ggml-bitnet.h"));
    assert!(layout.has_header("llama.h"));

    // Cleanup happens automatically via Drop
}

/// Tests feature spec: bitnet-integration-tests.md#test-execution-flow
#[test]
#[serial(bitnet_env)]
fn test_custom_libdir_override_workflow() {
    let mut env = TestEnvironment::new().expect("Failed to create TestEnvironment");

    // Create custom libdir layout
    let layout = DirectoryLayoutBuilder::new(LayoutType::CustomLibDir)
        .with_libs(true)
        .with_headers(true)
        .build()
        .expect("Failed to create layout");

    // Set BITNET_CROSSVAL_LIBDIR to custom_libs directory
    let custom_lib_path = layout.root_path().join("custom_libs");
    env.set_crossval_libdir(custom_lib_path.clone());

    // Verify override is set
    assert_eq!(
        std::env::var("BITNET_CROSSVAL_LIBDIR").unwrap(),
        custom_lib_path.display().to_string()
    );

    // Verify libraries exist in custom directory
    assert!(layout.has_library(LibraryType::BitNet));
    assert!(layout.has_library(LibraryType::Llama));
    assert!(layout.has_library(LibraryType::Ggml));
}

/// Tests feature spec: bitnet-integration-tests.md#scenario-5-missing-libs
#[test]
#[serial(bitnet_env)]
fn test_missing_libs_graceful_failure() {
    let mut env = TestEnvironment::new().expect("Failed to create TestEnvironment");

    // Create missing libs layout (headers only)
    let layout = DirectoryLayoutBuilder::new(LayoutType::MissingLibs)
        .with_libs(false)
        .with_headers(true)
        .build()
        .expect("Failed to create layout");

    env.set_bitnet_cpp_dir(layout.root_path().to_path_buf());

    // Verify headers exist
    assert!(layout.has_header("ggml-bitnet.h"));

    // Verify NO libraries exist
    assert!(!layout.has_library(LibraryType::BitNet));
    assert!(!layout.has_library(LibraryType::Llama));
    assert!(!layout.has_library(LibraryType::Ggml));

    // Verify build/ directory does not exist
    assert!(!layout.root_path().join("build").exists());
}

// ============================================================================
// Platform-Specific Library Name Tests
// ============================================================================

#[test]
fn test_library_filename_current_platform() {
    // Test that library filenames are correct for the current platform
    #[cfg(target_os = "linux")]
    {
        assert_eq!(
            MockLibrary::library_filename(LibraryType::BitNet, Platform::Linux),
            "libbitnet.so"
        );
        assert_eq!(
            MockLibrary::library_filename(LibraryType::Llama, Platform::Linux),
            "libllama.so"
        );
        assert_eq!(MockLibrary::library_filename(LibraryType::Ggml, Platform::Linux), "libggml.so");
    }

    #[cfg(target_os = "macos")]
    {
        assert_eq!(
            MockLibrary::library_filename(LibraryType::BitNet, Platform::MacOS),
            "libbitnet.dylib"
        );
        assert_eq!(
            MockLibrary::library_filename(LibraryType::Llama, Platform::MacOS),
            "libllama.dylib"
        );
        assert_eq!(
            MockLibrary::library_filename(LibraryType::Ggml, Platform::MacOS),
            "libggml.dylib"
        );
    }

    #[cfg(target_os = "windows")]
    {
        assert_eq!(
            MockLibrary::library_filename(LibraryType::BitNet, Platform::Windows),
            "bitnet.lib"
        );
        assert_eq!(
            MockLibrary::library_filename(LibraryType::Llama, Platform::Windows),
            "llama.lib"
        );
        assert_eq!(MockLibrary::library_filename(LibraryType::Ggml, Platform::Windows), "ggml.lib");
    }
}
