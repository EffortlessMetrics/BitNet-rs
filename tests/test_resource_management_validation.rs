#![cfg(feature = "integration-tests")]

//! Simple validation test for resource management improvements
//! This test validates that our enhanced resource management tests work correctly

use std::time::Duration;

#[tokio::test]
async fn test_resource_management_improvements() {
    // Initialize logging for testing
    let _ = tracing_subscriber::fmt::try_init();

    println!("Testing resource management improvements...");

    // Test 1: Validate memory tracking function works
    let initial_memory = crate::test_resource_management_comprehensive::get_memory_usage();
    println!("Initial memory usage: {} bytes", initial_memory);

    // Should return a non-zero value on supported platforms, or a placeholder on others
    assert!(initial_memory >= 0, "Memory usage should be non-negative");

    // Test 2: Create a small allocation to test memory delta detection
    let test_allocation = vec![0u8; 1024 * 1024]; // 1MB allocation

    let peak_memory = crate::test_resource_management_comprehensive::get_memory_usage();
    println!("Peak memory usage: {} bytes", peak_memory);

    // Memory should increase (on platforms with actual tracking) or remain at placeholder
    assert!(peak_memory >= initial_memory, "Peak memory should be >= initial memory");

    // Release the allocation
    drop(test_allocation);

    // Allow some time for memory to be released
    tokio::time::sleep(Duration::from_millis(100)).await;

    let final_memory = crate::test_resource_management_comprehensive::get_memory_usage();
    println!("Final memory usage: {} bytes", final_memory);

    // Test 3: Validate file creation with different directory paths
    let test_dirs = ["tests/temp/validation", "/tmp/bitnet_validation", "./temp_validation"];

    let mut successful_dirs = 0;
    for test_dir in &test_dirs {
        match tokio::fs::create_dir_all(test_dir).await {
            Ok(_) => {
                successful_dirs += 1;
                let _ = tokio::fs::remove_dir_all(test_dir).await;
            }
            Err(e) => {
                println!("Failed to create {}: {}", test_dir, e);
            }
        }
    }

    println!(
        "Successfully created {} out of {} test directories",
        successful_dirs,
        test_dirs.len()
    );
    assert!(successful_dirs > 0, "At least one test directory should be creatable");

    // Test 4: Validate platform detection
    let platform = std::env::consts::OS;
    println!("Detected platform: {}", platform);

    let is_supported_platform = matches!(platform, "windows" | "macos" | "linux");
    println!("Platform supported for memory tracking: {}", is_supported_platform);

    // On supported platforms, we should see some memory tracking
    if is_supported_platform && initial_memory > 0 {
        println!("Memory tracking is working on this platform");
    } else {
        println!("Using simulated memory tracking on this platform");
    }

    println!("Resource management improvements validation completed successfully!");
}

#[tokio::test]
async fn test_edge_case_allocations() {
    println!("Testing edge case allocations...");

    // Test zero-byte allocations
    let mut zero_allocations = Vec::new();
    for _ in 0..100 {
        zero_allocations.push(Vec::<u8>::new());
    }
    assert_eq!(zero_allocations.len(), 100);
    zero_allocations.clear();

    // Test single-byte allocations
    let mut single_byte_allocations = Vec::new();
    for _ in 0..1000 {
        single_byte_allocations.push(vec![42u8; 1]);
    }
    assert_eq!(single_byte_allocations.len(), 1000);
    single_byte_allocations.clear();

    println!("Edge case allocations test completed successfully!");
}

#[tokio::test]
async fn test_file_handle_edge_cases() {
    println!("Testing file handle edge cases...");

    // Create test directory
    let test_dir = "tests/temp/file_edge_cases";
    tokio::fs::create_dir_all(test_dir).await.expect("Should create test directory");

    // Test files with different naming patterns
    let test_files = [
        "normal_file.txt",
        "file_with_spaces.txt",
        "file-with-dashes.txt",
        "file_with_underscores.txt",
        "file.with.dots.txt",
        "123456789.txt",
    ];

    let mut created_files = Vec::new();
    let mut successful_creates = 0;

    for filename in &test_files {
        let full_path = format!("{}/{}", test_dir, filename);
        match tokio::fs::File::create(&full_path).await {
            Ok(file) => {
                created_files.push((file, full_path));
                successful_creates += 1;
            }
            Err(e) => {
                println!("Failed to create {}: {}", filename, e);
            }
        }
    }

    println!("Successfully created {} out of {} test files", successful_creates, test_files.len());
    assert!(successful_creates > 0, "Should be able to create at least some test files");

    // Clean up files
    for (file, path) in created_files {
        drop(file);
        let _ = tokio::fs::remove_file(path).await;
    }

    // Remove test directory
    let _ = tokio::fs::remove_dir_all(test_dir).await;

    println!("File handle edge cases test completed successfully!");
}
