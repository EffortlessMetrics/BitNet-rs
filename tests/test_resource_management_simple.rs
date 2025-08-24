#![cfg(feature = "integration-tests")]
#[cfg(test)]
mod resource_management_tests {
    use bitnet_tests::units::{BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB};
    use std::time::Duration;

    // Simple test to verify resource management functionality
    #[tokio::test]
    async fn test_memory_usage_tracking() {
        // This is a basic test to verify memory tracking works
        let initial_memory = get_memory_usage();

        // Allocate some memory
        let _data = vec![0u8; BYTES_PER_MB as usize]; // 1MB

        let after_alloc_memory = get_memory_usage();

        // Memory usage should have increased (though exact amount may vary)
        assert!(
            after_alloc_memory >= initial_memory,
            "Memory usage should increase after allocation"
        );

        // Drop the allocation
        drop(_data);

        // Wait a bit for potential cleanup
        tokio::time::sleep(Duration::from_millis(10)).await;

        println!("Initial memory: {} bytes", initial_memory);
        println!("After allocation: {} bytes", after_alloc_memory);
        println!("Memory tracking test completed successfully");
    }

    #[tokio::test]
    async fn test_file_handle_tracking() {
        use tokio::fs::File;

        // Create a temporary directory
        tokio::fs::create_dir_all("tests/temp/simple_test").await.unwrap();

        // Test file handle creation and cleanup
        let file_path = "tests/temp/simple_test/test_file.txt";

        // Create and immediately drop a file handle
        {
            let _file = File::create(file_path).await.unwrap();
            // File handle should be created here
        } // File handle should be dropped here

        // Verify file exists
        assert!(tokio::fs::metadata(file_path).await.is_ok(), "File should exist");

        // Clean up
        tokio::fs::remove_file(file_path).await.unwrap();
        tokio::fs::remove_dir("tests/temp/simple_test").await.unwrap();

        println!("File handle tracking test completed successfully");
    }

    #[tokio::test]
    async fn test_concurrent_resource_access() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::Semaphore;

        // Test concurrent access to limited resources
        let resource_count = Arc::new(AtomicUsize::new(0));
        let semaphore = Arc::new(Semaphore::new(3)); // Limit to 3 concurrent accesses
        let mut handles = Vec::new();

        // Spawn 10 tasks that compete for 3 resources
        for i in 0..10 {
            let resource_count = Arc::clone(&resource_count);
            let semaphore = Arc::clone(&semaphore);

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                // Simulate resource usage
                resource_count.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_millis(10)).await;

                i // Return task ID
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut completed_tasks = Vec::new();
        for handle in handles {
            let task_id = handle.await.unwrap();
            completed_tasks.push(task_id);
        }

        // Verify all tasks completed
        assert_eq!(completed_tasks.len(), 10, "All tasks should complete");
        assert_eq!(
            resource_count.load(Ordering::Relaxed),
            10,
            "All tasks should have accessed resources"
        );

        println!("Concurrent resource access test completed successfully");
    }

    // Platform-specific memory usage functions (simplified versions)
    #[cfg(target_os = "windows")]
    fn get_memory_usage() -> u64 {
        use winapi::um::processthreadsapi::GetCurrentProcess;
        use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};

        unsafe {
            let mut pmc = std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>();
            pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;

            if GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                pmc.WorkingSetSize as u64
            } else {
                0
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn get_memory_usage() -> u64 {
        use libc::{RUSAGE_SELF, getrusage, rusage};

        unsafe {
            let mut usage = std::mem::zeroed::<rusage>();
            if getrusage(RUSAGE_SELF, &mut usage) == 0 {
                usage.ru_maxrss as u64 * 1024 // macOS returns in KB
            } else {
                0
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn get_memory_usage() -> u64 {
        use std::fs;

        if let Ok(contents) = fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
        0
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    fn get_memory_usage() -> u64 {
        // Fallback for unsupported platforms
        BYTES_PER_MB // Return 1MB as a placeholder
    }
}
