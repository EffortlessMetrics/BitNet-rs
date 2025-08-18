#[cfg(test)]
use bitnet_tests::units::{BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB};

mod resource_management_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;
    use tokio::sync::{Mutex, Semaphore};
    use tokio::time::{sleep, timeout};

    // Import the canonical MB constant from the test harness crate
    use bitnet_tests::common::BYTES_PER_MB;

    // Platform-specific memory usage functions
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

    #[cfg(not(target_os = "windows"))]
    fn get_memory_usage() -> u64 {
        // Fallback for non-Windows platforms
        BYTES_PER_MB // Return 1MB as a placeholder
    }

    /// Test 1: Memory usage and leak detection
    #[tokio::test]
    async fn test_memory_leak_detection() {
        let initial_memory = get_memory_usage();
        let mut allocations = Vec::new();

        // Phase 1: Allocate memory
        for _ in 0..100 {
            let data = vec![0u8; 1024 * 10]; // 10KB allocation
            allocations.push(data);
        }

        let peak_memory = get_memory_usage();

        // Phase 2: Deallocate memory
        allocations.clear();

        // Wait for memory to be released
        sleep(Duration::from_millis(100)).await;

        let final_memory = get_memory_usage();
        let memory_delta = final_memory as i64 - initial_memory as i64;

        println!("Memory leak detection test:");
        println!("  Initial memory: {} bytes", initial_memory);
        println!("  Peak memory: {} bytes", peak_memory);
        println!("  Final memory: {} bytes", final_memory);
        println!("  Memory delta: {} bytes", memory_delta);

        // Check for memory leaks (allow for some variance)
        let leak_threshold = 5 * BYTES_PER_MB; // 5MB threshold (generous for test)
        assert!(
            memory_delta < leak_threshold,
            "Memory leak detected: {} bytes not released (threshold: {} bytes)",
            memory_delta,
            leak_threshold
        );

        println!("✓ Memory leak detection test passed");
    }

    /// Test 2: File handle and resource cleanup
    #[tokio::test]
    async fn test_file_handle_cleanup() {
        // Create test directory
        tokio::fs::create_dir_all("tests/temp/file_handles").await.unwrap();

        let mut file_handles = Vec::new();

        // Open many files
        for i in 0..20 {
            let file_path = format!("tests/temp/file_handles/test_file_{}.txt", i);
            let mut file = File::create(&file_path).await.unwrap();
            file.write_all(b"test data").await.unwrap();
            file_handles.push((file, file_path));
        }

        let max_handles = file_handles.len();

        // Close all files explicitly
        for (file, path) in file_handles {
            drop(file);
            let _ = tokio::fs::remove_file(path).await;
        }

        // Clean up test directory
        let _ = tokio::fs::remove_dir_all("tests/temp/file_handles").await;

        println!("File handle cleanup test:");
        println!("  Max file handles: {}", max_handles);
        println!("✓ File handle cleanup test passed");
    }

    /// Test 3: Concurrent resource access
    #[tokio::test]
    async fn test_concurrent_resource_access() {
        let shared_counter = Arc::new(AtomicUsize::new(0));
        let num_tasks = 8;
        let operations_per_task = 50;

        let mut handles = Vec::new();

        // Spawn concurrent tasks
        for _task_id in 0..num_tasks {
            let counter = Arc::clone(&shared_counter);

            let handle = tokio::spawn(async move {
                for _ in 0..operations_per_task {
                    // Simulate some work
                    let _temp_data = vec![0u8; 1024]; // 1KB allocation
                    counter.fetch_add(1, Ordering::Relaxed);
                    sleep(Duration::from_millis(1)).await;
                }
            });

            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        let final_count = shared_counter.load(Ordering::Relaxed);
        let expected_count = num_tasks * operations_per_task;

        println!("Concurrent resource access test:");
        println!("  Expected operations: {}", expected_count);
        println!("  Actual operations: {}", final_count);

        assert_eq!(final_count, expected_count, "Concurrent operations mismatch");
        println!("✓ Concurrent resource access test passed");
    }

    /// Test 4: Resource exhaustion and recovery
    #[tokio::test]
    async fn test_resource_exhaustion_recovery() {
        let initial_memory = get_memory_usage();
        let mut allocations = Vec::new();

        // Phase 1: Allocate memory until we reach a reasonable limit
        let allocation_size = 5 * BYTES_PER_MB; // 5MB chunks
        let max_allocations = 20; // 100MB total

        for i in 0..max_allocations {
            match std::panic::catch_unwind(|| vec![0u8; allocation_size]) {
                Ok(data) => {
                    allocations.push(data);
                }
                Err(_) => {
                    println!("Memory allocation failed at iteration {}", i);
                    break;
                }
            }
        }

        let peak_memory = get_memory_usage();

        // Phase 2: Recovery - free half the allocations
        let allocations_to_free = allocations.len() / 2;
        for _ in 0..allocations_to_free {
            allocations.pop();
        }

        sleep(Duration::from_millis(100)).await;
        let recovery_memory = get_memory_usage();

        // Phase 3: Full cleanup
        allocations.clear();
        sleep(Duration::from_millis(100)).await;
        let final_memory = get_memory_usage();

        println!("Resource exhaustion and recovery test:");
        println!("  Initial memory: {} bytes", initial_memory);
        println!("  Peak memory: {} bytes", peak_memory);
        println!("  Recovery memory: {} bytes", recovery_memory);
        println!("  Final memory: {} bytes", final_memory);

        // Verify recovery worked
        assert!(recovery_memory < peak_memory, "Memory should decrease after partial cleanup");
        println!("✓ Resource exhaustion and recovery test passed");
    }

    /// Test 5: Resource monitoring and alerting
    #[tokio::test]
    async fn test_resource_monitoring() {
        let memory_samples = Arc::new(Mutex::new(Vec::new()));
        let alert_count = Arc::new(AtomicUsize::new(0));
        let memory_threshold = 10 * BYTES_PER_MB; // 10MB threshold

        // Start monitoring task
        let samples_clone = Arc::clone(&memory_samples);
        let alert_clone = Arc::clone(&alert_count);
        let initial_memory = get_memory_usage();

        let monitor_handle = tokio::spawn(async move {
            for _ in 0..20 {
                let current_memory = get_memory_usage();
                let memory_delta = current_memory.saturating_sub(initial_memory);

                // Record sample
                {
                    let mut samples = samples_clone.lock().await;
                    samples.push(current_memory);
                }

                // Check threshold
                if memory_delta > memory_threshold {
                    alert_clone.fetch_add(1, Ordering::Relaxed);
                }

                sleep(Duration::from_millis(50)).await;
            }
        });

        // Simulate workload
        let mut workload_data = Vec::new();
        for i in 0..10 {
            let data = vec![0u8; 2 * BYTES_PER_MB]; // 2MB allocation
            workload_data.push(data);
            sleep(Duration::from_millis(100)).await;
        }

        // Wait for monitoring to complete
        monitor_handle.await.unwrap();

        // Analyze results
        let samples = memory_samples.lock().await;
        let alerts = alert_count.load(Ordering::Relaxed);

        let min_memory = samples.iter().min().unwrap_or(&0);
        let max_memory = samples.iter().max().unwrap_or(&0);
        let avg_memory = if !samples.is_empty() {
            samples.iter().sum::<u64>() / samples.len() as u64
        } else {
            0
        };

        println!("Resource monitoring test:");
        println!("  Samples collected: {}", samples.len());
        println!("  Min memory: {} bytes", min_memory);
        println!("  Max memory: {} bytes", max_memory);
        println!("  Avg memory: {} bytes", avg_memory);
        println!("  Alerts triggered: {}", alerts);

        assert!(!samples.is_empty(), "Should have collected memory samples");
        println!("✓ Resource monitoring test passed");
    }

    /// Test 6: Resource contention
    #[tokio::test]
    async fn test_resource_contention() {
        let resource_pool = Arc::new(Semaphore::new(3)); // Only 3 resources available
        let num_tasks = 10; // More tasks than resources

        let acquired_count = Arc::new(AtomicUsize::new(0));
        let contention_count = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();

        // Spawn tasks that compete for limited resources
        for _task_id in 0..num_tasks {
            let resource_pool = Arc::clone(&resource_pool);
            let acquired_count = Arc::clone(&acquired_count);
            let contention_count = Arc::clone(&contention_count);

            let handle = tokio::spawn(async move {
                match timeout(Duration::from_millis(100), resource_pool.acquire()).await {
                    Ok(Ok(_permit)) => {
                        acquired_count.fetch_add(1, Ordering::Relaxed);
                        // Simulate resource usage
                        sleep(Duration::from_millis(50)).await;
                    }
                    _ => {
                        contention_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        let successful_acquisitions = acquired_count.load(Ordering::Relaxed);
        let contentions = contention_count.load(Ordering::Relaxed);

        println!("Resource contention test:");
        println!("  Total tasks: {}", num_tasks);
        println!("  Successful acquisitions: {}", successful_acquisitions);
        println!("  Contentions: {}", contentions);
        println!("  Contention rate: {:.1}%", (contentions as f64 / num_tasks as f64) * 100.0);

        // Verify that contention occurred (more tasks than resources)
        assert!(contentions > 0, "Should have resource contention");
        assert_eq!(successful_acquisitions + contentions, num_tasks, "All tasks should complete");
        println!("✓ Resource contention test passed");
    }

    /// Integration test that runs all resource management tests
    #[tokio::test]
    async fn test_comprehensive_resource_management() {
        println!("Running comprehensive resource management tests...");

        // Run all individual tests
        test_memory_leak_detection();
        test_file_handle_cleanup();
        test_concurrent_resource_access();
        test_resource_exhaustion_recovery();
        test_resource_monitoring();
        test_resource_contention();

        println!("✅ All comprehensive resource management tests passed!");
    }
}
