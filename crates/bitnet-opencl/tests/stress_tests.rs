//! Comprehensive GPU stress test suite.
//!
//! All tests use `MockOpenClKernel` (CPU fallback) so they run without
//! real GPU hardware.  They are `#[ignore]` because they are slow.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use bitnet_common::QuantizationType;
use bitnet_kernels::KernelProvider;
use bitnet_opencl::stress_utils::{LoadGenerator, ResultCollector, StressTestRunner};
use bitnet_opencl::{MemoryBudget, MockOpenClKernel, OpenClConfig};

// ── helpers ──────────────────────────────────────────────────────────

/// Run a single matmul dispatch through the mock kernel.
fn dispatch_matmul(kernel: &dyn KernelProvider, size: usize) -> Result<(), String> {
    let a = vec![1_i8; size * size];
    let b = vec![1_u8; size * size];
    let mut c = vec![0_f32; size * size];
    kernel.matmul_i2s(&a, &b, &mut c, size, size, size).map_err(|e| format!("{e}"))
}

/// Run a single quantize dispatch.
fn _dispatch_quantize(kernel: &dyn KernelProvider, len: usize) -> Result<(), String> {
    let input = vec![0.5_f32; len];
    let mut output = vec![0_u8; len];
    let mut scales = vec![0_f32; len];
    kernel
        .quantize(&input, &mut output, &mut scales, QuantizationType::I2S)
        .map_err(|e| format!("{e}"))
}

// ── (a) Concurrent kernel dispatch ───────────────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_concurrent_kernel_dispatch() {
    let kernel = Arc::new(MockOpenClKernel::new());
    let threads = 12;
    let iters = 50;
    let barrier = Arc::new(Barrier::new(threads));

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let k = Arc::clone(&kernel);
            let b = Arc::clone(&barrier);
            std::thread::spawn(move || {
                b.wait(); // ensure simultaneous start
                for _ in 0..iters {
                    dispatch_matmul(k.as_ref(), 8).expect("matmul failed");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    assert_eq!(kernel.dispatch_count(), threads * iters);
}

// ── (b) Memory pressure ─────────────────────────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_memory_pressure_graceful_degradation() {
    // Budget: only 4 KiB — a 64×64 matmul needs 64*64*4 = 16 KiB
    let budget = MemoryBudget::new(4 * 1024);
    let kernel = MockOpenClKernel::new().with_memory_budget(budget.clone());

    // Small op should succeed
    dispatch_matmul(&kernel, 4).expect("small matmul should fit");

    // Large op should fail gracefully (not panic)
    let result = dispatch_matmul(&kernel, 64);
    assert!(result.is_err(), "should exceed budget");
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("memory budget exceeded"), "unexpected error: {err_msg}",);

    // Budget should still be usable after failure
    assert_eq!(budget.used(), 0, "budget should be fully freed");
    dispatch_matmul(&kernel, 4).expect("should still work after OOM");
}

// ── (c) Rapid alloc/free cycles ─────────────────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_rapid_alloc_free_cycles() {
    let budget = MemoryBudget::new(1024 * 1024); // 1 MiB
    let cycles = 10_000;

    for _ in 0..cycles {
        let alloc_size = 256_u64;
        assert!(budget.try_alloc(alloc_size), "alloc should succeed");
        budget.free(alloc_size);
    }

    assert_eq!(budget.used(), 0, "leak detected after {cycles} cycles");
    assert_eq!(budget.available(), 1024 * 1024);
}

// ── (d) Long-running generation ─────────────────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_long_running_generation_no_drift() {
    let kernel = MockOpenClKernel::new();
    let tokens = 1_000;
    let size = 4; // small for speed

    // Use deterministic input so we can compare outputs.
    let a = vec![1_i8; size * size];
    let b = vec![1_u8; size * size];

    let mut reference: Option<Vec<f32>> = None;
    let mut drift_detected = false;

    for token_idx in 0..tokens {
        let mut c = vec![0_f32; size * size];
        kernel
            .matmul_i2s(&a, &b, &mut c, size, size, size)
            .unwrap_or_else(|e| panic!("token {token_idx}: {e}"));

        match &reference {
            None => reference = Some(c),
            Some(r) => {
                if r != &c {
                    drift_detected = true;
                    break;
                }
            }
        }
    }

    assert!(!drift_detected, "numerical drift detected over {tokens} tokens");
    assert_eq!(kernel.dispatch_count(), tokens);
}

// ── (e) Multiple models loaded simultaneously ───────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_multiple_models_round_robin() {
    // Simulate 3 "models" with different kernels.
    let models: Vec<Arc<MockOpenClKernel>> =
        (0..3).map(|_| Arc::new(MockOpenClKernel::new())).collect();

    let queries = 90;
    let collector = ResultCollector::new();

    for q in 0..queries {
        let model = &models[q % 3];
        let start = Instant::now();
        match dispatch_matmul(model.as_ref(), 8) {
            Ok(()) => collector.record_pass(start.elapsed()),
            Err(e) => collector.record_fail(&e),
        }
    }

    let report = collector.report();
    assert_eq!(report.passed, queries as u64);

    // Each model should have handled 30 queries.
    for (i, m) in models.iter().enumerate() {
        assert_eq!(m.dispatch_count(), 30, "model {i} dispatch count mismatch");
    }
}

// ── (f) Error recovery under load ───────────────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_error_recovery_under_load() {
    let kernel = Arc::new(MockOpenClKernel::new());
    let collector = ResultCollector::new();
    let total = 200;

    for i in 0..total {
        // Inject errors for iterations 50..100
        kernel.set_error_injection(i >= 50 && i < 100);

        let start = Instant::now();
        match dispatch_matmul(kernel.as_ref(), 8) {
            Ok(()) => collector.record_pass(start.elapsed()),
            Err(e) => collector.record_fail(&e),
        }
    }

    let report = collector.report();
    assert_eq!(report.failed, 50, "expected exactly 50 injected errors");
    assert_eq!(report.passed, 150, "remaining should succeed");

    // Kernel still works after error injection is disabled.
    kernel.set_error_injection(false);
    dispatch_matmul(kernel.as_ref(), 8).expect("should recover");
}

// ── (g) Config hot reload during inference ──────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_config_hot_reload_during_inference() {
    let config = Arc::new(Mutex::new(OpenClConfig::default()));
    let kernel = Arc::new(MockOpenClKernel::new());
    let done = Arc::new(AtomicBool::new(false));
    let collector = ResultCollector::new();

    // Worker thread: continuously dispatch while config changes.
    let worker = {
        let k = Arc::clone(&kernel);
        let c = collector.clone();
        let d = Arc::clone(&done);
        let cfg = Arc::clone(&config);
        std::thread::spawn(move || {
            let mut iterations = 0_u64;
            while !d.load(Ordering::Acquire) {
                // Read config snapshot.
                let _batch = cfg.lock().expect("poisoned").batch_size;
                let start = Instant::now();
                match dispatch_matmul(k.as_ref(), 8) {
                    Ok(()) => c.record_pass(start.elapsed()),
                    Err(e) => c.record_fail(&e),
                }
                iterations += 1;
                if iterations > 500 {
                    break; // safety cap
                }
            }
        })
    };

    // Main thread: mutate config repeatedly.
    for batch in [2, 4, 8, 16, 1] {
        std::thread::sleep(Duration::from_millis(5));
        config.lock().expect("poisoned").batch_size = batch;
    }
    done.store(true, Ordering::Release);

    worker.join().expect("worker panicked");

    let report = collector.report();
    assert_eq!(report.failed, 0, "no failures expected during hot reload");
    assert!(report.passed > 0, "should have completed some work");
}

// ── (h) Queue overflow ──────────────────────────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_queue_overflow() {
    // Simulate a bounded queue: capacity = 16, but we batch-insert 3 per
    // iteration and only drain 1, so the queue fills up.
    let capacity = 16_usize;
    let queue = Arc::new(Mutex::new(Vec::<u64>::new()));
    let rejected = Arc::new(AtomicUsize::new(0));
    let submitted = 100_u64;

    let kernel = MockOpenClKernel::new();
    let collector = ResultCollector::new();

    for i in 0..submitted {
        // Try to enqueue 3 items per iteration (simulating burst).
        let mut q = queue.lock().expect("poisoned");
        let mut added = 0_usize;
        for slot in 0..3_u64 {
            if q.len() >= capacity {
                rejected.fetch_add(1, Ordering::Relaxed);
                collector.record_fail("queue full");
            } else {
                q.push(i * 3 + slot);
                added += 1;
            }
        }
        drop(q);

        // Process only one item per iteration.
        if added > 0 {
            let start = Instant::now();
            match dispatch_matmul(&kernel, 4) {
                Ok(()) => {
                    collector.record_pass(start.elapsed());
                    queue.lock().expect("poisoned").pop();
                }
                Err(e) => collector.record_fail(&e),
            }
        }
    }

    let report = collector.report();
    let rejects = rejected.load(Ordering::Acquire);
    assert!(rejects > 0, "queue should have overflowed at least once");
    assert!(report.passed > 0, "some work should have completed");
}

// ── (i) Interleaved batch/single requests ───────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_interleaved_batch_single_requests() {
    let kernel = Arc::new(MockOpenClKernel::new());
    let load_gen = LoadGenerator::new(32, 32).with_batch_probability(0.4);

    let runner = StressTestRunner::new(4, 100);
    let k = Arc::clone(&kernel);
    let g = Arc::new(load_gen);

    let report = runner.run(move |_tid, _i| {
        let req = g.next_request();
        if req.is_batch {
            // Batch: multiple dispatches.
            for _ in 0..4 {
                dispatch_matmul(k.as_ref(), 4)?;
            }
        } else {
            dispatch_matmul(k.as_ref(), 4)?;
        }
        Ok(())
    });

    assert_eq!(report.failed, 0);
    assert_eq!(report.total_requests, 400);
    // Kernel should have more than 400 dispatches (batches add extra).
    assert!(
        kernel.dispatch_count() >= 400,
        "dispatch count {} should be >= 400",
        kernel.dispatch_count()
    );
}

// ── (j) Backend switching under load ────────────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_backend_switching_under_load() {
    let gpu_kernel = Arc::new(MockOpenClKernel::new());
    let cpu_kernel = Arc::new(MockOpenClKernel::new());
    // Signal: false = GPU, true = CPU
    let use_cpu = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let collector = ResultCollector::new();

    // Worker threads.
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let gpu = Arc::clone(&gpu_kernel);
            let cpu = Arc::clone(&cpu_kernel);
            let flag = Arc::clone(&use_cpu);
            let d = Arc::clone(&done);
            let c = collector.clone();
            std::thread::spawn(move || {
                while !d.load(Ordering::Acquire) {
                    let kernel: &dyn KernelProvider =
                        if flag.load(Ordering::Acquire) { cpu.as_ref() } else { gpu.as_ref() };
                    let start = Instant::now();
                    match dispatch_matmul(kernel, 4) {
                        Ok(()) => c.record_pass(start.elapsed()),
                        Err(e) => c.record_fail(&e),
                    }
                }
            })
        })
        .collect();

    // Switch backends several times.
    for _ in 0..10 {
        std::thread::sleep(Duration::from_millis(5));
        use_cpu.fetch_xor(true, Ordering::Release);
    }

    done.store(true, Ordering::Release);
    for h in workers {
        h.join().expect("worker panicked");
    }

    let report = collector.report();
    assert_eq!(report.failed, 0, "switching should not cause errors");
    assert!(report.passed > 0, "should have completed work");

    let gpu_dispatches = gpu_kernel.dispatch_count();
    let cpu_dispatches = cpu_kernel.dispatch_count();
    assert!(
        gpu_dispatches > 0 && cpu_dispatches > 0,
        "both backends should have been used: gpu={gpu_dispatches} cpu={cpu_dispatches}"
    );
}

// ── (bonus) StressTestRunner integration ────────────────────────────

#[test]
#[ignore = "stress test — run with --ignored"]
fn stress_runner_with_mixed_results() {
    let runner = StressTestRunner::new(4, 50);
    let report = runner.run(|_tid, i| {
        if i % 10 == 0 {
            Err("periodic failure".into())
        } else {
            std::thread::sleep(Duration::from_micros(10));
            Ok(())
        }
    });

    // 4 threads × 50 iters = 200 total, 4×5 = 20 failures
    assert_eq!(report.total_requests, 200);
    assert_eq!(report.failed, 20);
    assert_eq!(report.passed, 180);
    assert!(!report.error_breakdown.is_empty());
    assert!(report.latency_p50 > Duration::ZERO);
}
