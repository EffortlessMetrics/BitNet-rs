#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::opencl_profiling::{KernelProfile, ProfilingSession};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ProfilingInput {
    /// Raw bytes to derive kernel names and timing values.
    entries: Vec<ProfileEntry>,
}

#[derive(Arbitrary, Debug)]
struct ProfileEntry {
    name_bytes: Vec<u8>,
    global_dims: Vec<u8>,
    local_dims: Vec<u8>,
    queued_ns: u64,
    submit_ns: u64,
    start_ns: u64,
    end_ns: u64,
    bytes_transferred: u16,
    flop_count: u32,
}

fuzz_target!(|input: ProfilingInput| {
    if input.entries.is_empty() || input.entries.len() > 128 {
        return;
    }

    let mut session = ProfilingSession::new();

    for entry in &input.entries {
        // Derive a kernel name from arbitrary bytes (valid UTF-8 subset).
        let name: String = entry
            .name_bytes
            .iter()
            .take(64)
            .map(|&b| {
                let c = b % 62;
                match c {
                    0..=9 => (b'0' + c) as char,
                    10..=35 => (b'a' + c - 10) as char,
                    36..=61 => (b'A' + c - 36) as char,
                    _ => '_',
                }
            })
            .collect();

        if name.is_empty() {
            continue;
        }

        let global_work_size: Vec<usize> =
            entry.global_dims.iter().take(3).map(|&b| (b as usize % 256) + 1).collect();
        let local_work_size: Vec<usize> =
            entry.local_dims.iter().take(3).map(|&b| (b as usize % 64) + 1).collect();

        let profile = KernelProfile {
            kernel_name: name.clone(),
            global_work_size,
            local_work_size,
            queued_ns: entry.queued_ns,
            submit_ns: entry.submit_ns,
            start_ns: entry.start_ns,
            end_ns: entry.end_ns,
        };

        // Derived timing metrics must not panic.
        let queue_lat = profile.queue_latency_us();
        let exec_time = profile.execution_time_us();
        let total_time = profile.total_time_us();
        let bw = profile.bandwidth_gb_s(entry.bytes_transferred as usize);
        let gf = profile.gflops(entry.flop_count as u64);

        // Invariant: all timing values are non-negative (saturating_sub).
        assert!(queue_lat >= 0.0, "queue_latency_us negative: {queue_lat}");
        assert!(exec_time >= 0.0, "execution_time_us negative: {exec_time}");
        assert!(total_time >= 0.0, "total_time_us negative: {total_time}");
        assert!(bw >= 0.0, "bandwidth negative: {bw}");
        assert!(gf >= 0.0, "gflops negative: {gf}");

        // Invariant: no NaN in derived metrics.
        assert!(!queue_lat.is_nan(), "queue_latency_us NaN");
        assert!(!exec_time.is_nan(), "execution_time_us NaN");
        assert!(!total_time.is_nan(), "total_time_us NaN");
        assert!(!bw.is_nan(), "bandwidth NaN");
        assert!(!gf.is_nan(), "gflops NaN");

        session.record(profile);
    }

    // Session operations must not panic.
    let _elapsed = session.elapsed();
    let _len = session.len();
    let _empty = session.is_empty();

    assert_eq!(session.is_empty(), session.len() == 0);

    // Query by kernel name must not panic.
    let _ = session.by_kernel("nonexistent_kernel_xyz");
    if let Some(entry) = input.entries.first() {
        let name: String = entry
            .name_bytes
            .iter()
            .take(64)
            .map(|&b| {
                let c = b % 62;
                match c {
                    0..=9 => (b'0' + c) as char,
                    10..=35 => (b'a' + c - 10) as char,
                    36..=61 => (b'A' + c - 36) as char,
                    _ => '_',
                }
            })
            .collect();
        let _ = session.by_kernel(&name);
    }

    // Slowest-N query must not panic.
    let _ = session.slowest(0);
    let _ = session.slowest(1);
    let _ = session.slowest(input.entries.len());
    let _ = session.slowest(input.entries.len() + 100);

    // GPU time must be non-negative.
    let gpu_time = session.total_gpu_time_ms();
    assert!(gpu_time >= 0.0, "total_gpu_time_ms negative: {gpu_time}");

    // Summary must not panic and must be internally consistent.
    if !session.is_empty() {
        let summary = session.summary();
        assert_eq!(summary.total_kernels, session.len());
        assert!(summary.total_gpu_time_ms >= 0.0);
        assert!(summary.avg_kernel_time_us >= 0.0);
    }
});
