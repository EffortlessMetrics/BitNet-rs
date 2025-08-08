use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Get current memory usage in bytes
pub fn get_memory_usage() -> u64 {
    // Platform-specific memory usage detection
    #[cfg(target_os = "linux")]
    {
        get_memory_usage_linux()
    }
    #[cfg(target_os = "macos")]
    {
        get_memory_usage_macos()
    }
    #[cfg(target_os = "windows")]
    {
        get_memory_usage_windows()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        0 // Fallback for unsupported platforms
    }
}

/// Get peak memory usage since last reset
pub fn get_peak_memory_usage() -> u64 {
    PEAK_MEMORY.load(Ordering::Relaxed)
}

/// Reset peak memory tracking
pub fn reset_peak_memory() {
    PEAK_MEMORY.store(0, Ordering::Relaxed);
}

/// Update peak memory if current usage is higher
pub fn update_peak_memory() {
    let current = get_memory_usage();
    let mut peak = PEAK_MEMORY.load(Ordering::Relaxed);

    while current > peak {
        match PEAK_MEMORY.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(new_peak) => peak = new_peak,
        }
    }
}

/// Measure execution time of a function
pub async fn measure_time<F, Fut, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let start = Instant::now();
    let result = f().await;
    let duration = start.elapsed();
    (result, duration)
}

/// Measure execution time and memory usage of a function
pub async fn measure_time_and_memory<F, Fut, T>(f: F) -> (T, Duration, u64, u64)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    reset_peak_memory();
    let start_memory = get_memory_usage();
    let start_time = Instant::now();

    let result = f().await;

    let duration = start_time.elapsed();
    let end_memory = get_memory_usage();
    let peak_memory = get_peak_memory_usage();

    (result, duration, end_memory - start_memory, peak_memory)
}

/// Memory tracker for continuous monitoring
pub struct MemoryTracker {
    peak: Arc<AtomicU64>,
    start: u64,
    monitoring: Arc<AtomicU64>, // 0 = stopped, 1 = running
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            peak: Arc::new(AtomicU64::new(0)),
            start: get_memory_usage(),
            monitoring: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start monitoring memory usage
    pub fn start(&self) {
        self.monitoring.store(1, Ordering::Relaxed);
        let peak = Arc::clone(&self.peak);
        let monitoring = Arc::clone(&self.monitoring);

        tokio::spawn(async move {
            while monitoring.load(Ordering::Relaxed) == 1 {
                let current = get_memory_usage();
                let mut current_peak = peak.load(Ordering::Relaxed);

                while current > current_peak {
                    match peak.compare_exchange_weak(
                        current_peak,
                        current,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(new_peak) => current_peak = new_peak,
                    }
                }

                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
    }

    /// Stop monitoring and get results
    pub fn stop(&self) -> (u64, u64) {
        self.monitoring.store(0, Ordering::Relaxed);
        let current = get_memory_usage();
        let peak = self.peak.load(Ordering::Relaxed);
        (current - self.start, peak)
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Format bytes in human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: u64 = 1024;

    if bytes < THRESHOLD {
        return format!("{} B", bytes);
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= THRESHOLD as f64 && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD as f64;
        unit_index += 1;
    }

    format!("{:.1} {}", size, UNITS[unit_index])
}

/// Format duration in human-readable format
pub fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let nanos = duration.subsec_nanos();

    if total_secs >= 3600 {
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if total_secs >= 60 {
        let minutes = total_secs / 60;
        let seconds = total_secs % 60;
        format!("{}m {}s", minutes, seconds)
    } else if total_secs > 0 {
        format!("{}.{:03}s", total_secs, nanos / 1_000_000)
    } else if nanos >= 1_000_000 {
        format!("{:.1}ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.1}μs", nanos as f64 / 1_000.0)
    } else {
        format!("{}ns", nanos)
    }
}

/// Generate a unique test ID
pub fn generate_test_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    format!("test_{}_{}", timestamp, id)
}

/// Check if running in CI environment
pub fn is_ci() -> bool {
    std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
        || std::env::var("GITLAB_CI").is_ok()
        || std::env::var("JENKINS_URL").is_ok()
}

/// Get number of CPU cores
pub fn get_cpu_cores() -> usize {
    num_cpus::get()
}

/// Get optimal number of parallel tests based on system resources
pub fn get_optimal_parallel_tests() -> usize {
    let cores = get_cpu_cores();
    if is_ci() {
        // Be more conservative in CI environments
        (cores / 2).max(1)
    } else {
        // Use more cores locally for faster feedback
        cores.max(1)
    }
}

// Global peak memory tracking
static PEAK_MEMORY: AtomicU64 = AtomicU64::new(0);

// Platform-specific memory usage implementations
#[cfg(target_os = "linux")]
fn get_memory_usage_linux() -> u64 {
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

#[cfg(target_os = "macos")]
fn get_memory_usage_macos() -> u64 {
    use std::mem;

    unsafe {
        let mut info: libc::mach_task_basic_info = mem::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;

        let result = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut _,
            &mut count,
        );

        if result == libc::KERN_SUCCESS {
            info.resident_size
        } else {
            0
        }
    }
}

#[cfg(target_os = "windows")]
fn get_memory_usage_windows() -> u64 {
    use std::mem;
    use winapi::um::processthreadsapi::GetCurrentProcess;
    use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};

    unsafe {
        let mut pmc: PROCESS_MEMORY_COUNTERS = mem::zeroed();
        let result = GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut pmc,
            mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        );

        if result != 0 {
            pmc.WorkingSetSize as u64
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_nanos(500)), "500ns");
        assert_eq!(format_duration(Duration::from_micros(1500)), "1.5μs");
        assert_eq!(format_duration(Duration::from_millis(1500)), "1.5ms");
        assert_eq!(format_duration(Duration::from_secs(1)), "1.000s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format_duration(Duration::from_secs(3665)), "1h 1m 5s");
    }

    #[tokio::test]
    async fn test_measure_time() {
        let (result, duration) = measure_time(|| async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            42
        })
        .await;

        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
        assert!(duration < Duration::from_millis(50)); // Allow some variance
    }

    #[test]
    fn test_generate_test_id() {
        let id1 = generate_test_id();
        let id2 = generate_test_id();

        assert_ne!(id1, id2);
        assert!(id1.starts_with("test_"));
        assert!(id2.starts_with("test_"));
    }

    #[test]
    fn test_get_cpu_cores() {
        let cores = get_cpu_cores();
        assert!(cores > 0);
        assert!(cores <= 256); // Reasonable upper bound
    }

    #[test]
    fn test_get_optimal_parallel_tests() {
        let optimal = get_optimal_parallel_tests();
        assert!(optimal > 0);
        assert!(optimal <= get_cpu_cores());
    }
}
