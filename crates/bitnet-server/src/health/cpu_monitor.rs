//! CPU health monitoring for BitNet.rs server
//!
//! Provides CPU metrics collection including:
//! - CPU utilization
//! - Core/thread counts
//! - SIMD capabilities detection
//! - Memory usage tracking

use serde::{Deserialize, Serialize};

/// CPU information for health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub cores: usize,
    pub threads: usize,
    pub utilization: f64,
    pub simd_capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// Memory information for health checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHealthInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub utilization: f64,
}

/// Collect CPU information for health monitoring
pub async fn collect_cpu_info() -> Result<CpuInfo, String> {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_cpu_all();

    // Small sleep to get accurate CPU measurement
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    sys.refresh_cpu_all();

    let cores = num_cpus::get_physical();
    let threads = num_cpus::get();
    let utilization = sys.global_cpu_usage() as f64 / 100.0; // Normalize to 0.0-1.0

    let simd_capabilities = detect_simd_capabilities();

    Ok(CpuInfo {
        cores,
        threads,
        utilization,
        simd_capabilities,
        temperature: None, // TODO: Implement temperature monitoring if available
    })
}

/// Detect available SIMD capabilities on the current CPU
fn detect_simd_capabilities() -> Vec<String> {
    let mut caps = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            caps.push("SSE".to_string());
        }
        if is_x86_feature_detected!("sse2") {
            caps.push("SSE2".to_string());
        }
        if is_x86_feature_detected!("avx") {
            caps.push("AVX".to_string());
        }
        if is_x86_feature_detected!("avx2") {
            caps.push("AVX2".to_string());
        }
        if is_x86_feature_detected!("avx512f") {
            caps.push("AVX512F".to_string());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            caps.push("NEON".to_string());
        }
    }

    caps
}

/// Collect memory health information
pub async fn collect_memory_health_info() -> Result<MemoryHealthInfo, String> {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();

    let total_bytes = sys.total_memory();
    let used_bytes = sys.used_memory();
    let available_bytes = total_bytes.saturating_sub(used_bytes);

    Ok(MemoryHealthInfo {
        total_bytes,
        available_bytes,
        utilization: if total_bytes > 0 {
            used_bytes as f64 / total_bytes as f64
        } else {
            0.0
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn collect_cpu_info_returns_valid_data() {
        let cpu_info = collect_cpu_info().await.expect("Failed to collect CPU info");

        // Validate cores and threads
        assert!(cpu_info.cores > 0, "CPU cores should be positive");
        assert!(cpu_info.threads >= cpu_info.cores, "Threads should be >= cores");

        // Validate utilization range
        assert!(
            cpu_info.utilization >= 0.0 && cpu_info.utilization <= 1.0,
            "CPU utilization should be between 0.0 and 1.0"
        );

        // Validate SIMD capabilities
        #[cfg(target_arch = "x86_64")]
        {
            // Most modern x86_64 CPUs support at least SSE2
            assert!(
                !cpu_info.simd_capabilities.is_empty(),
                "x86_64 CPU should have some SIMD capabilities"
            );
        }
    }

    #[tokio::test]
    async fn collect_memory_info_returns_valid_data() {
        let mem_info =
            collect_memory_health_info().await.expect("Failed to collect memory info");

        // Validate memory sizes
        assert!(mem_info.total_bytes > 0, "Total memory should be positive");
        assert!(
            mem_info.available_bytes <= mem_info.total_bytes,
            "Available memory should be <= total memory"
        );

        // Validate utilization range
        assert!(
            mem_info.utilization >= 0.0 && mem_info.utilization <= 1.0,
            "Memory utilization should be between 0.0 and 1.0"
        );
    }

    #[test]
    fn detect_simd_returns_non_empty_on_x86_64() {
        #[cfg(target_arch = "x86_64")]
        {
            let caps = detect_simd_capabilities();
            // Most modern x86_64 CPUs support at least SSE2
            assert!(!caps.is_empty(), "Should detect SIMD capabilities on x86_64");
        }
    }
}
