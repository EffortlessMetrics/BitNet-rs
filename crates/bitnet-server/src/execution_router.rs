//! Device-aware execution routing with intelligent fallback

use anyhow::Result;
use bitnet_common::Device;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Device capabilities and performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub device: Device,
    pub available: bool,
    pub memory_total_mb: u64,
    pub memory_free_mb: u64,
    pub compute_capability: Option<String>, // For CUDA devices
    pub simd_support: Vec<String>,          // For CPU devices (AVX2, AVX-512, NEON)
    pub avg_tokens_per_second: f64,
    pub last_benchmark: Option<std::time::SystemTime>,
}

/// Device selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceSelectionStrategy {
    /// Always prefer GPU if available
    PreferGpu,
    /// Always use CPU
    CpuOnly,
    /// Prefer device with best performance for workload
    PerformanceBased,
    /// Load balance across available devices
    LoadBalance,
    /// User-specified device preference
    UserPreference(Device),
}

/// Execution routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRouterConfig {
    pub strategy: DeviceSelectionStrategy,
    pub fallback_enabled: bool,
    pub health_check_interval: Duration,
    pub performance_threshold_tps: f64,
    pub memory_threshold_percent: f64,
    pub benchmark_on_startup: bool,
}

impl Default for ExecutionRouterConfig {
    fn default() -> Self {
        Self {
            strategy: DeviceSelectionStrategy::PerformanceBased,
            fallback_enabled: true,
            health_check_interval: Duration::from_secs(30),
            performance_threshold_tps: 10.0, // Minimum tokens per second
            memory_threshold_percent: 0.8,   // 80% memory usage threshold
            benchmark_on_startup: true,
        }
    }
}

/// Device health status
#[derive(Debug, Clone, Serialize)]
pub enum DeviceHealth {
    Healthy,
    Degraded { reason: String },
    Unavailable { reason: String },
}

/// Device execution statistics
#[derive(Debug)]
pub struct DeviceStats {
    pub requests_processed: AtomicU64,
    pub total_tokens_generated: AtomicU64,
    pub total_execution_time: AtomicU64, // in milliseconds
    pub last_request_time: RwLock<Option<Instant>>,
    pub consecutive_failures: AtomicU64,
}

impl Clone for DeviceStats {
    fn clone(&self) -> Self {
        Self {
            requests_processed: AtomicU64::new(self.requests_processed.load(Ordering::Relaxed)),
            total_tokens_generated: AtomicU64::new(
                self.total_tokens_generated.load(Ordering::Relaxed),
            ),
            total_execution_time: AtomicU64::new(self.total_execution_time.load(Ordering::Relaxed)),
            last_request_time: RwLock::new(None), // Reset on clone
            consecutive_failures: AtomicU64::new(self.consecutive_failures.load(Ordering::Relaxed)),
        }
    }
}

impl Default for DeviceStats {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceStats {
    pub fn new() -> Self {
        Self {
            requests_processed: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            total_execution_time: AtomicU64::new(0),
            last_request_time: RwLock::new(None),
            consecutive_failures: AtomicU64::new(0),
        }
    }

    pub async fn record_success(&self, tokens: u64, duration: Duration) {
        self.requests_processed.fetch_add(1, Ordering::Relaxed);
        self.total_tokens_generated.fetch_add(tokens, Ordering::Relaxed);
        self.total_execution_time.fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);

        let mut last_request = self.last_request_time.write().await;
        *last_request = Some(Instant::now());
    }

    pub fn record_failure(&self) {
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_avg_tokens_per_second(&self) -> f64 {
        let total_tokens = self.total_tokens_generated.load(Ordering::Relaxed);
        let total_time_ms = self.total_execution_time.load(Ordering::Relaxed);

        if total_time_ms > 0 && total_tokens > 0 {
            (total_tokens as f64 * 1000.0) / total_time_ms as f64
        } else {
            0.0
        }
    }
}

/// Device monitor for tracking health and performance
pub struct DeviceMonitor {
    pub device: Device,
    pub capabilities: RwLock<DeviceCapabilities>,
    pub health: RwLock<DeviceHealth>,
    pub stats: DeviceStats,
    system: RwLock<System>,
}

impl DeviceMonitor {
    pub fn new(device: Device) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        let capabilities = DeviceCapabilities {
            device,
            available: Self::check_device_availability(&device, &system),
            memory_total_mb: Self::get_device_memory_total(&device, &system),
            memory_free_mb: Self::get_device_memory_free(&device, &system),
            compute_capability: Self::get_compute_capability(&device),
            simd_support: Self::get_simd_support(&device),
            avg_tokens_per_second: 0.0,
            last_benchmark: None,
        };

        Self {
            device,
            capabilities: RwLock::new(capabilities),
            health: RwLock::new(DeviceHealth::Healthy),
            stats: DeviceStats::new(),
            system: RwLock::new(system),
        }
    }

    /// Check if device is available
    fn check_device_availability(device: &Device, _system: &System) -> bool {
        match device {
            Device::Cpu => true,
            Device::Cuda(device_id) => {
                // Check if CUDA device is available
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                {
                    use bitnet_kernels::device_features::gpu_available_runtime;

                    // First check if GPU is available at runtime
                    if !gpu_available_runtime() {
                        return false;
                    }

                    // Then verify the specific device ID is valid
                    let device_count = bitnet_kernels::gpu::cuda::cuda_device_count();
                    *device_id < device_count
                }
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                {
                    let _ = device_id;
                    false
                }
            }
            Device::Metal => false, // TODO: Implement Metal device support
        }
    }

    /// Get total device memory (in MB)
    fn get_device_memory_total(device: &Device, system: &System) -> u64 {
        match device {
            // sysinfo returns KiB - convert to MB
            Device::Cpu => system.total_memory() / 1024, // KiB → MB
            Device::Cuda(device_id) => {
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                {
                    use bitnet_kernels::gpu::cuda::CudaKernel;

                    // Query device information to get total memory
                    match CudaKernel::get_device_info(*device_id) {
                        Ok(info) => (info.total_memory / (1024 * 1024)) as u64, // Convert bytes to MB
                        Err(_) => {
                            warn!(
                                device_id = device_id,
                                "Failed to query CUDA device memory, using default"
                            );
                            8192 // Default 8GB fallback
                        }
                    }
                }
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                {
                    let _ = device_id;
                    0
                }
            }
            Device::Metal => {
                // TODO: Get Metal device memory
                8192 // Default 8GB for now
            }
        }
    }

    /// Get free device memory
    fn get_device_memory_free(device: &Device, system: &System) -> u64 {
        match device {
            Device::Cpu => system.free_memory() / (1024 * 1024), // Convert to MB
            Device::Cuda(device_id) => {
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                {
                    use cudarc::driver::{
                        CudaContext,
                        sys::{cuMemGetInfo_v2, cudaError_enum},
                    };

                    // Query free memory using CUDA memory info API
                    match CudaContext::new(*device_id) {
                        Ok(_ctx) => {
                            let mut free_bytes: usize = 0;
                            let mut _total_bytes: usize = 0;
                            let result =
                                unsafe { cuMemGetInfo_v2(&mut free_bytes, &mut _total_bytes) };
                            if result == cudaError_enum::CUDA_SUCCESS {
                                (free_bytes / (1024 * 1024)) as u64 // Convert bytes to MB
                            } else {
                                warn!(
                                    device_id = device_id,
                                    "Failed to query CUDA free memory, estimating"
                                );
                                // Estimate 50% free as conservative fallback
                                let total_mb = Self::get_device_memory_total(device, system);
                                total_mb / 2
                            }
                        }
                        Err(_) => {
                            warn!(
                                device_id = device_id,
                                "Failed to create CUDA context for memory query"
                            );
                            4096 // Default 4GB free fallback
                        }
                    }
                }
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                {
                    let _ = device_id;
                    0
                }
            }
            Device::Metal => 4096, // TODO: Get Metal device free memory
        }
    }

    /// Get compute capability for GPU devices
    fn get_compute_capability(device: &Device) -> Option<String> {
        match device {
            Device::Cpu => None,
            Device::Cuda(device_id) => {
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                {
                    use bitnet_kernels::gpu::cuda::CudaKernel;

                    // Query device information to get compute capability
                    match CudaKernel::get_device_info(*device_id) {
                        Ok(info) => {
                            let (major, minor) = info.compute_capability;
                            Some(format!("{}.{}", major, minor))
                        }
                        Err(_) => {
                            warn!(
                                device_id = device_id,
                                "Failed to query CUDA compute capability, using default"
                            );
                            Some("8.6".to_string()) // Default modern GPU capability fallback
                        }
                    }
                }
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                {
                    let _ = device_id;
                    None
                }
            }
            Device::Metal => Some("Metal 3".to_string()), // TODO: Get actual Metal version
        }
    }

    /// Get SIMD support optimized for BitNet quantization operations
    fn get_simd_support(device: &Device) -> Vec<String> {
        match device {
            Device::Cpu => Self::detect_cpu_simd_features(),
            Device::Cuda(_) | Device::Metal => Vec::new(), // GPU devices don't use CPU SIMD
        }
    }

    /// Detect CPU SIMD features optimized for BitNet quantization
    fn detect_cpu_simd_features() -> Vec<String> {
        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            // Order by performance preference for BitNet I2S operations
            if std::arch::is_x86_feature_detected!("avx512f") {
                features.push("AVX-512".to_string());
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                features.push("AVX2".to_string());
            }
            if std::arch::is_x86_feature_detected!("sse4.1") {
                features.push("SSE4.1".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM NEON is excellent for BitNet quantized operations
            features.push("NEON".to_string());
        }

        features
    }

    /// Update device health status
    pub async fn update_health(&self) {
        let mut system = self.system.write().await;
        system.refresh_all();

        let mut capabilities = self.capabilities.write().await;
        let mut health = self.health.write().await;

        // Update memory info
        capabilities.memory_free_mb = Self::get_device_memory_free(&self.device, &system);

        // Check device health
        let memory_usage_percent =
            1.0 - (capabilities.memory_free_mb as f64 / capabilities.memory_total_mb as f64);
        let consecutive_failures = self.stats.consecutive_failures.load(Ordering::Relaxed);

        *health = if !capabilities.available {
            DeviceHealth::Unavailable { reason: "Device not available".to_string() }
        } else if consecutive_failures > 5 {
            DeviceHealth::Unavailable {
                reason: format!("Too many consecutive failures: {}", consecutive_failures),
            }
        } else if memory_usage_percent > 0.95 {
            DeviceHealth::Degraded {
                reason: format!("High memory usage: {:.1}%", memory_usage_percent * 100.0),
            }
        } else if consecutive_failures > 2 {
            DeviceHealth::Degraded { reason: format!("Recent failures: {}", consecutive_failures) }
        } else {
            DeviceHealth::Healthy
        };
    }

    /// Perform benchmark to measure performance
    pub async fn benchmark(&self) -> Result<f64> {
        info!(device = ?self.device, "Starting device benchmark");

        // TODO: Implement actual benchmark using a small inference task
        // For now, return estimated performance based on device type
        let tokens_per_second = match &self.device {
            Device::Cpu => {
                let simd_support = Self::get_simd_support(&self.device);
                if simd_support.contains(&"AVX-512".to_string()) {
                    45.0
                } else if simd_support.contains(&"AVX2".to_string()) {
                    35.0
                } else if simd_support.contains(&"NEON".to_string()) {
                    25.0
                } else {
                    15.0
                }
            }
            Device::Cuda(_) => {
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                {
                    120.0 // Estimated GPU performance
                }
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                0.0
            }
            Device::Metal => {
                #[cfg(target_os = "macos")]
                {
                    80.0 // Estimated Metal performance
                }
                #[cfg(not(target_os = "macos"))]
                0.0
            }
        };

        // Update capabilities
        {
            let mut capabilities = self.capabilities.write().await;
            capabilities.avg_tokens_per_second = tokens_per_second;
            capabilities.last_benchmark = Some(std::time::SystemTime::now());
        }

        info!(
            device = ?self.device,
            tokens_per_second = tokens_per_second,
            "Benchmark completed"
        );

        Ok(tokens_per_second)
    }

    /// Get current device status
    pub async fn get_status(&self) -> DeviceStatus {
        let capabilities = self.capabilities.read().await;
        let health = self.health.read().await;

        DeviceStatus {
            device: self.device,
            health: health.clone(),
            capabilities: capabilities.clone(),
            avg_tokens_per_second: self.stats.get_avg_tokens_per_second(),
            requests_processed: self.stats.requests_processed.load(Ordering::Relaxed),
            consecutive_failures: self.stats.consecutive_failures.load(Ordering::Relaxed),
        }
    }
}

/// Device status summary
#[derive(Debug, Clone, Serialize)]
pub struct DeviceStatus {
    pub device: Device,
    pub health: DeviceHealth,
    pub capabilities: DeviceCapabilities,
    pub avg_tokens_per_second: f64,
    pub requests_processed: u64,
    pub consecutive_failures: u64,
}

/// Execution router with device-aware routing
pub struct ExecutionRouter {
    config: ExecutionRouterConfig,
    device_monitors: Vec<Arc<DeviceMonitor>>,
    current_device_index: AtomicU64,
}

impl ExecutionRouter {
    /// Create a new execution router
    pub async fn new(config: ExecutionRouterConfig, devices: Vec<Device>) -> Result<Self> {
        let mut device_monitors = Vec::new();

        for device in devices {
            let monitor = Arc::new(DeviceMonitor::new(device));

            // Perform initial health check
            monitor.update_health().await;

            // Benchmark if enabled
            if config.benchmark_on_startup
                && let Err(e) = monitor.benchmark().await
            {
                warn!(device = ?monitor.device, error = %e, "Failed to benchmark device");
            }

            device_monitors.push(monitor);
        }

        Ok(Self { config, device_monitors, current_device_index: AtomicU64::new(0) })
    }

    /// Select the best device for execution
    pub async fn select_device(&self) -> Option<Device> {
        match &self.config.strategy {
            DeviceSelectionStrategy::PreferGpu => self.select_device_prefer_gpu().await,
            DeviceSelectionStrategy::CpuOnly => self.select_cpu_device().await,
            DeviceSelectionStrategy::PerformanceBased => self.select_device_by_performance().await,
            DeviceSelectionStrategy::LoadBalance => self.select_device_load_balance().await,
            DeviceSelectionStrategy::UserPreference(device) => {
                if self.is_device_healthy(device).await {
                    Some(*device)
                } else if self.config.fallback_enabled {
                    self.select_fallback_device().await
                } else {
                    None
                }
            }
        }
    }

    /// Select GPU device if available and healthy
    async fn select_device_prefer_gpu(&self) -> Option<Device> {
        // First try GPU devices
        for monitor in &self.device_monitors {
            if matches!(monitor.device, Device::Cuda(_)) {
                let health = monitor.health.read().await;
                if matches!(*health, DeviceHealth::Healthy) {
                    return Some(monitor.device);
                }
            }
        }

        // Fallback to CPU if enabled
        if self.config.fallback_enabled { self.select_cpu_device().await } else { None }
    }

    /// Select CPU device
    async fn select_cpu_device(&self) -> Option<Device> {
        for monitor in &self.device_monitors {
            if matches!(monitor.device, Device::Cpu) {
                let health = monitor.health.read().await;
                if !matches!(*health, DeviceHealth::Unavailable { .. }) {
                    return Some(monitor.device);
                }
            }
        }
        None
    }

    /// Select device based on performance
    async fn select_device_by_performance(&self) -> Option<Device> {
        let mut best_device = None;
        let mut best_performance = 0.0;

        for monitor in &self.device_monitors {
            let health = monitor.health.read().await;
            if matches!(*health, DeviceHealth::Healthy) {
                let performance = monitor.stats.get_avg_tokens_per_second();
                let capabilities = monitor.capabilities.read().await;

                // Use benchmarked performance if available, otherwise use capabilities
                let device_performance = if performance > 0.0 {
                    performance
                } else {
                    capabilities.avg_tokens_per_second
                };

                if device_performance > best_performance {
                    best_performance = device_performance;
                    best_device = Some(monitor.device);
                }
            }
        }

        best_device
    }

    /// Load balance across available devices
    async fn select_device_load_balance(&self) -> Option<Device> {
        let healthy_devices: Vec<_> = self.get_healthy_devices().await;

        if healthy_devices.is_empty() {
            return None;
        }

        let index = self.current_device_index.fetch_add(1, Ordering::Relaxed) as usize;
        let selected_index = index % healthy_devices.len();

        Some(healthy_devices[selected_index])
    }

    /// Get list of healthy devices
    async fn get_healthy_devices(&self) -> Vec<Device> {
        let mut healthy = Vec::new();

        for monitor in &self.device_monitors {
            let health = monitor.health.read().await;
            if matches!(*health, DeviceHealth::Healthy) {
                healthy.push(monitor.device);
            }
        }

        healthy
    }

    /// Check if a specific device is healthy
    async fn is_device_healthy(&self, device: &Device) -> bool {
        for monitor in &self.device_monitors {
            if monitor.device == *device {
                let health = monitor.health.read().await;
                return !matches!(*health, DeviceHealth::Unavailable { .. });
            }
        }
        false
    }

    /// Select fallback device
    async fn select_fallback_device(&self) -> Option<Device> {
        // Prefer CPU as fallback
        self.select_cpu_device().await
    }

    /// Record execution result for device learning
    pub async fn record_execution_result(
        &self,
        device: &Device,
        tokens: u64,
        duration: Duration,
        success: bool,
    ) {
        for monitor in &self.device_monitors {
            if monitor.device == *device {
                if success {
                    monitor.stats.record_success(tokens, duration).await;
                } else {
                    monitor.stats.record_failure();
                }
                break;
            }
        }
    }

    /// Get all device statuses
    pub async fn get_device_statuses(&self) -> Vec<DeviceStatus> {
        let mut statuses = Vec::new();
        for monitor in &self.device_monitors {
            statuses.push(monitor.get_status().await);
        }
        statuses
    }

    /// Update device health (should be called periodically)
    pub async fn update_device_health(&self) {
        for monitor in &self.device_monitors {
            monitor.update_health().await;
        }
    }

    /// Get router health summary
    pub async fn get_health_summary(&self) -> ExecutionRouterHealth {
        let total_devices = self.device_monitors.len();
        let mut healthy_devices = 0;
        let mut degraded_devices = 0;
        let mut unavailable_devices = 0;

        for monitor in &self.device_monitors {
            let health = monitor.health.read().await;
            match *health {
                DeviceHealth::Healthy => healthy_devices += 1,
                DeviceHealth::Degraded { .. } => degraded_devices += 1,
                DeviceHealth::Unavailable { .. } => unavailable_devices += 1,
            }
        }

        ExecutionRouterHealth {
            total_devices,
            healthy_devices,
            degraded_devices,
            unavailable_devices,
            fallback_enabled: self.config.fallback_enabled,
            strategy: format!("{:?}", self.config.strategy),
        }
    }
}

/// Router health summary
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionRouterHealth {
    pub total_devices: usize,
    pub healthy_devices: usize,
    pub degraded_devices: usize,
    pub unavailable_devices: usize,
    pub fallback_enabled: bool,
    pub strategy: String,
}
