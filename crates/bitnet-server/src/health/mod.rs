//! Health monitoring modules for AC05
//!
//! This module provides health check functionality including:
//! - CPU metrics collection and monitoring
//! - GPU metrics collection and monitoring
//! - Component health status tracking
//! - System resource monitoring
//! - Performance indicators for SLA compliance

pub mod cpu_monitor;
pub mod gpu_monitor;
pub mod performance;

// Re-export for convenience
pub use cpu_monitor::{CpuInfo, MemoryHealthInfo, collect_cpu_info, collect_memory_health_info};
pub use gpu_monitor::{GpuMemoryLeakDetector, GpuMemoryLeakStatus, GpuMetrics};
pub use performance::{PerformanceIndicators, PerformanceMetrics};
