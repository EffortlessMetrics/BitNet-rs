use crate::common::errors::{ImplementationError, ImplementationResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Configuration for inference operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub stop_tokens: Vec<String>,
    pub seed: Option<u64>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            repetition_penalty: 1.0,
            stop_tokens: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
            seed: None,
        }
    }
}

/// Result of an inference operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub probabilities: Option<Vec<f32>>,
    pub logits: Option<Vec<Vec<f32>>>,
    pub duration: Duration,
    pub memory_usage: u64,
    pub token_count: usize,
}

/// Performance metrics collected during operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub model_load_time: Duration,
    pub tokenization_time: Duration,
    pub inference_time: Duration,
    pub total_time: Duration,
    pub peak_memory: u64,
    pub average_memory: u64,
    pub tokens_per_second: f64,
    pub memory_efficiency: f64,
    pub custom_metrics: HashMap<String, f64>,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }

    pub fn calculate_tokens_per_second(&mut self, token_count: usize) {
        if self.inference_time.as_secs_f64() > 0.0 {
            self.tokens_per_second = token_count as f64 / self.inference_time.as_secs_f64();
        }
    }

    pub fn calculate_memory_efficiency(&mut self, model_size: u64) {
        if model_size > 0 {
            self.memory_efficiency = model_size as f64 / self.peak_memory as f64;
        }
    }
}

/// Resource management information
#[derive(Debug, Clone)]
pub struct ResourceInfo {
    pub memory_usage: u64,
    pub file_handles: usize,
    pub thread_count: usize,
    pub gpu_memory: Option<u64>,
}

/// Abstract trait for BitNet implementations
#[async_trait]
pub trait BitNetImplementation: Send + Sync {
    /// Get the name of this implementation
    fn implementation_name(&self) -> &str;

    /// Get the version of this implementation
    fn implementation_version(&self) -> &str;

    /// Check if this implementation is available on the current system
    async fn is_available(&self) -> bool;

    /// Initialize the implementation with optional configuration
    async fn initialize(&mut self, config: Option<&str>) -> ImplementationResult<()>;

    /// Load a model from the specified path
    async fn load_model(&mut self, model_path: &Path) -> ImplementationResult<()>;

    /// Unload the currently loaded model
    async fn unload_model(&mut self) -> ImplementationResult<()>;

    /// Check if a model is currently loaded
    fn is_model_loaded(&self) -> bool;

    /// Get information about the loaded model
    fn get_model_info(&self) -> Option<ModelInfo>;

    /// Tokenize input text into tokens
    async fn tokenize(&self, text: &str) -> ImplementationResult<Vec<u32>>;

    /// Detokenize tokens back into text
    async fn detokenize(&self, tokens: &[u32]) -> ImplementationResult<String>;

    /// Run inference on the provided tokens
    async fn inference(
        &self,
        tokens: &[u32],
        config: &InferenceConfig,
    ) -> ImplementationResult<InferenceResult>;

    /// Run inference on text directly (tokenize + inference + detokenize)
    async fn inference_text(
        &self,
        text: &str,
        config: &InferenceConfig,
    ) -> ImplementationResult<String> {
        let tokens = self.tokenize(text).await?;
        let result = self.inference(&tokens, config).await?;
        Ok(result.text)
    }

    /// Get current performance metrics
    fn get_metrics(&self) -> PerformanceMetrics;

    /// Reset performance metrics
    fn reset_metrics(&mut self);

    /// Get current resource usage information
    fn get_resource_info(&self) -> ResourceInfo;

    /// Cleanup resources and prepare for shutdown
    async fn cleanup(&mut self) -> ImplementationResult<()>;

    /// Get implementation-specific capabilities
    fn get_capabilities(&self) -> ImplementationCapabilities;
}

/// Information about a loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub format: ModelFormat,
    pub size_bytes: u64,
    pub parameter_count: Option<u64>,
    pub context_length: Option<usize>,
    pub vocabulary_size: Option<usize>,
    pub architecture: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Supported model formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFormat {
    GGUF,
    SafeTensors,
    PyTorch,
    ONNX,
    Custom(String),
}

/// Implementation capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationCapabilities {
    pub supports_streaming: bool,
    pub supports_batching: bool,
    pub supports_gpu: bool,
    pub supports_quantization: bool,
    pub max_context_length: Option<usize>,
    pub supported_formats: Vec<ModelFormat>,
    pub custom_capabilities: HashMap<String, bool>,
}

/// Implementation discovery and loading
pub struct ImplementationRegistry {
    implementations: HashMap<String, Box<dyn ImplementationFactory>>,
}

impl ImplementationRegistry {
    pub fn new() -> Self {
        Self {
            implementations: HashMap::new(),
        }
    }

    pub fn register<F>(&mut self, name: String, factory: F)
    where
        F: ImplementationFactory + 'static,
    {
        self.implementations.insert(name, Box::new(factory));
    }

    pub async fn create_implementation(
        &self,
        name: &str,
    ) -> ImplementationResult<Box<dyn BitNetImplementation>> {
        let factory =
            self.implementations
                .get(name)
                .ok_or_else(|| ImplementationError::NotAvailable {
                    name: name.to_string(),
                })?;

        factory.create().await
    }

    pub fn list_available(&self) -> Vec<String> {
        self.implementations.keys().cloned().collect()
    }

    pub async fn discover_implementations(&mut self) -> ImplementationResult<Vec<String>> {
        let mut available = Vec::new();

        for (name, factory) in &self.implementations {
            let impl_instance = factory.create().await?;
            if impl_instance.is_available().await {
                available.push(name.clone());
            }
        }

        Ok(available)
    }
}

impl Default for ImplementationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory trait for creating implementations
#[async_trait]
pub trait ImplementationFactory: Send + Sync {
    async fn create(&self) -> ImplementationResult<Box<dyn BitNetImplementation>>;
}

/// Resource manager for tracking and cleaning up implementation resources
pub struct ResourceManager {
    active_implementations: HashMap<String, Box<dyn BitNetImplementation>>,
    resource_limits: ResourceLimits,
}

#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_memory: Option<u64>,
    pub max_implementations: Option<usize>,
    pub max_models_per_implementation: Option<usize>,
}

impl ResourceManager {
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            active_implementations: HashMap::new(),
            resource_limits: limits,
        }
    }

    pub async fn add_implementation(
        &mut self,
        id: String,
        implementation: Box<dyn BitNetImplementation>,
    ) -> ImplementationResult<()> {
        // Check resource limits
        if let Some(max_impls) = self.resource_limits.max_implementations {
            if self.active_implementations.len() >= max_impls {
                return Err(ImplementationError::NotAvailable {
                    name: "Resource limit exceeded".to_string(),
                });
            }
        }

        if let Some(max_memory) = self.resource_limits.max_memory {
            let current_memory = self.get_total_memory_usage();
            let impl_memory = implementation.get_resource_info().memory_usage;
            if current_memory + impl_memory > max_memory {
                return Err(ImplementationError::NotAvailable {
                    name: "Memory limit exceeded".to_string(),
                });
            }
        }

        self.active_implementations.insert(id, implementation);
        Ok(())
    }

    pub async fn remove_implementation(&mut self, id: &str) -> ImplementationResult<()> {
        if let Some(mut implementation) = self.active_implementations.remove(id) {
            implementation.cleanup().await?;
        }
        Ok(())
    }

    pub fn get_implementation(&self, id: &str) -> Option<&dyn BitNetImplementation> {
        self.active_implementations.get(id).map(|b| b.as_ref())
    }

    pub fn get_implementation_mut(&mut self, id: &str) -> Option<&mut dyn BitNetImplementation> {
        self.active_implementations.get_mut(id).map(|b| b.as_mut())
    }

    pub fn get_total_memory_usage(&self) -> u64 {
        self.active_implementations
            .values()
            .map(|impl_| impl_.get_resource_info().memory_usage)
            .sum()
    }

    pub async fn cleanup_all(&mut self) -> ImplementationResult<()> {
        let ids: Vec<String> = self.active_implementations.keys().cloned().collect();
        for id in ids {
            self.remove_implementation(&id).await?;
        }
        Ok(())
    }

    pub fn get_resource_summary(&self) -> ResourceSummary {
        let total_memory = self.get_total_memory_usage();
        let total_implementations = self.active_implementations.len();
        let total_file_handles: usize = self
            .active_implementations
            .values()
            .map(|impl_| impl_.get_resource_info().file_handles)
            .sum();

        ResourceSummary {
            total_memory,
            total_implementations,
            total_file_handles,
            active_implementations: self.active_implementations.keys().cloned().collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSummary {
    pub total_memory: u64,
    pub total_implementations: usize,
    pub total_file_handles: usize,
    pub active_implementations: Vec<String>,
}

/// Utility functions for implementation management
pub mod utils {
    use super::*;

    /// Get system memory usage
    pub fn get_memory_usage() -> u64 {
        #[cfg(target_os = "linux")]
        {
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
        }

        #[cfg(target_os = "macos")]
        {
            use libc::{getrusage, rusage, RUSAGE_SELF};
            unsafe {
                let mut usage = std::mem::zeroed::<rusage>();
                if getrusage(RUSAGE_SELF, &mut usage) == 0 {
                    return usage.ru_maxrss as u64; // Already in bytes on macOS
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            use winapi::um::processthreadsapi::GetCurrentProcess;
            use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
            unsafe {
                let mut counters = std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>();
                counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
                if GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
                    return counters.WorkingSetSize as u64;
                }
            }
        }

        0 // Fallback
    }

    /// Get peak memory usage since process start
    pub fn get_peak_memory_usage() -> u64 {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(contents) = fs::read_to_string("/proc/self/status") {
                for line in contents.lines() {
                    if line.starts_with("VmHWM:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use libc::{getrusage, rusage, RUSAGE_SELF};
            unsafe {
                let mut usage = std::mem::zeroed::<rusage>();
                if getrusage(RUSAGE_SELF, &mut usage) == 0 {
                    return usage.ru_maxrss as u64; // Already in bytes on macOS
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            use winapi::um::processthreadsapi::GetCurrentProcess;
            use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
            unsafe {
                let mut counters = std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>();
                counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
                if GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0 {
                    return counters.PeakWorkingSetSize as u64;
                }
            }
        }

        get_memory_usage() // Fallback to current usage
    }

    /// Measure execution time and memory usage of an async operation
    pub async fn measure_performance<F, T>(operation: F) -> (T, Duration, u64, u64)
    where
        F: std::future::Future<Output = T>,
    {
        let start_time = Instant::now();
        let start_memory = get_memory_usage();

        let result = operation.await;

        let duration = start_time.elapsed();
        let end_memory = get_memory_usage();
        let peak_memory = get_peak_memory_usage();

        (
            result,
            duration,
            end_memory.saturating_sub(start_memory),
            peak_memory,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert!(config.top_k.is_none());
    }

    #[tokio::test]
    async fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        metrics.add_custom_metric("test_metric".to_string(), 42.0);
        assert_eq!(metrics.custom_metrics.get("test_metric"), Some(&42.0));
    }

    #[tokio::test]
    async fn test_implementation_registry() {
        let registry = ImplementationRegistry::new();
        assert!(registry.list_available().is_empty());
    }

    #[tokio::test]
    async fn test_resource_manager() {
        let limits = ResourceLimits {
            max_memory: Some(1024 * 1024 * 1024), // 1GB
            max_implementations: Some(5),
            max_models_per_implementation: Some(2),
        };
        let manager = ResourceManager::new(limits);
        assert_eq!(manager.get_total_memory_usage(), 0);
    }

    #[test]
    fn test_memory_utils() {
        let memory = utils::get_memory_usage();
        assert!(memory > 0); // Should have some memory usage

        let peak_memory = utils::get_peak_memory_usage();
        assert!(peak_memory >= memory); // Peak should be >= current
    }
}
