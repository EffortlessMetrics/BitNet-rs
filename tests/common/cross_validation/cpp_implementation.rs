//! C++ implementation wrapper for BitNet.cpp cross-validation testing
//!
//! This module provides a wrapper around the BitNet.cpp implementation that conforms
//! to the BitNetImplementation trait for cross-validation testing.

use crate::cross_validation::implementation::{
    BitNetImplementation, ImplementationCapabilities, ImplementationFactory, InferenceConfig,
    InferenceResult, ModelFormat, ModelInfo, PerformanceMetrics, ResourceInfo,
};
use crate::errors::{ImplementationError, ImplementationResult};
use async_trait::async_trait;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::Command as AsyncCommand;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, warn};

// Note: FFI functions are expected to be provided by linking against BitNet.cpp
// In test environments, stub implementations may be needed

/// FFI bindings to the C++ BitNet implementation
#[repr(C)]
pub struct BitNetCppHandle {
    pub _private: [u8; 0],
}

/// Thread-safe wrapper for the C++ handle
struct CppHandleWrapper(*mut BitNetCppHandle);

// SAFETY: The C++ implementation is assumed to be thread-safe
// In a real implementation, this would need to be verified
unsafe impl Send for CppHandleWrapper {}
unsafe impl Sync for CppHandleWrapper {}

impl CppHandleWrapper {
    fn new(handle: *mut BitNetCppHandle) -> Self {
        Self(handle)
    }

    fn as_ptr(&self) -> *mut BitNetCppHandle {
        self.0
    }

    fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// C++ inference configuration
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CppInferenceConfig {
    pub max_tokens: c_uint,
    pub temperature: c_float,
    pub top_p: c_float,
    pub top_k: c_int, // -1 for disabled
    pub repetition_penalty: c_float,
    pub seed: c_int, // -1 for random
}

/// C++ inference result
#[repr(C)]
pub struct CppInferenceResult {
    pub tokens: *mut c_uint,
    pub token_count: c_uint,
    pub text: *const c_char,
    pub duration_ms: c_uint,
    pub memory_usage: c_uint,
}
/// C++ model information
#[repr(C)]
pub struct CppModelInfo {
    pub name: *const c_char,
    pub format: c_int,
    pub size_bytes: c_uint,
    pub parameter_count: c_uint,
    pub context_length: c_uint,
    pub vocabulary_size: c_uint,
}

/// C++ performance metrics
#[repr(C)]
pub struct CppPerformanceMetrics {
    pub model_load_time_ms: c_uint,
    pub tokenization_time_ms: c_uint,
    pub inference_time_ms: c_uint,
    pub peak_memory: c_uint,
    pub tokens_per_second: c_float,
}

// External C++ function declarations
#[cfg_attr(feature = "cpp-ffi", link(name = "bitnet_cpp"))]
unsafe extern "C" {
    fn bitnet_cpp_create() -> *mut BitNetCppHandle;
    fn bitnet_cpp_destroy(handle: *mut BitNetCppHandle);
    fn bitnet_cpp_is_available() -> c_int;
    fn bitnet_cpp_load_model(handle: *mut BitNetCppHandle, path: *const c_char) -> c_int;
    fn bitnet_cpp_unload_model(handle: *mut BitNetCppHandle) -> c_int;
    fn bitnet_cpp_is_model_loaded(handle: *mut BitNetCppHandle) -> c_int;
    fn bitnet_cpp_get_model_info(handle: *mut BitNetCppHandle) -> CppModelInfo;
    fn bitnet_cpp_tokenize(
        handle: *mut BitNetCppHandle,
        text: *const c_char,
        tokens: *mut *mut c_uint,
        token_count: *mut c_uint,
    ) -> c_int;
    fn bitnet_cpp_detokenize(
        handle: *mut BitNetCppHandle,
        tokens: *const c_uint,
        token_count: c_uint,
        text: *mut *mut c_char,
    ) -> c_int;
    fn bitnet_cpp_inference(
        handle: *mut BitNetCppHandle,
        tokens: *const c_uint,
        token_count: c_uint,
        config: *const CppInferenceConfig,
        result: *mut CppInferenceResult,
    ) -> c_int;
    fn bitnet_cpp_get_metrics(handle: *mut BitNetCppHandle) -> CppPerformanceMetrics;
    fn bitnet_cpp_reset_metrics(handle: *mut BitNetCppHandle);
    fn bitnet_cpp_cleanup(handle: *mut BitNetCppHandle) -> c_int;
    fn bitnet_cpp_free_string(ptr: *mut c_char);
    fn bitnet_cpp_free_tokens(ptr: *mut c_uint);
}
/// C++ implementation wrapper for BitNet.cpp
pub struct CppImplementation {
    /// Name of this implementation
    name: String,
    /// Version of this implementation
    version: String,
    /// Path to the C++ binary
    binary_path: Option<PathBuf>,
    /// FFI handle to the C++ implementation
    handle: Option<CppHandleWrapper>,
    /// Loaded model information
    model_info: Option<ModelInfo>,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Resource tracking
    resource_info: Arc<RwLock<ResourceInfo>>,
    /// Implementation capabilities
    capabilities: ImplementationCapabilities,
    /// Whether the implementation is available
    is_available: bool,
}

impl CppImplementation {
    /// Create a new C++ implementation wrapper
    pub fn new() -> Self {
        let capabilities = ImplementationCapabilities {
            supports_streaming: true,
            supports_batching: false, // C++ implementation doesn't support batching yet
            supports_gpu: true,       // Depends on compilation flags
            supports_quantization: true,
            max_context_length: Some(4096), // Default context length for C++ implementation
            supported_formats: vec![ModelFormat::GGUF, ModelFormat::Custom("bin".to_string())],
            custom_capabilities: HashMap::new(),
        };

        Self {
            name: "BitNet.cpp".to_string(),
            version: "1.0.0".to_string(), // Will be detected from binary
            binary_path: None,
            handle: None,
            model_info: None,
            metrics: Arc::new(RwLock::new(PerformanceMetrics::new())),
            resource_info: Arc::new(RwLock::new(ResourceInfo {
                memory_usage: 0,
                file_handles: 0,
                thread_count: 1,
                gpu_memory: None,
            })),
            capabilities,
            is_available: false,
        }
    }
    /// Discover the C++ binary in the system
    async fn discover_binary(&mut self) -> ImplementationResult<()> {
        // Search paths for the C++ binary
        let search_paths = vec![
            PathBuf::from("./legacy/bitnet.cpp/build/bin/bitnet"),
            PathBuf::from("./legacy/bitnet.cpp/bitnet"),
            PathBuf::from("/usr/local/bin/bitnet"),
            PathBuf::from("/usr/bin/bitnet"),
            PathBuf::from("bitnet"), // In PATH
        ];

        // Add Windows executable extension
        #[cfg(target_os = "windows")]
        let search_paths: Vec<PathBuf> =
            search_paths.into_iter().map(|p| p.with_extension("exe")).collect();

        for path in search_paths {
            if path.exists() {
                // Verify the binary works
                if self.verify_binary(&path).await? {
                    self.binary_path = Some(path);
                    self.is_available = true;
                    info!("Found C++ binary at: {}", self.binary_path.as_ref().unwrap().display());
                    return Ok(());
                }
            }
        }

        // Try to find in PATH
        if let Ok(output) = AsyncCommand::new("which").arg("bitnet").output().await {
            if output.status.success() {
                let path_str = String::from_utf8_lossy(&output.stdout);
                let path_str = path_str.trim();
                let path = PathBuf::from(path_str);
                if self.verify_binary(&path).await? {
                    self.binary_path = Some(path);
                    self.is_available = true;
                    info!("Found C++ binary in PATH: {}", path_str);
                    return Ok(());
                }
            }
        }

        warn!("C++ binary not found in any search paths");
        Err(ImplementationError::NotAvailable { name: "BitNet.cpp binary not found".to_string() })
    }

    /// Verify that a binary is the correct BitNet.cpp implementation
    async fn verify_binary(&self, path: &Path) -> ImplementationResult<bool> {
        let output = AsyncCommand::new(path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| ImplementationError::FfiError {
                message: format!("Failed to execute binary: {}", e),
            })?;

        if !output.status.success() {
            return Ok(false);
        }

        let version_output = String::from_utf8_lossy(&output.stdout);
        Ok(version_output.contains("BitNet") || version_output.contains("bitnet"))
    }

    /// Initialize the FFI handle
    fn initialize_ffi(&mut self) -> ImplementationResult<()> {
        if self.handle.is_some() {
            return Ok(());
        }

        // Check if C++ implementation is available via FFI
        let available = unsafe { bitnet_cpp_is_available() };
        if available == 0 {
            return Err(ImplementationError::NotAvailable {
                name: "C++ FFI not available".to_string(),
            });
        }

        // Create FFI handle
        let handle = unsafe { bitnet_cpp_create() };
        if handle.is_null() {
            return Err(ImplementationError::FfiError {
                message: "Failed to create C++ handle".to_string(),
            });
        }

        self.handle = Some(CppHandleWrapper::new(handle));
        Ok(())
    }

    /// Convert internal inference config to C++ config
    fn convert_inference_config(&self, config: &InferenceConfig) -> CppInferenceConfig {
        CppInferenceConfig {
            max_tokens: config.max_tokens as c_uint,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k.map(|k| k as c_int).unwrap_or(-1),
            repetition_penalty: config.repetition_penalty,
            seed: config.seed.map(|s| s as c_int).unwrap_or(-1),
        }
    }

    /// Convert C++ model info to internal format
    fn convert_model_info(&self, cpp_info: CppModelInfo, model_path: &Path) -> ModelInfo {
        let name = unsafe {
            if cpp_info.name.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(cpp_info.name).to_string_lossy().to_string()
            }
        };

        let format = match cpp_info.format {
            0 => ModelFormat::GGUF,
            1 => ModelFormat::Custom("bin".to_string()),
            _ => ModelFormat::Custom("unknown".to_string()),
        };

        let mut metadata = HashMap::new();
        metadata.insert("implementation".to_string(), self.name.clone());
        metadata.insert("version".to_string(), self.version.clone());
        metadata.insert("ffi".to_string(), "true".to_string());

        ModelInfo {
            name,
            path: model_path.to_path_buf(),
            format,
            size_bytes: cpp_info.size_bytes as u64,
            parameter_count: Some(cpp_info.parameter_count as u64),
            context_length: Some(cpp_info.context_length as usize),
            vocabulary_size: Some(cpp_info.vocabulary_size as usize),
            architecture: Some("BitNet".to_string()),
            metadata,
        }
    }

    /// Convert C++ performance metrics to internal format
    fn convert_performance_metrics(
        &self,
        cpp_metrics: CppPerformanceMetrics,
    ) -> PerformanceMetrics {
        PerformanceMetrics {
            model_load_time: Duration::from_millis(cpp_metrics.model_load_time_ms as u64),
            tokenization_time: Duration::from_millis(cpp_metrics.tokenization_time_ms as u64),
            inference_time: Duration::from_millis(cpp_metrics.inference_time_ms as u64),
            total_time: Duration::from_millis(
                cpp_metrics.model_load_time_ms as u64
                    + cpp_metrics.tokenization_time_ms as u64
                    + cpp_metrics.inference_time_ms as u64,
            ),
            peak_memory: cpp_metrics.peak_memory as u64,
            average_memory: cpp_metrics.peak_memory as u64, // C++ doesn't track average
            tokens_per_second: cpp_metrics.tokens_per_second as f64,
            memory_efficiency: 0.0, // Will be calculated separately
            custom_metrics: HashMap::new(),
        }
    }

    /// Update resource information
    async fn update_resource_info(&self) {
        let mut resource_info = self.resource_info.write().await;
        resource_info.memory_usage = self.get_memory_usage();
        resource_info.file_handles = self.get_file_handle_count();
        resource_info.thread_count = self.get_thread_count();
        resource_info.gpu_memory = Some(self.get_gpu_memory_usage());
    }

    /// Get current memory usage
    fn get_memory_usage(&self) -> u64 {
        crate::cross_validation::implementation::utils::get_memory_usage()
    }

    /// Get file handle count (placeholder implementation)
    fn get_file_handle_count(&self) -> usize {
        if self.model_info.is_some() {
            3 // Model file + tokenizer + temp files
        } else {
            1 // Just the binary handle
        }
    }

    /// Get thread count (placeholder implementation)
    fn get_thread_count(&self) -> usize {
        // C++ implementation typically uses multiple threads
        num_cpus::get().min(8) // Cap at 8 threads
    }

    /// Get GPU memory usage (placeholder implementation)
    fn get_gpu_memory_usage(&self) -> u64 {
        // In a real implementation, this would query GPU memory usage
        // For now, return 0
        0
    }
}

impl Default for CppImplementation {
    fn default() -> Self {
        Self::new()
    }
}

// Implement Drop to ensure proper cleanup
impl Drop for CppImplementation {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            unsafe {
                bitnet_cpp_destroy(handle.as_ptr());
            }
        }
    }
}
#[async_trait]
impl BitNetImplementation for CppImplementation {
    fn implementation_name(&self) -> &str {
        &self.name
    }

    fn implementation_version(&self) -> &str {
        &self.version
    }

    #[instrument(skip(self))]
    async fn is_available(&self) -> bool {
        self.is_available
    }

    #[instrument(skip(self, config))]
    async fn initialize(&mut self, config: Option<&str>) -> ImplementationResult<()> {
        info!("Initializing C++ implementation");

        if let Some(config_str) = config {
            debug!("Using configuration: {}", config_str);
        }

        // First try to discover the binary
        if let Err(e) = self.discover_binary().await {
            warn!("Binary discovery failed: {}", e);
            // Continue with FFI initialization
        }

        // Initialize FFI
        if let Err(e) = self.initialize_ffi() {
            warn!("FFI initialization failed: {}", e);
            // If both binary and FFI fail, mark as unavailable
            if self.binary_path.is_none() {
                return Err(ImplementationError::NotAvailable {
                    name: "Neither binary nor FFI available".to_string(),
                });
            }
        }

        // Update initial resource info
        self.update_resource_info().await;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn load_model(&mut self, model_path: &Path) -> ImplementationResult<()> {
        info!("Loading model from: {}", model_path.display());

        let handle = self
            .handle
            .as_ref()
            .ok_or(ImplementationError::NotAvailable {
                name: "C++ handle not initialized".to_string(),
            })?
            .as_ptr();

        // Convert path to C string
        let path_cstr = CString::new(model_path.to_string_lossy().as_bytes()).map_err(|e| {
            ImplementationError::FfiError { message: format!("Invalid path: {}", e) }
        })?;

        // Load model via FFI
        let result = unsafe { bitnet_cpp_load_model(handle, path_cstr.as_ptr()) };
        if result != 0 {
            return Err(ImplementationError::ModelLoadError {
                message: format!("C++ model loading failed with code: {}", result),
            });
        }

        // Get model info
        let cpp_model_info = unsafe { bitnet_cpp_get_model_info(handle) };
        self.model_info = Some(self.convert_model_info(cpp_model_info, model_path));

        // Update resource info
        self.update_resource_info().await;

        info!("Model loaded successfully");
        Ok(())
    }
    #[instrument(skip(self))]
    async fn unload_model(&mut self) -> ImplementationResult<()> {
        info!("Unloading model");

        if let Some(handle) = &self.handle {
            let result = unsafe { bitnet_cpp_unload_model(handle.as_ptr()) };
            if result != 0 {
                warn!("C++ model unloading failed with code: {}", result);
            }
        }

        self.model_info = None;

        // Update resource info
        self.update_resource_info().await;

        Ok(())
    }

    fn is_model_loaded(&self) -> bool {
        if let Some(handle) = &self.handle {
            unsafe { bitnet_cpp_is_model_loaded(handle.as_ptr()) != 0 }
        } else {
            false
        }
    }

    fn get_model_info(&self) -> Option<ModelInfo> {
        self.model_info.clone()
    }

    #[instrument(skip(self, text))]
    async fn tokenize(&self, text: &str) -> ImplementationResult<Vec<u32>> {
        let handle = self.handle.as_ref().ok_or(ImplementationError::ModelNotLoaded)?.as_ptr();

        // Convert text to C string
        let text_cstr = CString::new(text).map_err(|e| ImplementationError::TokenizationError {
            message: format!("Invalid text: {}", e),
        })?;

        let mut tokens_ptr: *mut c_uint = std::ptr::null_mut();
        let mut token_count: c_uint = 0;

        // Tokenize via FFI
        let result = unsafe {
            bitnet_cpp_tokenize(handle, text_cstr.as_ptr(), &mut tokens_ptr, &mut token_count)
        };

        if result != 0 {
            return Err(ImplementationError::TokenizationError {
                message: format!("C++ tokenization failed with code: {}", result),
            });
        }

        if tokens_ptr.is_null() || token_count == 0 {
            return Ok(Vec::new());
        }

        // Copy tokens to Rust Vec
        let tokens: Vec<u32> = unsafe {
            std::slice::from_raw_parts(tokens_ptr, token_count as usize)
                .iter()
                .map(|&t| t as u32)
                .collect()
        };

        // Free C++ allocated memory
        unsafe {
            bitnet_cpp_free_tokens(tokens_ptr);
        }

        debug!("Tokenized {} characters into {} tokens", text.len(), tokens.len());

        Ok(tokens)
    }

    #[instrument(skip(self, tokens))]
    async fn detokenize(&self, tokens: &[u32]) -> ImplementationResult<String> {
        let handle = self.handle.as_ref().ok_or(ImplementationError::ModelNotLoaded)?.as_ptr();

        // Convert tokens to C array
        let c_tokens: Vec<c_uint> = tokens.iter().map(|&t| t as c_uint).collect();
        let mut text_ptr: *mut c_char = std::ptr::null_mut();

        // Detokenize via FFI
        let result = unsafe {
            bitnet_cpp_detokenize(
                handle,
                c_tokens.as_ptr(),
                c_tokens.len() as c_uint,
                &mut text_ptr,
            )
        };

        if result != 0 {
            return Err(ImplementationError::TokenizationError {
                message: format!("C++ detokenization failed with code: {}", result),
            });
        }

        if text_ptr.is_null() {
            return Ok(String::new());
        }

        // Convert C string to Rust String
        let text = unsafe { CStr::from_ptr(text_ptr).to_string_lossy().to_string() };

        // Free C++ allocated memory
        unsafe {
            bitnet_cpp_free_string(text_ptr);
        }

        debug!("Detokenized {} tokens into {} characters", tokens.len(), text.len());

        Ok(text)
    }

    #[instrument(skip(self, tokens, config))]
    async fn inference(
        &self,
        tokens: &[u32],
        config: &InferenceConfig,
    ) -> ImplementationResult<InferenceResult> {
        let start_time = Instant::now();
        let handle = self.handle.as_ref().ok_or(ImplementationError::ModelNotLoaded)?.as_ptr();

        // Convert tokens and config
        let c_tokens: Vec<c_uint> = tokens.iter().map(|&t| t as c_uint).collect();
        let cpp_config = self.convert_inference_config(config);
        let mut cpp_result = CppInferenceResult {
            tokens: std::ptr::null_mut(),
            token_count: 0,
            text: std::ptr::null(),
            duration_ms: 0,
            memory_usage: 0,
        };

        // Run inference via FFI
        let result = unsafe {
            bitnet_cpp_inference(
                handle,
                c_tokens.as_ptr(),
                c_tokens.len() as c_uint,
                &cpp_config,
                &mut cpp_result,
            )
        };

        if result != 0 {
            return Err(ImplementationError::InferenceError {
                message: format!("C++ inference failed with code: {}", result),
            });
        }

        // Convert result
        let output_tokens = if cpp_result.tokens.is_null() || cpp_result.token_count == 0 {
            Vec::new()
        } else {
            unsafe {
                std::slice::from_raw_parts(cpp_result.tokens, cpp_result.token_count as usize)
                    .iter()
                    .map(|&t| t as u32)
                    .collect()
            }
        };

        let output_text = if cpp_result.text.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(cpp_result.text).to_string_lossy().to_string() }
        };

        let duration = start_time.elapsed();
        let token_count = tokens.len() + output_tokens.len();

        // Create result
        let inference_result = InferenceResult {
            tokens: output_tokens,
            text: output_text,
            probabilities: None, // C++ implementation doesn't expose probabilities
            logits: None,        // C++ implementation doesn't expose logits
            duration,
            memory_usage: cpp_result.memory_usage as u64,
            token_count,
        };

        // Free C++ allocated memory
        unsafe {
            if !cpp_result.tokens.is_null() {
                bitnet_cpp_free_tokens(cpp_result.tokens);
            }
            if !cpp_result.text.is_null() {
                bitnet_cpp_free_string(cpp_result.text as *mut c_char);
            }
        }

        info!(
            "Inference completed: {} tokens generated in {:?}",
            inference_result.tokens.len(),
            duration
        );

        Ok(inference_result)
    }
    fn get_metrics(&self) -> PerformanceMetrics {
        if let Some(handle) = &self.handle {
            let cpp_metrics = unsafe { bitnet_cpp_get_metrics(handle.as_ptr()) };
            self.convert_performance_metrics(cpp_metrics)
        } else {
            PerformanceMetrics::new()
        }
    }

    fn reset_metrics(&mut self) {
        if let Some(handle) = &self.handle {
            unsafe {
                bitnet_cpp_reset_metrics(handle.as_ptr());
            }
        }
    }

    fn get_resource_info(&self) -> ResourceInfo {
        self.resource_info.try_read().map(|info| info.clone()).unwrap_or(ResourceInfo {
            memory_usage: self.get_memory_usage(),
            file_handles: self.get_file_handle_count(),
            thread_count: self.get_thread_count(),
            gpu_memory: Some(self.get_gpu_memory_usage()),
        })
    }

    #[instrument(skip(self))]
    async fn cleanup(&mut self) -> ImplementationResult<()> {
        info!("Cleaning up C++ implementation");

        // Unload model if loaded
        if self.is_model_loaded() {
            self.unload_model().await?;
        }

        // Cleanup FFI handle
        if let Some(handle) = self.handle.take() {
            let result = unsafe { bitnet_cpp_cleanup(handle.as_ptr()) };
            if result != 0 {
                warn!("C++ cleanup failed with code: {}", result);
            }
            unsafe {
                bitnet_cpp_destroy(handle.as_ptr());
            }
        }

        // Reset state
        self.model_info = None;
        self.is_available = false;

        // Reset resource info
        let mut resource_info = self.resource_info.write().await;
        *resource_info =
            ResourceInfo { memory_usage: 0, file_handles: 0, thread_count: 1, gpu_memory: None };

        Ok(())
    }

    fn get_capabilities(&self) -> ImplementationCapabilities {
        self.capabilities.clone()
    }
}

///
// Factory for creating C++ implementation instances
pub struct CppImplementationFactory {
    binary_path: Option<PathBuf>,
}

impl CppImplementationFactory {
    pub fn new() -> Self {
        Self { binary_path: None }
    }

    pub fn with_binary_path(binary_path: PathBuf) -> Self {
        Self { binary_path: Some(binary_path) }
    }
}

impl Default for CppImplementationFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImplementationFactory for CppImplementationFactory {
    async fn create(&self) -> ImplementationResult<Box<dyn BitNetImplementation>> {
        let mut implementation = CppImplementation::new();

        // Set binary path if provided
        if let Some(path) = &self.binary_path {
            implementation.binary_path = Some(path.clone());
        }

        implementation.initialize(None).await?;
        Ok(Box::new(implementation))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_cpp_implementation_creation() {
        let implementation = CppImplementation::new();
        assert_eq!(implementation.implementation_name(), "BitNet.cpp");
        assert!(!implementation.is_model_loaded());
    }

    #[tokio::test]
    async fn test_cpp_implementation_capabilities() {
        let implementation = CppImplementation::new();
        let capabilities = implementation.get_capabilities();

        assert!(capabilities.supports_streaming);
        assert!(!capabilities.supports_batching); // C++ doesn't support batching yet
        assert!(capabilities.supports_gpu);
        assert!(capabilities.supports_quantization);
        assert!(capabilities.supported_formats.contains(&ModelFormat::GGUF));
    }

    #[tokio::test]
    async fn test_cpp_implementation_binary_discovery() {
        let mut implementation = CppImplementation::new();

        // This will likely fail in test environment, but should not panic
        let result = implementation.discover_binary().await;
        // We don't assert success because the binary might not be available
        match result {
            Ok(_) => {
                assert!(implementation.binary_path.is_some());
                assert!(implementation.is_available);
            }
            Err(_) => {
                assert!(implementation.binary_path.is_none());
                assert!(!implementation.is_available);
            }
        }
    }

    #[tokio::test]
    async fn test_cpp_implementation_config_conversion() {
        let implementation = CppImplementation::new();

        let config = InferenceConfig {
            max_tokens: 50,
            temperature: 0.8,
            top_p: 0.95,
            top_k: Some(40),
            repetition_penalty: 1.1,
            stop_tokens: vec!["</s>".to_string()],
            seed: Some(42),
        };

        let cpp_config = implementation.convert_inference_config(&config);
        assert_eq!(cpp_config.max_tokens, 50);
        assert_eq!(cpp_config.temperature, 0.8);
        assert_eq!(cpp_config.top_p, 0.95);
        assert_eq!(cpp_config.top_k, 40);
        assert_eq!(cpp_config.repetition_penalty, 1.1);
        assert_eq!(cpp_config.seed, 42);
    }

    #[tokio::test]
    async fn test_cpp_implementation_config_conversion_defaults() {
        let implementation = CppImplementation::new();

        let config = InferenceConfig {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            repetition_penalty: 1.0,
            stop_tokens: vec![],
            seed: None,
        };

        let cpp_config = implementation.convert_inference_config(&config);
        assert_eq!(cpp_config.max_tokens, 100);
        assert_eq!(cpp_config.temperature, 0.7);
        assert_eq!(cpp_config.top_p, 0.9);
        assert_eq!(cpp_config.top_k, -1); // Disabled
        assert_eq!(cpp_config.repetition_penalty, 1.0);
        assert_eq!(cpp_config.seed, -1); // Random
    }

    #[tokio::test]
    async fn test_cpp_implementation_metrics() {
        let implementation = CppImplementation::new();
        let metrics = implementation.get_metrics();

        // Should return default metrics when no handle is available
        assert_eq!(metrics.model_load_time, Duration::ZERO);
        assert_eq!(metrics.inference_time, Duration::ZERO);
        assert_eq!(metrics.tokens_per_second, 0.0);
    }

    #[tokio::test]
    async fn test_cpp_implementation_resource_info() {
        let implementation = CppImplementation::new();
        let resource_info = implementation.get_resource_info();

        assert!(resource_info.memory_usage >= 0);
        assert!(resource_info.thread_count > 0);
        assert_eq!(resource_info.file_handles, 1); // Just binary handle
    }

    #[tokio::test]
    async fn test_cpp_implementation_cleanup() {
        let mut implementation = CppImplementation::new();
        let result = implementation.cleanup().await;
        assert!(result.is_ok());
        assert!(!implementation.is_model_loaded());
        assert!(!implementation.is_available);
    }

    #[tokio::test]
    async fn test_cpp_implementation_factory() {
        let factory = CppImplementationFactory::new();
        // This will likely fail in test environment due to missing binary/FFI
        let result = factory.create().await;

        match result {
            Ok(implementation) => {
                assert_eq!(implementation.implementation_name(), "BitNet.cpp");
            }
            Err(e) => {
                // Expected in test environment
                assert!(matches!(e, ImplementationError::NotAvailable { .. }));
            }
        }
    }

    #[tokio::test]
    async fn test_cpp_implementation_factory_with_binary() {
        let temp_dir = TempDir::new().unwrap();
        let binary_path = temp_dir.path().join("fake_bitnet");

        let factory = CppImplementationFactory::with_binary_path(binary_path.clone());
        let result = factory.create().await;

        // Should fail because the fake binary doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_model_not_loaded_errors() {
        let implementation = CppImplementation::new();

        // These operations should fail because no model is loaded and no handle exists
        assert!(!implementation.is_model_loaded());
        assert!(implementation.get_model_info().is_none());
    }

    #[test]
    fn test_cpp_model_info_conversion() {
        let implementation = CppImplementation::new();
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.gguf");

        let cpp_info = CppModelInfo {
            name: std::ptr::null(), // Will use "unknown"
            format: 0,              // GGUF
            size_bytes: 1024,
            parameter_count: 1000000,
            context_length: 2048,
            vocabulary_size: 50000,
        };

        let model_info = implementation.convert_model_info(cpp_info, &model_path);

        assert_eq!(model_info.name, "unknown");
        assert_eq!(model_info.format, ModelFormat::GGUF);
        assert_eq!(model_info.size_bytes, 1024);
        assert_eq!(model_info.parameter_count, Some(1000000));
        assert_eq!(model_info.context_length, Some(2048));
        assert_eq!(model_info.vocabulary_size, Some(50000));
        assert_eq!(model_info.architecture, Some("BitNet".to_string()));
        assert!(model_info.metadata.contains_key("implementation"));
        assert_eq!(model_info.metadata.get("implementation"), Some(&"BitNet.cpp".to_string()));
    }

    #[test]
    fn test_cpp_performance_metrics_conversion() {
        let implementation = CppImplementation::new();

        let cpp_metrics = CppPerformanceMetrics {
            model_load_time_ms: 1000,
            tokenization_time_ms: 50,
            inference_time_ms: 200,
            peak_memory: BYTES_PER_MB, // 1MB
            tokens_per_second: 100.5,
        };

        let metrics = implementation.convert_performance_metrics(cpp_metrics);

        assert_eq!(metrics.model_load_time, Duration::from_millis(1000));
        assert_eq!(metrics.tokenization_time, Duration::from_millis(50));
        assert_eq!(metrics.inference_time, Duration::from_millis(200));
        assert_eq!(metrics.total_time, Duration::from_millis(1250));
        assert_eq!(metrics.peak_memory, BYTES_PER_MB);
        assert_eq!(metrics.tokens_per_second, 100.5);
    }

    #[test]
    fn test_resource_tracking() {
        let implementation = CppImplementation::new();

        let memory = implementation.get_memory_usage();
        assert!(memory > 0);

        let file_handles = implementation.get_file_handle_count();
        assert_eq!(file_handles, 1); // No model loaded

        let thread_count = implementation.get_thread_count();
        assert!(thread_count > 0);
        assert!(thread_count <= 8); // Capped at 8

        let gpu_memory = implementation.get_gpu_memory_usage();
        assert_eq!(gpu_memory, 0); // Placeholder implementation
    }
}
