//! Progressive loading system for large models in WebAssembly

use wasm_bindgen::prelude::*;
use js_sys::{Promise, Uint8Array, Object, Reflect};
use web_sys::{console, ReadableStream, ReadableStreamDefaultReader};
use wasm_bindgen_futures::JsFuture;
use std::collections::HashMap;

use crate::utils::{JsError, to_js_error};
use crate::memory::{MemoryManager, WasmBuffer};

/// Configuration for progressive loading
#[derive(Debug, Clone)]
#[wasm_bindgen]
pub struct ProgressiveLoadConfig {
    /// Chunk size for loading (default: 64MB)
    chunk_size_bytes: usize,
    /// Maximum concurrent chunks (default: 2)
    max_concurrent_chunks: usize,
    /// Enable compression for chunks
    enable_compression: bool,
    /// Prefetch next chunk while processing current
    enable_prefetch: bool,
    /// Memory limit for buffering chunks
    buffer_memory_limit: usize,
}

#[wasm_bindgen]
impl ProgressiveLoadConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ProgressiveLoadConfig {
        ProgressiveLoadConfig {
            chunk_size_bytes: 64 * 1024 * 1024, // 64MB
            max_concurrent_chunks: 2,
            enable_compression: false, // Disabled by default for compatibility
            enable_prefetch: true,
            buffer_memory_limit: 256 * 1024 * 1024, // 256MB
        }
    }

    #[wasm_bindgen(getter)]
    pub fn chunk_size_bytes(&self) -> usize {
        self.chunk_size_bytes
    }

    #[wasm_bindgen(setter)]
    pub fn set_chunk_size_bytes(&mut self, size: usize) {
        self.chunk_size_bytes = size.max(1024 * 1024); // Minimum 1MB
    }

    #[wasm_bindgen(getter)]
    pub fn max_concurrent_chunks(&self) -> usize {
        self.max_concurrent_chunks
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_concurrent_chunks(&mut self, count: usize) {
        self.max_concurrent_chunks = count.max(1).min(8); // 1-8 chunks
    }

    #[wasm_bindgen(getter)]
    pub fn enable_compression(&self) -> bool {
        self.enable_compression
    }

    #[wasm_bindgen(setter)]
    pub fn set_enable_compression(&mut self, enabled: bool) {
        self.enable_compression = enabled;
    }

    #[wasm_bindgen(getter)]
    pub fn enable_prefetch(&self) -> bool {
        self.enable_prefetch
    }

    #[wasm_bindgen(setter)]
    pub fn set_enable_prefetch(&mut self, enabled: bool) {
        self.enable_prefetch = enabled;
    }

    #[wasm_bindgen(getter)]
    pub fn buffer_memory_limit(&self) -> usize {
        self.buffer_memory_limit
    }

    #[wasm_bindgen(setter)]
    pub fn set_buffer_memory_limit(&mut self, limit: usize) {
        self.buffer_memory_limit = limit;
    }
}

impl Default for ProgressiveLoadConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Progressive model loader for WebAssembly
#[wasm_bindgen]
pub struct ProgressiveLoader {
    config: ProgressiveLoadConfig,
    memory_manager: MemoryManager,
    chunks: HashMap<usize, WasmBuffer>,
    total_size: Option<usize>,
    loaded_size: usize,
    current_chunk: usize,
    loading_complete: bool,
}

#[wasm_bindgen]
impl ProgressiveLoader {
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<ProgressiveLoadConfig>) -> Result<ProgressiveLoader, JsError> {
        let config = config.unwrap_or_default();
        let memory_manager = MemoryManager::new(Some(config.buffer_memory_limit))?;

        Ok(ProgressiveLoader {
            config,
            memory_manager,
            chunks: HashMap::new(),
            total_size: None,
            loaded_size: 0,
            current_chunk: 0,
            loading_complete: false,
        })
    }

    /// Start loading from a URL with progressive chunking
    #[wasm_bindgen]
    pub fn load_from_url(&mut self, url: &str) -> Promise {
        let url = url.to_string();
        let config = self.config.clone();

        wasm_bindgen_futures::future_to_promise(async move {
            Self::load_from_url_impl(url, config).await
                .map(|result| JsValue::from_serde(&result).unwrap_or(JsValue::NULL))
                .map_err(to_js_error)
        })
    }

    /// Load from a ReadableStream with chunked processing
    #[wasm_bindgen]
    pub fn load_from_stream(&mut self, stream: ReadableStream) -> Promise {
        let config = self.config.clone();

        wasm_bindgen_futures::future_to_promise(async move {
            Self::load_from_stream_impl(stream, config).await
                .map(|result| JsValue::from_serde(&result).unwrap_or(JsValue::NULL))
                .map_err(to_js_error)
        })
    }

    /// Load a specific chunk by index
    #[wasm_bindgen]
    pub fn load_chunk(&mut self, chunk_index: usize, data: &Uint8Array) -> Result<f64, JsError> {
        let chunk_data = data.to_vec();
        let chunk_size = chunk_data.len();

        // Check memory constraints
        if chunk_size > self.config.buffer_memory_limit {
            return Err(JsError::new(&format!(
                "Chunk size ({} bytes) exceeds buffer limit ({} bytes)",
                chunk_size, self.config.buffer_memory_limit
            )));
        }

        // Create buffer for this chunk
        let mut buffer = WasmBuffer::new(chunk_size, chunk_size * 2);
        buffer.append(&chunk_data)?;

        // Track memory usage
        self.memory_manager.track_allocation(
            format!("chunk_{}", chunk_index),
            chunk_size,
        )?;

        // Store chunk
        self.chunks.insert(chunk_index, buffer);
        self.loaded_size += chunk_size;

        // Calculate progress
        let progress = if let Some(total) = self.total_size {
            (self.loaded_size as f64 / total as f64).min(1.0)
        } else {
            0.0
        };

        console::log_1(&format!(
            "Loaded chunk {}: {} bytes, total: {} bytes, progress: {:.1}%",
            chunk_index,
            chunk_size,
            self.loaded_size,
            progress * 100.0
        ).into());

        // Trigger GC if memory usage is high
        if self.memory_manager.should_gc() {
            self.memory_manager.gc()?;
        }

        Ok(progress)
    }

    /// Get a loaded chunk by index
    #[wasm_bindgen]
    pub fn get_chunk(&self, chunk_index: usize) -> Option<Uint8Array> {
        self.chunks.get(&chunk_index).map(|buffer| buffer.to_uint8_array())
    }

    /// Check if a chunk is loaded
    #[wasm_bindgen]
    pub fn is_chunk_loaded(&self, chunk_index: usize) -> bool {
        self.chunks.contains_key(&chunk_index)
    }

    /// Get loading progress (0.0 to 1.0)
    #[wasm_bindgen]
    pub fn get_progress(&self) -> f64 {
        if let Some(total) = self.total_size {
            (self.loaded_size as f64 / total as f64).min(1.0)
        } else {
            0.0
        }
    }

    /// Set total expected size
    #[wasm_bindgen]
    pub fn set_total_size(&mut self, size: usize) {
        self.total_size = Some(size);
    }

    /// Get total expected size
    #[wasm_bindgen]
    pub fn get_total_size(&self) -> Option<usize> {
        self.total_size
    }

    /// Get loaded size in bytes
    #[wasm_bindgen]
    pub fn get_loaded_size(&self) -> usize {
        self.loaded_size
    }

    /// Get number of loaded chunks
    #[wasm_bindgen]
    pub fn get_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Check if loading is complete
    #[wasm_bindgen]
    pub fn is_complete(&self) -> bool {
        self.loading_complete || (
            self.total_size.map_or(false, |total| self.loaded_size >= total)
        )
    }

    /// Mark loading as complete
    #[wasm_bindgen]
    pub fn mark_complete(&mut self) {
        self.loading_complete = true;
        console::log_1(&"Progressive loading marked as complete".into());
    }

    /// Unload a specific chunk to free memory
    #[wasm_bindgen]
    pub fn unload_chunk(&mut self, chunk_index: usize) -> Result<usize, JsError> {
        if let Some(buffer) = self.chunks.remove(&chunk_index) {
            let size = buffer.size();
            self.loaded_size = self.loaded_size.saturating_sub(size);
            
            // Track memory deallocation
            let freed = self.memory_manager.track_deallocation(&format!("chunk_{}", chunk_index));
            
            console::log_1(&format!("Unloaded chunk {}: {} bytes", chunk_index, freed).into());
            Ok(freed)
        } else {
            Ok(0)
        }
    }

    /// Unload all chunks to free memory
    #[wasm_bindgen]
    pub fn unload_all(&mut self) -> Result<usize, JsError> {
        let total_freed = self.loaded_size;
        self.chunks.clear();
        self.loaded_size = 0;
        self.loading_complete = false;
        
        let freed = self.memory_manager.gc()?;
        console::log_1(&format!("Unloaded all chunks: {} bytes", total_freed).into());
        Ok(freed)
    }

    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> JsValue {
        let stats = self.memory_manager.get_stats();
        JsValue::from_serde(&stats).unwrap_or(JsValue::NULL)
    }

    /// Force garbage collection
    #[wasm_bindgen]
    pub fn gc(&mut self) -> Result<usize, JsError> {
        self.memory_manager.gc()
    }
}

impl ProgressiveLoader {
    /// Internal implementation for URL loading
    async fn load_from_url_impl(
        url: String,
        config: ProgressiveLoadConfig,
    ) -> Result<LoadResult, JsError> {
        console::log_1(&format!("Starting progressive load from URL: {}", url).into());

        // Create fetch request
        let window = web_sys::window().ok_or_else(|| JsError::new("No window object"))?;
        let request = web_sys::Request::new_with_str(&url)
            .map_err(|_| JsError::new("Failed to create request"))?;

        // Fetch the response
        let response_promise = window.fetch_with_request(&request);
        let response = JsFuture::from(response_promise).await
            .map_err(|_| JsError::new("Fetch failed"))?;
        
        let response: web_sys::Response = response.dyn_into()
            .map_err(|_| JsError::new("Invalid response"))?;

        if !response.ok() {
            return Err(JsError::new(&format!("HTTP error: {}", response.status())));
        }

        // Get content length if available
        let content_length = response.headers().get("content-length")
            .ok()
            .flatten()
            .and_then(|s| s.parse::<usize>().ok());

        // Get response body as stream
        let body = response.body().ok_or_else(|| JsError::new("No response body"))?;
        
        Self::load_from_stream_impl(body, config).await
    }

    /// Internal implementation for stream loading
    async fn load_from_stream_impl(
        stream: ReadableStream,
        config: ProgressiveLoadConfig,
    ) -> Result<LoadResult, JsError> {
        console::log_1(&"Starting progressive load from stream".into());

        let reader = stream.get_reader().dyn_into::<ReadableStreamDefaultReader>()
            .map_err(|_| JsError::new("Failed to get stream reader"))?;

        let mut total_loaded = 0;
        let mut chunk_index = 0;
        let mut chunks_info = Vec::new();

        loop {
            let read_promise = reader.read();
            let result = JsFuture::from(read_promise).await
                .map_err(|_| JsError::new("Stream read failed"))?;

            let done = Reflect::get(&result, &"done".into())
                .map_err(|_| JsError::new("Failed to get done property"))?
                .as_bool()
                .unwrap_or(true);

            if done {
                break;
            }

            let value = Reflect::get(&result, &"value".into())
                .map_err(|_| JsError::new("Failed to get value property"))?;

            let chunk_array: Uint8Array = value.dyn_into()
                .map_err(|_| JsError::new("Invalid chunk data"))?;

            let chunk_size = chunk_array.length() as usize;
            total_loaded += chunk_size;

            chunks_info.push(ChunkInfo {
                index: chunk_index,
                size: chunk_size,
                offset: total_loaded - chunk_size,
            });

            console::log_1(&format!(
                "Loaded chunk {}: {} bytes (total: {} bytes)",
                chunk_index, chunk_size, total_loaded
            ).into());

            chunk_index += 1;

            // Simulate processing delay for demonstration
            Self::delay(10).await;
        }

        let _ = reader.release_lock();

        Ok(LoadResult {
            total_size: total_loaded,
            chunk_count: chunk_index,
            chunks: chunks_info,
            success: true,
        })
    }

    /// Async delay utility
    async fn delay(ms: i32) {
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            if let Some(window) = web_sys::window() {
                let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                    &resolve,
                    ms,
                );
            }
        });
        let _ = JsFuture::from(promise).await;
    }
}

/// Information about a loaded chunk
#[derive(Debug, Clone, serde::Serialize)]
struct ChunkInfo {
    index: usize,
    size: usize,
    offset: usize,
}

/// Result of progressive loading operation
#[derive(Debug, serde::Serialize)]
struct LoadResult {
    total_size: usize,
    chunk_count: usize,
    chunks: Vec<ChunkInfo>,
    success: bool,
}

/// Progressive loading statistics
#[wasm_bindgen]
pub struct LoadingStats {
    chunks_loaded: usize,
    total_bytes: usize,
    loading_time_ms: f64,
    average_chunk_size: usize,
    throughput_mbps: f64,
}

#[wasm_bindgen]
impl LoadingStats {
    #[wasm_bindgen(getter)]
    pub fn chunks_loaded(&self) -> usize {
        self.chunks_loaded
    }

    #[wasm_bindgen(getter)]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn loading_time_ms(&self) -> f64 {
        self.loading_time_ms
    }

    #[wasm_bindgen(getter)]
    pub fn average_chunk_size(&self) -> usize {
        self.average_chunk_size
    }

    #[wasm_bindgen(getter)]
    pub fn throughput_mbps(&self) -> f64 {
        self.throughput_mbps
    }
}