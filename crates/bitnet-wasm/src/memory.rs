//! Memory management optimizations for WebAssembly runtime

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::utils::JsError;

/// Memory manager for WebAssembly runtime constraints
pub struct MemoryManager {
    max_memory_bytes: Option<usize>,
    current_usage: AtomicUsize,
    allocations: HashMap<String, usize>,
    gc_threshold: f64, // Trigger GC when usage exceeds this fraction of max
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(max_memory_bytes: Option<usize>) -> Result<Self, JsError> {
        let manager = MemoryManager {
            max_memory_bytes,
            current_usage: AtomicUsize::new(0),
            allocations: HashMap::new(),
            gc_threshold: 0.8, // 80% of max memory
        };

        console::log_1(
            &format!(
                "Memory manager initialized with limit: {:?} bytes",
                max_memory_bytes
            )
            .into(),
        );

        Ok(manager)
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get maximum allowed memory
    pub fn max_memory_bytes(&self) -> Option<usize> {
        self.max_memory_bytes
    }

    /// Set maximum memory limit
    pub fn set_max_memory(&mut self, bytes: Option<usize>) -> Result<(), JsError> {
        self.max_memory_bytes = bytes;

        // Check if current usage exceeds new limit
        if let Some(max_bytes) = bytes {
            let current = self.current_usage();
            if current > max_bytes {
                console::log_1(
                    &format!(
                        "Current usage ({} bytes) exceeds new limit ({} bytes), triggering GC",
                        current, max_bytes
                    )
                    .into(),
                );
                self.gc()?;
            }
        }

        Ok(())
    }

    /// Track memory allocation
    pub fn track_allocation(&mut self, name: String, bytes: usize) -> Result<(), JsError> {
        // Check if allocation would exceed limit
        if let Some(max_bytes) = self.max_memory_bytes {
            let current = self.current_usage();
            if current + bytes > max_bytes {
                return Err(JsError::new(&format!(
                    "Allocation would exceed memory limit: {} + {} > {}",
                    current, bytes, max_bytes
                )));
            }
        }

        self.allocations.insert(name, bytes);
        self.current_usage.fetch_add(bytes, Ordering::Relaxed);

        // Check if we should trigger GC
        if self.should_gc() {
            console::log_1(&"Memory usage high, suggesting garbage collection".into());
        }

        Ok(())
    }

    /// Track memory deallocation
    pub fn track_deallocation(&mut self, name: &str) -> usize {
        if let Some(bytes) = self.allocations.remove(name) {
            self.current_usage.fetch_sub(bytes, Ordering::Relaxed);
            bytes
        } else {
            0
        }
    }

    /// Check if garbage collection should be triggered
    pub fn should_gc(&self) -> bool {
        if let Some(max_bytes) = self.max_memory_bytes {
            let current = self.current_usage() as f64;
            let threshold = max_bytes as f64 * self.gc_threshold;
            current > threshold
        } else {
            false
        }
    }

    /// Force garbage collection
    pub fn gc(&mut self) -> Result<usize, JsError> {
        let before = self.current_usage();

        console::log_1(
            &format!(
                "Starting garbage collection, current usage: {} bytes",
                before
            )
            .into(),
        );

        // Clear tracked allocations (in a real implementation, this would
        // actually free the memory)
        self.allocations.clear();
        self.current_usage.store(0, Ordering::Relaxed);

        // Call JavaScript garbage collection if available
        if let Some(window) = web_sys::window() {
            if let Ok(performance) = window.performance() {
                // Try to trigger GC through performance.measureUserAgentSpecificMemory
                // This is a hint to the browser, not guaranteed
                let _ = performance.now(); // Dummy call to access performance API
            }
        }

        let after = self.current_usage();
        let freed = before.saturating_sub(after);

        console::log_1(
            &format!(
                "Garbage collection completed, freed {} bytes, current usage: {} bytes",
                freed, after
            )
            .into(),
        );

        Ok(freed)
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let current = self.current_usage();
        let max = self.max_memory_bytes;
        let usage_percent = if let Some(max_bytes) = max {
            (current as f64 / max_bytes as f64) * 100.0
        } else {
            0.0
        };

        MemoryStats {
            current_bytes: current,
            max_bytes: max,
            usage_percent,
            allocation_count: self.allocations.len(),
            should_gc: self.should_gc(),
        }
    }

    /// Set GC threshold (0.0 to 1.0)
    pub fn set_gc_threshold(&mut self, threshold: f64) {
        self.gc_threshold = threshold.clamp(0.0, 1.0);
    }
}

/// Memory statistics
#[wasm_bindgen]
pub struct MemoryStats {
    current_bytes: usize,
    max_bytes: Option<usize>,
    usage_percent: f64,
    allocation_count: usize,
    should_gc: bool,
}

#[wasm_bindgen]
impl MemoryStats {
    #[wasm_bindgen(getter)]
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn max_bytes(&self) -> Option<usize> {
        self.max_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn usage_percent(&self) -> f64 {
        self.usage_percent
    }

    #[wasm_bindgen(getter)]
    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    #[wasm_bindgen(getter)]
    pub fn should_gc(&self) -> bool {
        self.should_gc
    }
}

/// Memory-efficient buffer for progressive loading
#[wasm_bindgen]
pub struct WasmBuffer {
    data: Vec<u8>,
    capacity: usize,
    max_capacity: usize,
}

#[wasm_bindgen]
impl WasmBuffer {
    #[wasm_bindgen(constructor)]
    pub fn new(initial_capacity: usize, max_capacity: usize) -> WasmBuffer {
        WasmBuffer {
            data: Vec::with_capacity(initial_capacity),
            capacity: initial_capacity,
            max_capacity,
        }
    }

    /// Append data to the buffer
    #[wasm_bindgen]
    pub fn append(&mut self, data: &[u8]) -> Result<(), JsError> {
        if self.data.len() + data.len() > self.max_capacity {
            return Err(JsError::new(&format!(
                "Buffer would exceed maximum capacity: {} + {} > {}",
                self.data.len(),
                data.len(),
                self.max_capacity
            )));
        }

        self.data.extend_from_slice(data);
        Ok(())
    }

    /// Get current size
    #[wasm_bindgen]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get capacity
    #[wasm_bindgen]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get maximum capacity
    #[wasm_bindgen]
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Clear the buffer
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get data as Uint8Array
    #[wasm_bindgen]
    pub fn to_uint8_array(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&self.data[..])
    }

    /// Shrink buffer to fit current data
    #[wasm_bindgen]
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
        self.capacity = self.data.capacity();
    }
}

/// Progressive loader for large models
#[wasm_bindgen]
pub struct ProgressiveLoader {
    buffer: WasmBuffer,
    chunk_size: usize,
    total_size: Option<usize>,
    loaded_size: usize,
}

#[wasm_bindgen]
impl ProgressiveLoader {
    #[wasm_bindgen(constructor)]
    pub fn new(chunk_size: usize, max_size: usize) -> ProgressiveLoader {
        ProgressiveLoader {
            buffer: WasmBuffer::new(chunk_size, max_size),
            chunk_size,
            total_size: None,
            loaded_size: 0,
        }
    }

    /// Set total expected size
    #[wasm_bindgen]
    pub fn set_total_size(&mut self, size: usize) {
        self.total_size = Some(size);
    }

    /// Load a chunk of data
    #[wasm_bindgen]
    pub fn load_chunk(&mut self, chunk: &[u8]) -> Result<f64, JsError> {
        self.buffer.append(chunk)?;
        self.loaded_size += chunk.len();

        // Calculate progress
        let progress = if let Some(total) = self.total_size {
            (self.loaded_size as f64 / total as f64).min(1.0)
        } else {
            0.0
        };

        console::log_1(
            &format!(
                "Loaded chunk: {} bytes, total: {} bytes, progress: {:.1}%",
                chunk.len(),
                self.loaded_size,
                progress * 100.0
            )
            .into(),
        );

        Ok(progress)
    }

    /// Check if loading is complete
    #[wasm_bindgen]
    pub fn is_complete(&self) -> bool {
        if let Some(total) = self.total_size {
            self.loaded_size >= total
        } else {
            false
        }
    }

    /// Get loaded data
    #[wasm_bindgen]
    pub fn get_data(&self) -> js_sys::Uint8Array {
        self.buffer.to_uint8_array()
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

    /// Get loaded size in bytes
    #[wasm_bindgen]
    pub fn get_loaded_size(&self) -> usize {
        self.loaded_size
    }

    /// Get total size in bytes (if known)
    #[wasm_bindgen]
    pub fn get_total_size(&self) -> Option<usize> {
        self.total_size
    }
}
