//! Utility functions and types for WebAssembly bindings

use wasm_bindgen::prelude::*;
use js_sys::{Object, Reflect, Array, Uint8Array};
use web_sys::{console, Performance, Window};
use serde::{Serialize, Deserialize};

/// JavaScript-friendly error type
#[wasm_bindgen]
pub struct JsError {
    message: String,
    name: String,
    stack: Option<String>,
}

#[wasm_bindgen]
impl JsError {
    #[wasm_bindgen(constructor)]
    pub fn new(message: &str) -> JsError {
        JsError {
            message: message.to_string(),
            name: "BitNetError".to_string(),
            stack: None,
        }
    }

    /// Create error with custom name
    pub fn with_name(name: &str, message: &str) -> JsError {
        JsError {
            message: message.to_string(),
            name: name.to_string(),
            stack: None,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn stack(&self) -> Option<String> {
        self.stack.clone()
    }

    /// Convert to JavaScript Error object
    #[wasm_bindgen]
    pub fn to_js_error(&self) -> js_sys::Error {
        let error = js_sys::Error::new(&self.message);
        error.set_name(&self.name);
        error
    }
}

impl From<serde_json::Error> for JsError {
    fn from(err: serde_json::Error) -> Self {
        JsError::with_name("SerializationError", &err.to_string())
    }
}

impl From<serde_wasm_bindgen::Error> for JsError {
    fn from(err: serde_wasm_bindgen::Error) -> Self {
        JsError::with_name("SerializationError", &err.to_string())
    }
}

impl From<bitnet_common::BitNetError> for JsError {
    fn from(err: bitnet_common::BitNetError) -> Self {
        JsError::with_name("BitNetError", &err.to_string())
    }
}

/// Convert Rust error to JavaScript error
pub fn to_js_error<E: std::fmt::Display>(err: E) -> JsValue {
    JsError::new(&err.to_string()).to_js_error().into()
}

/// Performance monitoring utilities
#[wasm_bindgen]
pub struct PerformanceMonitor {
    start_time: f64,
    marks: Vec<(String, f64)>,
}

#[wasm_bindgen]
impl PerformanceMonitor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PerformanceMonitor {
        let start_time = get_current_time();
        console::log_1(&format!("Performance monitor started at {:.2}ms", start_time).into());
        
        PerformanceMonitor {
            start_time,
            marks: Vec::new(),
        }
    }

    /// Add a performance mark
    #[wasm_bindgen]
    pub fn mark(&mut self, name: &str) {
        let time = get_current_time();
        let elapsed = time - self.start_time;
        self.marks.push((name.to_string(), elapsed));
        
        console::log_1(&format!("Performance mark '{}': {:.2}ms", name, elapsed).into());
    }

    /// Get elapsed time since start
    #[wasm_bindgen]
    pub fn elapsed(&self) -> f64 {
        get_current_time() - self.start_time
    }

    /// Get all marks as JavaScript object
    #[wasm_bindgen]
    pub fn get_marks(&self) -> JsValue {
        let obj = Object::new();
        for (name, time) in &self.marks {
            let _ = Reflect::set(&obj, &name.as_str().into(), &(*time).into());
        }
        obj.into()
    }

    /// Reset the monitor
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.start_time = get_current_time();
        self.marks.clear();
        console::log_1(&"Performance monitor reset".into());
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current high-resolution time
fn get_current_time() -> f64 {
    if let Some(window) = web_sys::window() {
        if let Ok(performance) = window.performance() {
            return performance.now();
        }
    }
    js_sys::Date::now()
}

/// Logging utilities for WebAssembly
#[wasm_bindgen]
pub struct Logger;

#[wasm_bindgen]
impl Logger {
    /// Log info message
    #[wasm_bindgen]
    pub fn info(message: &str) {
        console::log_1(&format!("[INFO] {}", message).into());
    }

    /// Log warning message
    #[wasm_bindgen]
    pub fn warn(message: &str) {
        console::warn_1(&format!("[WARN] {}", message).into());
    }

    /// Log error message
    #[wasm_bindgen]
    pub fn error(message: &str) {
        console::error_1(&format!("[ERROR] {}", message).into());
    }

    /// Log debug message (only in debug builds)
    #[wasm_bindgen]
    pub fn debug(message: &str) {
        #[cfg(debug_assertions)]
        console::log_1(&format!("[DEBUG] {}", message).into());
    }
}

/// Utility functions for working with JavaScript types
#[wasm_bindgen]
pub struct JsUtils;

#[wasm_bindgen]
impl JsUtils {
    /// Convert Uint8Array to Vec<u8>
    #[wasm_bindgen]
    pub fn uint8_array_to_vec(array: &Uint8Array) -> Vec<u8> {
        array.to_vec()
    }

    /// Convert Vec<u8> to Uint8Array
    #[wasm_bindgen]
    pub fn vec_to_uint8_array(vec: &[u8]) -> Uint8Array {
        Uint8Array::from(vec)
    }

    /// Check if running in a Web Worker
    #[wasm_bindgen]
    pub fn is_web_worker() -> bool {
        js_sys::global()
            .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
            .is_ok()
    }

    /// Check if running in main thread
    #[wasm_bindgen]
    pub fn is_main_thread() -> bool {
        web_sys::window().is_some()
    }

    /// Get available memory (if supported)
    #[wasm_bindgen]
    pub fn get_available_memory() -> Option<f64> {
        if let Some(window) = web_sys::window() {
            if let Ok(navigator) = window.navigator() {
                // Try to access deviceMemory (experimental API)
                if let Ok(memory) = Reflect::get(&navigator, &"deviceMemory".into()) {
                    return memory.as_f64();
                }
            }
        }
        None
    }

    /// Estimate WASM memory usage
    #[wasm_bindgen]
    pub fn get_wasm_memory_usage() -> usize {
        // This is an approximation - actual implementation would need
        // access to WebAssembly.Memory
        0
    }
}

/// Configuration validation utilities
pub fn validate_config<T: Serialize + for<'de> Deserialize<'de>>(
    config: &T,
) -> Result<(), JsError> {
    // Serialize and deserialize to validate structure
    let json = serde_json::to_string(config)
        .map_err(|e| JsError::with_name("ConfigValidationError", &e.to_string()))?;
    
    let _: T = serde_json::from_str(&json)
        .map_err(|e| JsError::with_name("ConfigValidationError", &e.to_string()))?;
    
    Ok(())
}

/// Async utilities for WebAssembly
#[wasm_bindgen]
pub struct AsyncUtils;

#[wasm_bindgen]
impl AsyncUtils {
    /// Create a delay promise
    #[wasm_bindgen]
    pub fn delay(ms: i32) -> js_sys::Promise {
        js_sys::Promise::new(&mut |resolve, _reject| {
            if let Some(window) = web_sys::window() {
                let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                    &resolve,
                    ms,
                );
            }
        })
    }

    /// Check if async/await is supported
    #[wasm_bindgen]
    pub fn supports_async() -> bool {
        // Check if Promise is available
        js_sys::Promise::resolve(&JsValue::NULL).is_instance_of::<js_sys::Promise>()
    }

    /// Create an AbortController for cancellation
    #[wasm_bindgen]
    pub fn create_abort_controller() -> Result<web_sys::AbortController, JsError> {
        web_sys::AbortController::new()
            .map_err(|_| JsError::new("Failed to create AbortController"))
    }
}

/// Memory utilities
#[wasm_bindgen]
pub struct MemoryUtils;

#[wasm_bindgen]
impl MemoryUtils {
    /// Format bytes as human-readable string
    #[wasm_bindgen]
    pub fn format_bytes(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }

    /// Parse human-readable byte string to number
    #[wasm_bindgen]
    pub fn parse_bytes(input: &str) -> Result<usize, JsError> {
        let input = input.trim().to_uppercase();
        
        let (number_part, unit_part) = if let Some(pos) = input.find(char::is_alphabetic) {
            (&input[..pos], &input[pos..])
        } else {
            (input.as_str(), "B")
        };

        let number: f64 = number_part.trim().parse()
            .map_err(|_| JsError::new("Invalid number format"))?;

        let multiplier = match unit_part.trim() {
            "B" => 1,
            "KB" => 1024,
            "MB" => 1024 * 1024,
            "GB" => 1024 * 1024 * 1024,
            "TB" => 1024_usize.pow(4),
            _ => return Err(JsError::new("Unknown unit")),
        };

        Ok((number * multiplier as f64) as usize)
    }
}

/// Feature detection utilities
#[wasm_bindgen]
pub struct FeatureDetection;

#[wasm_bindgen]
impl FeatureDetection {
    /// Check if WebAssembly SIMD is supported
    #[wasm_bindgen]
    pub fn supports_wasm_simd() -> bool {
        // This would need to be implemented with actual SIMD detection
        // For now, return false as a conservative default
        false
    }

    /// Check if WebAssembly threads are supported
    #[wasm_bindgen]
    pub fn supports_wasm_threads() -> bool {
        // Check for SharedArrayBuffer support
        js_sys::global()
            .get(&"SharedArrayBuffer".into())
            .is_function()
    }

    /// Check if WebAssembly bulk memory operations are supported
    #[wasm_bindgen]
    pub fn supports_wasm_bulk_memory() -> bool {
        // This would need actual feature detection
        // For now, assume modern browsers support it
        true
    }

    /// Get WebAssembly feature support summary
    #[wasm_bindgen]
    pub fn get_feature_support() -> JsValue {
        let obj = Object::new();
        let _ = Reflect::set(&obj, &"simd".into(), &Self::supports_wasm_simd().into());
        let _ = Reflect::set(&obj, &"threads".into(), &Self::supports_wasm_threads().into());
        let _ = Reflect::set(&obj, &"bulkMemory".into(), &Self::supports_wasm_bulk_memory().into());
        obj.into()
    }
}