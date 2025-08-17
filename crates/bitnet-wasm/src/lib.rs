//! WebAssembly bindings for BitNet 1-bit LLM inference
//!
//! This crate provides WebAssembly bindings for the BitNet inference engine,
//! enabling 1-bit LLM inference in browsers and edge environments with
//! optimized memory usage and JavaScript async/await support.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

// Import the `console.log` function from the `console` module
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Define a macro to provide `println!(..)`-style syntax for `console.log` logging
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

// Set up panic hook for better error messages in development
#[cfg(feature = "debug")]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

// Initialize wasm-logger for logging support
pub fn init_logger() {
    wasm_logger::init(wasm_logger::Config::default());
}

// Memory allocator optimization for WebAssembly
#[cfg(feature = "browser")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// For now, comment out modules that have compilation issues on wasm32
// These will need to be fixed properly in a future PR

// Re-export main components
// pub mod benchmark;
// pub mod kernels;
// pub mod memory;
// pub mod utils;

// These modules depend on bitnet-inference, so only include them when the feature is enabled
#[cfg(feature = "inference")]
pub mod inference;
#[cfg(feature = "inference")]
pub mod model;
#[cfg(feature = "inference")]
pub mod progressive;
#[cfg(feature = "inference")]
pub mod streaming;

// pub use benchmark::*;
// pub use kernels::*;
// pub use memory::*;
// pub use utils::*;

#[cfg(feature = "inference")]
pub use inference::*;
#[cfg(feature = "inference")]
pub use model::*;
#[cfg(feature = "inference")]
pub use progressive::*;
#[cfg(feature = "inference")]
pub use streaming::*;

// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "debug")]
    set_panic_hook();

    init_logger();
    console_log!("BitNet WASM module initialized");
}

/// Minimal async entry that compiles on wasm even when the Rust engine is not enabled.
#[wasm_bindgen]
pub async fn generate(prompt: String) -> Result<String, JsValue> {
    generate_impl(prompt).await.map_err(|e| JsValue::from_str(&e))
}

#[cfg(feature = "inference")]
async fn generate_impl(prompt: String) -> Result<String, String> {
    // When inference is enabled, use the real implementation
    // This would use bitnet_inference components
    Err("Inference implementation not yet ready for WASM".to_string())
}

#[cfg(not(feature = "inference"))]
async fn generate_impl(_prompt: String) -> Result<String, String> {
    Ok("bitnet-wasm built without `inference` feature. Enable it with --features inference".into())
}
