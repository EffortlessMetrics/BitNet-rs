//! WebAssembly bindings for BitNet 1-bit LLM inference
//!
//! This crate provides WebAssembly bindings for the BitNet inference engine,
//! enabling 1-bit LLM inference in browsers and edge environments with
//! optimized memory usage and JavaScript async/await support.

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

// Re-export main components
pub mod benchmark;
pub mod inference;
pub mod kernels;
pub mod memory;
pub mod model;
pub mod progressive;
pub mod streaming;
pub mod utils;

pub use benchmark::*;
pub use inference::*;
pub use kernels::*;
pub use memory::*;
pub use model::*;
pub use progressive::*;
pub use streaming::*;
pub use utils::*;

// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "debug")]
    set_panic_hook();

    init_logger();
    console_log!("BitNet WASM module initialized");
}
