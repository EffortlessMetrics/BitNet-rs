//! WebAssembly bindings for BitNet 1-bit LLM inference.
//!
//! This crate provides WebAssembly bindings for the BitNet inference engine,
//! enabling 1-bit LLM inference in browsers and edge environments with
//! optimized memory usage and JavaScript async/await support.
//!
//! ## Cross-platform modules
//!
//! [`core_types`] and [`error`] compile on **all** targets so that shared
//! logic (config validation, serialization) can be unit-tested with plain
//! `cargo test` on the host.
//!
//! ## Wasm-only modules
//!
//! The remaining modules (`callback`, `utils`, `inference`, `model`, etc.)
//! are gated behind `#[cfg(target_arch = "wasm32")]` and require
//! `wasm-pack build` or `cargo build --target wasm32-unknown-unknown`.

// ── Platform-agnostic modules (always compiled) ──────────────────────
pub mod core_types;
pub mod error;

// ── Wasm32-only modules ──────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
pub mod callback;

#[cfg(all(target_arch = "wasm32", feature = "wasm-utils"))]
pub mod utils;

#[cfg(all(target_arch = "wasm32", feature = "inference"))]
pub mod inference;
#[cfg(all(target_arch = "wasm32", feature = "inference"))]
pub mod memory;
#[cfg(all(target_arch = "wasm32", feature = "inference"))]
pub mod model;
#[cfg(all(target_arch = "wasm32", feature = "inference"))]
pub mod progressive;
#[cfg(all(target_arch = "wasm32", feature = "inference"))]
pub mod streaming;

// Re-exports for convenience
#[cfg(all(target_arch = "wasm32", feature = "wasm-utils"))]
pub use utils::*;

#[cfg(all(target_arch = "wasm32", feature = "inference"))]
pub use inference::*;
#[cfg(all(target_arch = "wasm32", feature = "inference"))]
pub use model::*;
#[cfg(all(target_arch = "wasm32", feature = "inference"))]
pub use progressive::*;
#[cfg(all(target_arch = "wasm32", feature = "inference"))]
pub use streaming::*;

// ── Wasm32 entry points ─────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "debug"))]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

#[cfg(target_arch = "wasm32")]
pub fn init_logger() {
    wasm_logger::init(wasm_logger::Config::default());
}

// Only use dlmalloc on actual wasm32 targets to avoid replacing the host allocator.
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

/// Initialize the WASM module (called automatically by wasm-bindgen).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "debug")]
    set_panic_hook();

    init_logger();
    console_log!("BitNet WASM module initialized");
}

/// Minimal async entry that compiles on wasm even when the Rust engine is not enabled.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn generate(prompt: String) -> Result<String, JsValue> {
    generate_impl(prompt).await.map_err(|e| JsValue::from_str(&e))
}

#[cfg(all(target_arch = "wasm32", feature = "inference"))]
async fn generate_impl(prompt: String) -> Result<String, String> {
    let _ = prompt;
    Err("Inference implementation not yet ready for WASM".to_string())
}

#[cfg(all(target_arch = "wasm32", not(feature = "inference")))]
async fn generate_impl(_prompt: String) -> Result<String, String> {
    Ok("bitnet-wasm built without `inference` feature. Enable it with --features inference".into())
}
