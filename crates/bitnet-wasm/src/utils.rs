#![cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use web_sys::window;

/// High-resolution timestamp (ms since navigation start).
#[wasm_bindgen]
pub fn now_ms() -> f64 {
    window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

/// Browser/JS runtime user agent string (Node will return something minimal).
#[wasm_bindgen]
pub fn user_agent() -> String {
    window()
        .map(|w| w.navigator())
        .and_then(|n| n.user_agent().ok())
        .unwrap_or_else(|| "unknown".into())
}