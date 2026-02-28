//! Cross-platform error types for bitnet-wasm.
//!
//! On `wasm32`, [`JsError`] re-exports [`wasm_bindgen::JsError`] for direct
//! use in `#[wasm_bindgen]` return types. On native targets a lightweight
//! stand-in is provided so that shared code compiles everywhere.

use std::fmt;

// ── wasm32 path ──────────────────────────────────────────────────────
#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen::JsError;

#[cfg(target_arch = "wasm32")]
pub fn to_js_error<E: fmt::Display>(err: E) -> wasm_bindgen::JsValue {
    wasm_bindgen::JsValue::from_str(&err.to_string())
}

// ── native path ──────────────────────────────────────────────────────
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::fmt;

    /// Stand-in for `wasm_bindgen::JsError` on non-wasm targets.
    #[derive(Debug, Clone)]
    pub struct JsError {
        message: String,
    }

    impl JsError {
        pub fn new(msg: &str) -> Self {
            Self { message: msg.to_string() }
        }

        pub fn message(&self) -> &str {
            &self.message
        }
    }

    impl fmt::Display for JsError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.message)
        }
    }

    impl std::error::Error for JsError {}

    impl From<String> for JsError {
        fn from(s: String) -> Self {
            Self { message: s }
        }
    }

    impl From<&str> for JsError {
        fn from(s: &str) -> Self {
            Self { message: s.to_string() }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::JsError;

// ── Platform-agnostic error wrapper ──────────────────────────────────

/// Convenience error used by platform-agnostic code in [`crate::core_types`].
#[derive(Debug, Clone)]
pub struct WasmError {
    message: String,
}

impl WasmError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { message: msg.into() }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for WasmError {}

impl From<String> for WasmError {
    fn from(s: String) -> Self {
        Self { message: s }
    }
}

impl From<&str> for WasmError {
    fn from(s: &str) -> Self {
        Self { message: s.to_string() }
    }
}

impl From<JsError> for WasmError {
    fn from(e: JsError) -> Self {
        Self { message: e.to_string() }
    }
}
