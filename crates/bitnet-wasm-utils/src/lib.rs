//! Shared WebAssembly runtime helpers.

use wasm_bindgen::prelude::*;

/// JavaScript-friendly error type used by WASM wrappers.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct JsError {
    message: String,
}

impl JsError {
    /// Construct a new JavaScript-friendly error.
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    /// Read the error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<JsError> for JsValue {
    fn from(value: JsError) -> Self {
        JsValue::from_str(&value.message)
    }
}

impl From<JsValue> for JsError {
    fn from(value: JsValue) -> Self {
        if let Some(msg) = value.as_string() {
            return Self::new(msg);
        }

        Self::new("Unknown JavaScript error")
    }
}

/// Convert any displayable error to a JavaScript value.
pub fn to_js_error<E: core::fmt::Display>(error: E) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// High-resolution timestamp in milliseconds since navigation start.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn now_ms() -> f64 {
    web_sys::window().and_then(|w| w.performance()).map_or(0.0, |p| p.now())
}

/// Browser/JS runtime user-agent string.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn user_agent() -> String {
    web_sys::window()
        .map(|w| w.navigator())
        .and_then(|n| n.user_agent().ok())
        .unwrap_or_else(|| "unknown".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn js_error_message_is_preserved() {
        let error = JsError::new("bad things");
        assert_eq!(error.message(), "bad things");
    }
}
