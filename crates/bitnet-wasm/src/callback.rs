//! Streaming token generation via JS function callbacks.
//!
//! Provides [`generate_with_callback`] which invokes a JS function for each
//! token, yielding to the browser event loop between tokens so the UI remains
//! responsive.

#[cfg(target_arch = "wasm32")]
use js_sys::Function;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Generate tokens and invoke `on_token(text, position, is_final)` for each.
///
/// Returns the full generated text once complete. The callback may be used to
/// update the UI incrementally.
///
/// # Arguments
/// * `prompt` – input text
/// * `on_token` – JS callback `(token: string, position: number, isFinal: boolean) => void`
/// * `max_tokens` – maximum tokens to generate (default 100)
/// * `temperature` – sampling temperature (default 0.7)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn generate_with_callback(
    prompt: String,
    on_token: Function,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
) -> Result<String, JsValue> {
    let max_tokens = max_tokens.unwrap_or(100);
    let _temperature = temperature.unwrap_or(0.7);

    let tokens = simulated_tokens(&prompt, max_tokens);
    let mut output = String::new();
    let total = tokens.len();

    for (i, tok) in tokens.iter().enumerate() {
        let is_final = i + 1 == total;
        output.push_str(tok);

        let _ = on_token.call3(
            &JsValue::NULL,
            &JsValue::from_str(tok),
            &JsValue::from_f64(i as f64),
            &JsValue::from_bool(is_final),
        );

        // Yield to the event loop so the browser stays responsive.
        yield_now().await;
    }

    Ok(output)
}

/// Load model bytes from an `ArrayBuffer` and return metadata as JSON.
///
/// This is a thin async wrapper that validates the buffer and returns size
/// information.  Full model parsing is deferred to the inference feature.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn load_model_from_arraybuffer(buffer: js_sys::ArrayBuffer) -> Result<JsValue, JsValue> {
    let byte_len = buffer.byte_length() as usize;
    if byte_len == 0 {
        return Err(JsValue::from_str("ArrayBuffer is empty"));
    }

    let metadata = crate::core_types::ModelMetadata {
        format: detect_format_from_magic(&js_sys::Uint8Array::new(&buffer)),
        size_bytes: byte_len,
        quantization: None,
        num_parameters: None,
    };

    serde_wasm_bindgen::to_value(&metadata).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Peek at the first 4 bytes to guess the file format.
#[cfg(target_arch = "wasm32")]
fn detect_format_from_magic(arr: &js_sys::Uint8Array) -> String {
    if arr.length() < 4 {
        return "unknown".into();
    }
    let mut magic = [0u8; 4];
    arr.slice(0, 4).copy_to(&mut magic);
    match &magic {
        b"GGUF" => "gguf".into(),
        _ => "unknown".into(),
    }
}

// ── helpers ──────────────────────────────────────────────────────────

/// Simulated token list (placeholder until real inference is wired in).
#[cfg(target_arch = "wasm32")]
fn simulated_tokens(prompt: &str, max_tokens: usize) -> Vec<String> {
    let base: Vec<&str> =
        vec![" The", " answer", " to", " \"", prompt, "\"", " is", " being", " computed", "."];
    base.into_iter().map(String::from).take(max_tokens).collect()
}

/// Yield to the JS event loop via a zero-delay `setTimeout` promise.
#[cfg(target_arch = "wasm32")]
async fn yield_now() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        // Use global `setTimeout` via js_sys to avoid depending on web_sys::Window
        // (which is unavailable in workers / Node).
        let _ = js_sys::eval("setTimeout").ok().and_then(|f| f.dyn_into::<Function>().ok()).map(
            |set_timeout| {
                let _ = set_timeout.call2(&JsValue::NULL, &resolve, &JsValue::from_f64(0.0));
            },
        );
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}
