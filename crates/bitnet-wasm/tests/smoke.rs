#![cfg(feature = "integration-tests")]
#![cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

// We rely on Node (default). Do NOT enable `run_in_browser` here.

#[wasm_bindgen_test]
fn running_under_node() {
    let is_node = js_sys::eval("typeof window === 'undefined'")
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(is_node, "WASM tests are expected to run under Node, not a browser");
}

// Call the stubbed `generate`—works even without the `inference` feature.
#[wasm_bindgen_test]
async fn generate_stub_runs() {
    let out = bitnet_wasm::generate("hello".to_string()).await;
    assert!(out.is_ok(), "generate() should return Ok on stub");
    let msg = out.unwrap();
    assert!(
        msg.contains("without `inference`") || msg.contains("built without"),
        "message should be informative about missing inference feature: {msg}"
    );
}

// Verify the callback entry-point is exported.
#[wasm_bindgen_test]
async fn callback_generate_invokes_js_fn() {
    use wasm_bindgen::JsValue;

    let tokens_collected = js_sys::Array::new();
    let tokens_ref = tokens_collected.clone();

    // Create a JS callback that pushes tokens into the array.
    let cb = wasm_bindgen::closure::Closure::wrap(Box::new(
        move |token: JsValue, _pos: JsValue, _is_final: JsValue| {
            tokens_ref.push(&token);
        },
    )
        as Box<dyn FnMut(JsValue, JsValue, JsValue)>);

    let result = bitnet_wasm::callback::generate_with_callback(
        "test prompt".into(),
        cb.as_ref().unchecked_ref(),
        Some(5),
        None,
    )
    .await;

    cb.forget(); // prevent deallocation
    assert!(result.is_ok(), "generate_with_callback should succeed");
    assert!(tokens_collected.length() > 0, "callback should have been invoked at least once");
}
