#![cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

// We rely on Node (default). Do NOT enable `run_in_browser` here.
// Assert runner is Node to make it explicit:

#[wasm_bindgen_test]
fn running_under_node() {
    // In Node, `window` is undefined.
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
    // The stub returns a message about inference being disabled
    assert!(
        msg.contains("without `inference`") || msg.contains("built without"),
        "message should be informative about missing inference feature"
    );
}
