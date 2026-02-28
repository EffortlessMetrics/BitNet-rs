#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct RequestInput {
    raw_json: Vec<u8>,
    prompt: Vec<u8>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    repetition_penalty: Option<f32>,
    priority: Option<Vec<u8>>,
    device_pref: Option<Vec<u8>>,
    timeout_ms: Option<u64>,
    model_path: Vec<u8>,
}

fuzz_target!(|input: RequestInput| {
    // Fuzz InferenceRequest deserialization from arbitrary JSON.
    if let Ok(s) = std::str::from_utf8(&input.raw_json) {
        if s.len() <= 4096 {
            let _ = serde_json::from_str::<bitnet_server::InferenceRequest>(s);
            let _ = serde_json::from_str::<bitnet_server::EnhancedInferenceRequest>(s);
            let _ = serde_json::from_str::<bitnet_server::ModelLoadRequest>(s);
        }
    }

    // Construct valid-ish JSON from structured input and parse.
    let prompt = std::str::from_utf8(&input.prompt).unwrap_or("test");
    let model_path = std::str::from_utf8(&input.model_path).unwrap_or("/tmp/model.gguf");

    let mut obj = serde_json::Map::new();
    obj.insert("prompt".into(), serde_json::Value::String(prompt.to_string()));
    if let Some(mt) = input.max_tokens {
        obj.insert("max_tokens".into(), serde_json::json!(mt));
    }
    if let Some(t) = input.temperature {
        if t.is_finite() {
            obj.insert("temperature".into(), serde_json::json!(t));
        }
    }
    if let Some(p) = input.top_p {
        if p.is_finite() {
            obj.insert("top_p".into(), serde_json::json!(p));
        }
    }
    if let Some(k) = input.top_k {
        obj.insert("top_k".into(), serde_json::json!(k));
    }
    if let Some(rp) = input.repetition_penalty {
        if rp.is_finite() {
            obj.insert("repetition_penalty".into(), serde_json::json!(rp));
        }
    }

    let json_str = serde_json::Value::Object(obj.clone()).to_string();
    let _ = serde_json::from_str::<bitnet_server::InferenceRequest>(&json_str);

    // Add enhanced fields and parse as EnhancedInferenceRequest.
    if let Some(ref p) = input.priority {
        if let Ok(ps) = std::str::from_utf8(p) {
            obj.insert("priority".into(), serde_json::Value::String(ps.to_string()));
        }
    }
    if let Some(ref dp) = input.device_pref {
        if let Ok(ds) = std::str::from_utf8(dp) {
            obj.insert("device_preference".into(), serde_json::Value::String(ds.to_string()));
        }
    }
    if let Some(tms) = input.timeout_ms {
        obj.insert("timeout_ms".into(), serde_json::json!(tms));
    }

    let enhanced_str = serde_json::Value::Object(obj).to_string();
    let _ = serde_json::from_str::<bitnet_server::EnhancedInferenceRequest>(&enhanced_str);

    // ModelLoadRequest parsing.
    let model_obj = serde_json::json!({
        "model_path": model_path,
    });
    let _ = serde_json::from_str::<bitnet_server::ModelLoadRequest>(&model_obj.to_string());

    // Edge: empty object, null values.
    let _ = serde_json::from_str::<bitnet_server::InferenceRequest>("{}");
    let _ = serde_json::from_str::<bitnet_server::InferenceRequest>("null");
    let _ = serde_json::from_str::<bitnet_server::InferenceRequest>("[]");
});
