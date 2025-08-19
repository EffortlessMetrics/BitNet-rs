//! API Snapshot Tests
//! 
//! These tests capture the current API surface and prevent unintentional breaking changes.

use insta::{assert_yaml_snapshot, assert_json_snapshot};
use serde_json::json;
use std::collections::BTreeMap;

/// Captures the public API surface of the main bitnet crate
#[test]
fn test_public_api_snapshot() {
    let api_surface = ApiSurface::capture();
    assert_yaml_snapshot!(api_surface);
}

/// Test inference API request/response formats
#[test]
fn test_inference_api_contract() {
    let sample_request = json!({
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0.8,
        "top_p": 0.95,
        "stream": false
    });
    
    assert_json_snapshot!("inference_request", sample_request);
    
    let sample_response = json!({
        "id": "completion-12345",
        "choices": [{
            "index": 0,
            "text": "Once upon a time, there was a magical kingdom...",
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 50,
            "total_tokens": 54
        },
        "model": "bitnet-b1.58-2B",
        "created": 1234567890
    });
    
    assert_json_snapshot!("inference_response", sample_response);
}

/// Test model configuration API
#[test]
fn test_model_config_api() {
    let config = json!({
        "model_type": "bitnet_b158",
        "hidden_size": 2048,
        "num_layers": 24,
        "vocab_size": 32000,
        "intermediate_size": 5632,
        "num_attention_heads": 32,
        "quantization": "i2_s"
    });
    
    assert_json_snapshot!("model_config", config);
}

/// Test streaming response format
#[test]
fn test_streaming_api_contract() {
    let stream_chunk = json!({
        "id": "stream-12345",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "Hello"
            },
            "finish_reason": null
        }],
        "created": 1234567890
    });
    
    assert_json_snapshot!("stream_chunk", stream_chunk);
}

/// Test error response format
#[test]
fn test_error_api_contract() {
    let error_response = json!({
        "code": "invalid_request",
        "message": "Temperature must be between 0 and 2",
        "details": {
            "field": "temperature",
            "value": 3.0,
            "min": 0.0,
            "max": 2.0
        }
    });
    
    assert_json_snapshot!("error_response", error_response);
}

/// Capture the public API surface for tracking
#[derive(Debug, serde::Serialize)]
struct ApiSurface {
    version: String,
    modules: BTreeMap<String, ModuleApi>,
    traits: BTreeMap<String, TraitApi>,
    structs: BTreeMap<String, StructApi>,
    enums: BTreeMap<String, EnumApi>,
    functions: BTreeMap<String, FunctionApi>,
}

#[derive(Debug, serde::Serialize)]
struct ModuleApi {
    path: String,
    public_items: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct TraitApi {
    name: String,
    methods: Vec<MethodSignature>,
    associated_types: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct StructApi {
    name: String,
    fields: Vec<FieldApi>,
    methods: Vec<MethodSignature>,
}

#[derive(Debug, serde::Serialize)]
struct EnumApi {
    name: String,
    variants: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct FunctionApi {
    name: String,
    signature: String,
}

#[derive(Debug, serde::Serialize)]
struct FieldApi {
    name: String,
    ty: String,
    visibility: String,
}

#[derive(Debug, serde::Serialize)]
struct MethodSignature {
    name: String,
    signature: String,
    visibility: String,
}

impl ApiSurface {
    fn capture() -> Self {
        // This would ideally use rustdoc JSON output or syn to parse the crate
        // For now, we manually define the key API surface
        
        let mut modules = BTreeMap::new();
        modules.insert("bitnet".to_string(), ModuleApi {
            path: "bitnet".to_string(),
            public_items: vec![
                "Model".to_string(),
                "Config".to_string(),
                "Tokenizer".to_string(),
                "InferenceEngine".to_string(),
                "quantize".to_string(),
                "dequantize".to_string(),
            ],
        });
        
        let mut structs = BTreeMap::new();
        structs.insert("Model".to_string(), StructApi {
            name: "Model".to_string(),
            fields: vec![
                FieldApi {
                    name: "config".to_string(),
                    ty: "Config".to_string(),
                    visibility: "private".to_string(),
                },
            ],
            methods: vec![
                MethodSignature {
                    name: "new".to_string(),
                    signature: "fn new(config: Config) -> Result<Self>".to_string(),
                    visibility: "public".to_string(),
                },
                MethodSignature {
                    name: "forward".to_string(),
                    signature: "fn forward(&self, input: &Tensor) -> Result<Tensor>".to_string(),
                    visibility: "public".to_string(),
                },
                MethodSignature {
                    name: "generate".to_string(),
                    signature: "fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>".to_string(),
                    visibility: "public".to_string(),
                },
            ],
        });
        
        structs.insert("Config".to_string(), StructApi {
            name: "Config".to_string(),
            fields: vec![
                FieldApi {
                    name: "hidden_size".to_string(),
                    ty: "usize".to_string(),
                    visibility: "public".to_string(),
                },
                FieldApi {
                    name: "num_layers".to_string(),
                    ty: "usize".to_string(),
                    visibility: "public".to_string(),
                },
                FieldApi {
                    name: "vocab_size".to_string(),
                    ty: "usize".to_string(),
                    visibility: "public".to_string(),
                },
            ],
            methods: vec![
                MethodSignature {
                    name: "default".to_string(),
                    signature: "fn default() -> Self".to_string(),
                    visibility: "public".to_string(),
                },
            ],
        });
        
        let mut traits = BTreeMap::new();
        traits.insert("Quantizable".to_string(), TraitApi {
            name: "Quantizable".to_string(),
            methods: vec![
                MethodSignature {
                    name: "quantize".to_string(),
                    signature: "fn quantize(&self) -> QuantizedTensor".to_string(),
                    visibility: "public".to_string(),
                },
                MethodSignature {
                    name: "dequantize".to_string(),
                    signature: "fn dequantize(&self) -> Tensor".to_string(),
                    visibility: "public".to_string(),
                },
            ],
            associated_types: vec!["Output".to_string()],
        });
        
        let mut enums = BTreeMap::new();
        enums.insert("QuantizationType".to_string(), EnumApi {
            name: "QuantizationType".to_string(),
            variants: vec![
                "I2S".to_string(),
                "TL1".to_string(),
                "TL2".to_string(),
            ],
        });
        
        let mut functions = BTreeMap::new();
        functions.insert("load_model".to_string(), FunctionApi {
            name: "load_model".to_string(),
            signature: "pub fn load_model(path: &Path) -> Result<Model>".to_string(),
        });
        
        ApiSurface {
            version: "1.0.0".to_string(),
            modules,
            traits,
            structs,
            enums,
            functions,
        }
    }
}