//! Tests that run on the **host** target (`cargo test -p bitnet-wasm`).
//!
//! These exercise the platform-agnostic modules (`core_types`, `error`) which
//! carry no dependency on wasm-bindgen / web-sys.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use bitnet_wasm::core_types::*;
    use bitnet_wasm::error::{JsError, WasmError};

    // ── error module ─────────────────────────────────────────────────

    #[test]
    fn js_error_new_and_display() {
        let err = JsError::new("something broke");
        assert_eq!(err.to_string(), "something broke");
    }

    #[test]
    fn wasm_error_from_str() {
        let err: WasmError = "oops".into();
        assert_eq!(err.message(), "oops");
    }

    #[test]
    fn wasm_error_from_string() {
        let err: WasmError = String::from("oops").into();
        assert_eq!(err.message(), "oops");
    }

    #[test]
    fn wasm_error_from_js_error() {
        let js = JsError::new("js fail");
        let w: WasmError = js.into();
        assert_eq!(w.message(), "js fail");
    }

    // ── GenerationConfig ─────────────────────────────────────────────

    #[test]
    fn default_config_validates() {
        assert!(GenerationConfig::default().validate().is_ok());
    }

    #[test]
    fn greedy_config_validates() {
        let cfg = GenerationConfig::greedy(8);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.temperature, 0.0);
    }

    #[test]
    fn zero_max_tokens_rejected() {
        let cfg = GenerationConfig { max_new_tokens: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn negative_temp_rejected() {
        let cfg = GenerationConfig { temperature: -1.0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn top_p_above_one_rejected() {
        let cfg = GenerationConfig { top_p: Some(1.01), ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn top_p_none_accepted() {
        let cfg = GenerationConfig { top_p: None, ..Default::default() };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = GenerationConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: GenerationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_new_tokens, 100);
        assert!((back.temperature - 0.7).abs() < f32::EPSILON);
    }

    // ── TokenEvent ───────────────────────────────────────────────────

    #[test]
    fn token_event_serde() {
        let evt = TokenEvent { text: "hi".into(), position: 3, is_final: true };
        let json = serde_json::to_string(&evt).unwrap();
        let back: TokenEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(evt, back);
    }

    // ── GenerationStats ──────────────────────────────────────────────

    #[test]
    fn stats_from_measurements() {
        let stats = GenerationStats::from_measurements(20, 2000.0, 10);
        assert_eq!(stats.tokens_generated, 20);
        assert!((stats.tokens_per_second - 10.0).abs() < 1e-6);
    }

    #[test]
    fn stats_zero_time_no_panic() {
        let stats = GenerationStats::from_measurements(5, 0.0, 0);
        assert_eq!(stats.tokens_per_second, 0.0);
    }

    // ── ModelMetadata ────────────────────────────────────────────────

    #[test]
    fn metadata_default() {
        let m = ModelMetadata::default();
        assert!(m.format.is_empty());
        assert_eq!(m.size_bytes, 0);
    }

    #[test]
    fn metadata_serde() {
        let m = ModelMetadata {
            format: "gguf".into(),
            size_bytes: 42,
            quantization: Some("I2_S".into()),
            num_parameters: Some(2_000_000_000),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }
}
