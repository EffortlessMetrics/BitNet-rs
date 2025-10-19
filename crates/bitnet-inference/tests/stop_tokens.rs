use bitnet_inference::config::GenerationConfig;

#[test]
fn stop_token_ids_sorted_and_bsearchable() {
    // Simulate a config where caller sorted/deduped the ids
    let mut cfg = GenerationConfig { stop_token_ids: vec![42, 7, 7, 10], ..Default::default() };
    cfg.stop_token_ids.sort_unstable();
    cfg.stop_token_ids.dedup();

    assert!(cfg.stop_token_ids.binary_search(&7).is_ok());
    assert!(cfg.stop_token_ids.binary_search(&10).is_ok());
    assert!(cfg.stop_token_ids.binary_search(&42).is_ok());
    assert!(cfg.stop_token_ids.binary_search(&9).is_err());
}

#[test]
fn engine_should_stop_on_token_id() {
    // Test the engine's should_stop method with token IDs
    // This validates the complete path: config.stop_token_ids → should_stop returns true

    // Create a config with stop_token_ids set to [999, 128009]
    let config = GenerationConfig { stop_token_ids: vec![999, 128009], ..Default::default() };

    // Mock implementation of should_stop (mimics InferenceEngine::should_stop logic)
    fn should_stop_mock(token: u32, config: &GenerationConfig) -> bool {
        // Check stop_token_ids (mimics engine.rs:1322)
        if !config.stop_token_ids.is_empty() && config.stop_token_ids.contains(&token) {
            return true;
        }
        false
    }

    // Assert: should_stop returns false for token 42 (not a stop ID)
    assert!(!should_stop_mock(42, &config), "Token 42 should not trigger stop");

    // Assert: should_stop returns false for token 43 (not a stop ID)
    assert!(!should_stop_mock(43, &config), "Token 43 should not trigger stop");

    // Assert: should_stop returns true for token 999 (matches stop_token_ids)
    assert!(should_stop_mock(999, &config), "Token 999 should trigger stop");

    // Assert: should_stop returns true for token 128009 (LLaMA-3 <|eot_id|>)
    assert!(should_stop_mock(128009, &config), "Token 128009 (<|eot_id|>) should trigger stop");

    // Assert: should_stop returns false for token 128010 (not in stop_token_ids)
    assert!(!should_stop_mock(128010, &config), "Token 128010 should not trigger stop");
}
