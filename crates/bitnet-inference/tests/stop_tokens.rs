use bitnet_inference::config::GenerationConfig;
#[test]
fn stop_token_ids_sorted_and_bsearchable() {
    let mut cfg = GenerationConfig::default().with_stop_token_ids(vec![42, 7, 7, 10]);
    cfg.stop_token_ids.sort_unstable();
    cfg.stop_token_ids.dedup();
    cfg.rebuild_stop_token_set();
    assert!(cfg.stop_token_ids.binary_search(&7).is_ok());
    assert!(cfg.stop_token_ids.binary_search(&10).is_ok());
    assert!(cfg.stop_token_ids.binary_search(&42).is_ok());
    assert!(cfg.stop_token_ids.binary_search(&9).is_err());
}
#[test]
fn engine_should_stop_on_token_id() {
    let config = GenerationConfig::default().with_stop_token_ids(vec![999, 128009]);
    fn should_stop_mock(token: u32, config: &GenerationConfig) -> bool {
        if config.is_stop_token(token) {
            return true;
        }
        false
    }
    assert!(!should_stop_mock(42, &config), "Token 42 should not trigger stop");
    assert!(!should_stop_mock(43, &config), "Token 43 should not trigger stop");
    assert!(should_stop_mock(999, &config), "Token 999 should trigger stop");
    assert!(should_stop_mock(128009, &config), "Token 128009 (<|eot_id|>) should trigger stop");
    assert!(!should_stop_mock(128010, &config), "Token 128010 should not trigger stop");
}
