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
