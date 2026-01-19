use bitnet_server::batch_engine::{BatchEngine, BatchEngineConfig, BatchRequest};
use bitnet_inference::GenerationConfig;
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[tokio::test]
async fn test_stats_updates() {
    let config = BatchEngineConfig::default();
    let engine = BatchEngine::new(config);

    let gen_config = GenerationConfig::default();
    let request = BatchRequest::new("test".to_string(), gen_config);

    // Submit request
    let result = engine.submit_request(request).await;

    // Check if result is error, print it
    if let Err(e) = &result {
        println!("Submit request failed: {:?}", e);
    }
    assert!(result.is_ok());

    // Check stats
    let stats = engine.get_stats().await;
    println!("Stats: total_requests_processed={}", stats.total_requests_processed);

    // Verify stats are updated correctly (shared across clones)
    assert!(stats.total_requests_processed > 0, "Expected > 0, got {}", stats.total_requests_processed);
}

#[tokio::test]
async fn test_orphaned_requests_bug() {
    let mut config = BatchEngineConfig::default();
    config.max_batch_size = 5;
    let engine = BatchEngine::new(config);

    // 1. Submit 5 timed-out requests
    for i in 0..5 {
        let gen_config = GenerationConfig::default();
        let mut request = BatchRequest::new(format!("timeout_{}", i), gen_config);
        // Set created_at to past to force timeout
        request.created_at = Instant::now() - Duration::from_secs(10);
        request.timeout = Some(Duration::from_secs(1));

        let engine_clone = engine.clone();
        tokio::spawn(async move {
            let _ = engine_clone.submit_request(request).await;
        });
    }

    // Give a moment for them to be queued
    sleep(Duration::from_millis(100)).await;

    // 2. Submit 1 valid request
    let gen_config = GenerationConfig::default();
    let request = BatchRequest::new("valid".to_string(), gen_config);

    println!("Submitting valid request...");
    let result = tokio::time::timeout(Duration::from_millis(500), engine.submit_request(request)).await;

    match result {
        Ok(Ok(_)) => println!("Valid request processed."),
        Ok(Err(e)) => panic!("Request failed: {}", e),
        Err(_) => panic!("Valid request timed out (orphaned). This confirms the bug."),
    }
}
