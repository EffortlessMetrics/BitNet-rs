use bitnet_inference::GenerationConfig;
use bitnet_server::batch_engine::{BatchEngine, BatchEngineConfig, BatchRequest};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_underfilled_batch_due_to_timeouts() {
    // 1. Setup BatchEngine with small batch size and single concurrency
    let config = BatchEngineConfig {
        max_batch_size: 5,
        max_concurrent_batches: 1,
        batch_timeout: Duration::from_secs(1),
        ..Default::default()
    };
    let engine = BatchEngine::new(config);

    // 2. Occupy the single concurrent slot with a slow request
    let slow_config = GenerationConfig::default().with_max_tokens(10);
    let slow_req = BatchRequest::new("slow".to_string(), slow_config.clone());

    let engine_clone = engine.clone();
    let _handle = tokio::spawn(async move {
        let _ = engine_clone.submit_request(slow_req).await;
    });

    // Give it a moment to start processing and acquire the semaphore
    sleep(Duration::from_millis(50)).await;

    // 3. Fill the queue with a mix of timed-out and valid requests
    let timeout_duration = Duration::from_millis(1);

    // Pattern: [Timeout, Valid, Timeout, Valid, Valid, Valid]
    let mut req1 = BatchRequest::new("timeout1".to_string(), slow_config.clone());
    req1 = req1.with_timeout(timeout_duration);

    let req2 = BatchRequest::new("valid1".to_string(), slow_config.clone());

    let mut req3 = BatchRequest::new("timeout2".to_string(), slow_config.clone());
    req3 = req3.with_timeout(timeout_duration);

    let req4 = BatchRequest::new("valid2".to_string(), slow_config.clone());
    let req5 = BatchRequest::new("valid3".to_string(), slow_config.clone());
    let req6 = BatchRequest::new("valid4".to_string(), slow_config.clone());

    let requests = vec![req1, req2, req3, req4, req5, req6];

    // Submit them (they will just queue up because semaphore is busy)
    for req in requests {
        let e = engine.clone();
        tokio::spawn(async move {
            let _ = e.submit_request(req).await;
        });
    }

    // Wait for timeouts to expire and slow request to *almost* finish
    sleep(Duration::from_millis(150)).await; // Slow request takes ~100ms

    // At this point, slow request should be done (or close to).
    // And the queue is full.
    // BUT no one is processing the queue because previous spawn calls returned early.
    // So we submit a "trigger" request to start the processing loop again.

    let trigger_req = BatchRequest::new("trigger".to_string(), slow_config.clone());
    let e_trigger = engine.clone();
    tokio::spawn(async move {
        let _ = e_trigger.submit_request(trigger_req).await;
    });

    // Wait for batch processing
    // Batch 1 (mixed) takes ~400ms (4 valid requests * 100ms / size ?)
    // simulate_batch_execution: 100ms * batch_size.
    // If buggy: batch size 3 -> 300ms.
    // Then next batch: size 2 (valid4 + trigger) -> 200ms.

    sleep(Duration::from_millis(1000)).await;

    let stats = engine.get_stats().await;
    println!("Stats: {:?}", stats);

    // We expect:
    // Batch 1: Slow (Size 1)
    // Batch 2: Mixed (Buggy: Size 3 [valid1, valid2, valid3])
    // Batch 3: Leftover (Buggy: Size 2 [valid4, trigger])
    // Total batches: 3

    // Fixed:
    // Batch 1: Slow (Size 1)
    // Batch 2: Full (Fixed: Size 5 [valid1, valid2, valid3, valid4, trigger])
    // Total batches: 2
    // Total processing time: 100ms + 500ms = 600ms.
    // Average batch time: 300ms.

    // Buggy:
    // Batch 1: Slow (Size 1)
    // Batch 2: Partial (Size 3) -> 300ms.
    // Total processing time: 400ms.
    // Average batch time: 200ms.

    assert_eq!(
        stats.total_batches_processed, 2,
        "Should have processed exactly 2 batches. Found {}",
        stats.total_batches_processed
    );

    // We expect ~300ms average batch time if optimization works.
    assert!(
        stats.average_batch_time_ms >= 290.0,
        "Average batch time should reflect full batches. Expected ~300ms, found {}",
        stats.average_batch_time_ms
    );
}
