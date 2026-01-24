use anyhow::Result;
use bitnet_inference::GenerationConfig;
use bitnet_server::batch_engine::{BatchEngine, BatchEngineConfig, BatchRequest};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_batch_filling_with_timeouts() -> Result<()> {
    // 1. Setup BatchEngine with max_batch_size = 2
    let config = BatchEngineConfig {
        max_batch_size: 2,
        batch_timeout: Duration::from_millis(50),
        max_concurrent_batches: 1,
        priority_queue_enabled: false,
        adaptive_batching: false,
        quantization_aware: false,
        simd_optimization: false,
    };
    let engine = BatchEngine::new(config);

    let gen_config = GenerationConfig::default().with_max_tokens(10);

    // 2. Submit Dummy Request to block the engine
    let engine_clone = engine.clone();
    let gc = gen_config.clone();
    let h_dummy = tokio::spawn(async move {
        let req = BatchRequest::new("Dummy".to_string(), gc);
        engine_clone.submit_request(req).await
    });

    // Wait for dummy to start and acquire semaphore
    sleep(Duration::from_millis(20)).await;

    // 3. Submit R1, R2(TO), R3(TO), R4
    let engine_clone = engine.clone();
    let gc = gen_config.clone();
    let h1 = tokio::spawn(async move {
        let req = BatchRequest::new("R1".to_string(), gc);
        engine_clone.submit_request(req).await
    });

    let engine_clone = engine.clone();
    let gc = gen_config.clone();
    let _h2 = tokio::spawn(async move {
        let mut req = BatchRequest::new("R2".to_string(), gc);
        req = req.with_timeout(Duration::from_millis(10));
        engine_clone.submit_request(req).await
    });

    let engine_clone = engine.clone();
    let gc = gen_config.clone();
    let _h3 = tokio::spawn(async move {
        let mut req = BatchRequest::new("R3".to_string(), gc);
        req = req.with_timeout(Duration::from_millis(10));
        engine_clone.submit_request(req).await
    });

    let engine_clone = engine.clone();
    let gc = gen_config.clone();
    let h4 = tokio::spawn(async move {
        let req = BatchRequest::new("R4".to_string(), gc);
        engine_clone.submit_request(req).await
    });

    // 4. Wait for timeouts
    sleep(Duration::from_millis(200)).await;

    // Check if Dummy finished
    let _ = h_dummy.await?;

    // 5. Submit Probe to trigger processing
    let engine_clone = engine.clone();
    let gc = gen_config.clone();
    let _h_probe = tokio::spawn(async move {
        let req = BatchRequest::new("Probe".to_string(), gc);
        engine_clone.submit_request(req).await
    });

    // 6. Verify Results
    let r1 = h1.await??;
    let r4 = h4.await??;

    println!("Batch size for R1: {}", r1.batch_size);
    println!("Batch size for R4: {}", r4.batch_size);

    assert_eq!(r1.batch_size, 2, "Batch size should be 2 (R1 + R4)");
    assert_eq!(r4.batch_id, r1.batch_id, "R4 should be in same batch as R1");

    Ok(())
}
