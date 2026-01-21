use bitnet_inference::GenerationConfig;
use bitnet_server::batch_engine::{BatchEngine, BatchEngineConfig, BatchRequest};
use std::time::Duration;
use tokio::time::{sleep, timeout};

#[tokio::test]
async fn test_underfilled_batches_due_to_timeouts() {
    let config = BatchEngineConfig {
        max_batch_size: 10,
        batch_timeout: Duration::from_millis(50),
        max_concurrent_batches: 1,
        quantization_aware: false, // Disable optimization to simplify channel mapping in test
        ..Default::default()
    };

    let engine = BatchEngine::new(config);
    let gen_config = GenerationConfig::default();

    println!("Starting test");

    // 1. Submit a blocker request to occupy the engine
    let blocker_req = BatchRequest::new("blocker".to_string(), gen_config.clone());

    let engine_clone = engine.clone();
    tokio::spawn(async move {
        println!("Submitting blocker");
        let res = engine_clone.submit_request(blocker_req).await;
        println!("Blocker done: {:?}", res.is_ok());
    });

    // Give it a moment to acquire semaphore
    sleep(Duration::from_millis(10)).await;

    // 2. Submit 5 requests with very short timeout (will timeout while in queue)
    for i in 0..5 {
        let req = BatchRequest::new(format!("timeout_{}", i), gen_config.clone())
            .with_timeout(Duration::from_millis(1));

        let engine_clone = engine.clone();
        tokio::spawn(async move {
            let _ = engine_clone.submit_request(req).await;
        });
    }

    // 3. Submit 10 valid requests
    let mut handles = Vec::new();
    for i in 0..10 {
        let req = BatchRequest::new(format!("valid_{}", i), gen_config.clone());
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move { engine_clone.submit_request(req).await });
        handles.push(handle);
    }

    // Sleep enough for blocker to finish and timeouts to expire
    sleep(Duration::from_millis(200)).await;

    // 4. Submit trigger request to wake up processing
    let trigger_req = BatchRequest::new("trigger".to_string(), gen_config.clone());
    let engine_clone = engine.clone();
    let trigger_handle =
        tokio::spawn(async move { engine_clone.submit_request(trigger_req).await });
    handles.push(trigger_handle);

    println!("Collecting results");

    // Collect results with generous timeout
    let mut batch_sizes = Vec::new();
    for (i, handle) in handles.into_iter().enumerate() {
        // Wait up to 5 seconds for each request
        match timeout(Duration::from_secs(5), handle).await {
            Ok(Ok(Ok(result))) => {
                println!("Request {} result: batch_size={}", i, result.batch_size);
                batch_sizes.push(result.batch_size);
            }
            Ok(Ok(Err(e))) => {
                println!("Request {} failed: {}", i, e);
            }
            Ok(Err(_)) => {
                println!("Request {} join error", i);
            }
            Err(_) => {
                println!("Request {} timed out", i);
            }
        }
    }

    println!("Batch sizes: {:?}", batch_sizes);

    let underfilled_count = batch_sizes.iter().filter(|&&s| s == 5).count();
    println!("Underfilled batches count (size 5): {}", underfilled_count);

    // We expect NO underfilled batches of size 5.
    // We expect valid requests to be in a batch of size 10.

    let full_batch_count = batch_sizes.iter().filter(|&&s| s == 10).count();
    println!("Full batches count (size 10): {}", full_batch_count);

    assert_eq!(underfilled_count, 0, "Should NOT have underfilled batches");
    assert!(full_batch_count >= 10, "Should have processed all 10 requests in full batches");
}
