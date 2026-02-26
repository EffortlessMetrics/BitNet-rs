use bitnet_inference::generation::sampling::{SamplingConfig, SamplingStrategy};
use bitnet_common::{BitNetTensor};
use candle_core::{Tensor, Device};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[tokio::test]
#[ignore]
async fn benchmark_sampling_performance() {
    let vocab_size = 32000;
    let iterations = 1000;

    // Create random logits
    let mut rng = StdRng::seed_from_u64(42);
    let logits_vec: Vec<f32> = (0..vocab_size).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::from_slice(&logits_vec, &[1, 1, vocab_size], &Device::Cpu).unwrap();
    let logits = BitNetTensor::new(tensor);

    let config = SamplingConfig {
        temperature: 0.7,
        top_k: Some(50),
        top_p: Some(0.9),
        repetition_penalty: 1.1,
        do_sample: true,
    };

    let mut strategy = SamplingStrategy::new(config);
    // Add some repetition history
    for i in 0..100 {
        strategy.track_token(i % vocab_size);
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = strategy.sample(&logits, &mut rng).await.unwrap();
    }
    let duration = start.elapsed();

    println!("Total time: {:?}", duration);
    println!("Avg time per sample: {:?}", duration / iterations as u32);
}
