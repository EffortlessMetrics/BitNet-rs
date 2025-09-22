use bitnet_quantization::tl2::TL2Quantizer;
use std::sync::Arc;
use std::thread;

#[test]
fn test_tl2_lookup_table_thread_safety() {
    let quantizer = Arc::new(TL2Quantizer::new());
    let num_threads = 16;
    let min_val = -2.0f32;
    let max_val = 2.0f32;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let quantizer_clone = Arc::clone(&quantizer);
            thread::spawn(move || {
                let _table = quantizer_clone.get_or_create_lookup_table(min_val, max_val);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Implicitly, if we reach this point without panicking, the RwLock worked correctly
}

#[test]
fn test_tl2_lookup_table_concurrent_access() {
    let quantizer = Arc::new(TL2Quantizer::new());
    let num_threads = 32;
    let scales = [(-1.0f32, 1.0f32), (-2.0f32, 2.0f32), (0.0f32, 4.0f32), (-4.0f32, 4.0f32)];

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let quantizer_clone = Arc::clone(&quantizer);
            let (min_val, max_val) = scales[i % scales.len()];
            thread::spawn(move || {
                let table = quantizer_clone.get_or_create_lookup_table(min_val, max_val);
                assert!(table.forward_len() == 256);
                assert!(table.reverse_len() > 0);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}
