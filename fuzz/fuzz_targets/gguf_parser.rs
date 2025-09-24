#![no_main]

use bitnet_models::formats::gguf::GgufReader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Skip very small inputs that can't be valid GGUF files
    if data.len() < 16 {
        return;
    }

    // Try to parse the data as a GGUF file
    // This should not panic, even with malformed input
    let _ = GgufReader::new(data);

    // If parsing succeeds, try to read metadata
    if let Ok(reader) = GgufReader::new(data) {
        // Test metadata reading doesn't panic
        let _ = reader.metadata_keys();
        let _ = reader.tensor_count();

        // Test tensor enumeration doesn't panic
        let count = reader.tensor_count() as usize;
        for i in 0..count.min(100) {
            // Limit to prevent timeout
            let _ = reader.get_tensor_info(i);
        }
    }
});

// This fuzz target uses the real GgufReader from bitnet-models
