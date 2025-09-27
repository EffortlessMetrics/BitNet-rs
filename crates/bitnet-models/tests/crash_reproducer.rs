//! Test crash reproducers to understand the issues before fixing

use bitnet_models::formats::gguf::GgufReader;
use std::fs;

#[test]
fn test_crash_69e8aa7_gguf_malformed_metadata() {
    let crash_file = "/home/steven/code/Rust/BitNet-rs/fuzz/artifacts/gguf_parser/crash-69e8aa7487115a5484cc9c94c0decd84c1361bcb";
    if let Ok(data) = fs::read(crash_file) {
        println!("Testing crash 69e8aa7 with {} bytes", data.len());
        // This should fail gracefully, not panic
        match GgufReader::new(&data) {
            Ok(reader) => {
                println!(
                    "69e8aa7: Parsing succeeded unexpectedly with {} tensors",
                    reader.tensor_count()
                );
                // Try to trigger any secondary crashes
                let _ = reader.metadata_keys();
                for i in 0..reader.tensor_count().min(10) {
                    let _ = reader.get_tensor_info(i as usize);
                }
            }
            Err(e) => println!("69e8aa7: Expected error: {}", e),
        }
    } else {
        println!("Crash file 69e8aa7 not found - skipping");
    }
}

#[test]
fn test_crash_8052f5d_gguf_tensor_overflow() {
    let crash_file = "/home/steven/code/Rust/BitNet-rs/fuzz/artifacts/gguf_parser/crash-8052f5de4a2a64de976c40f34a950131912e678d";
    if let Ok(data) = fs::read(crash_file) {
        println!("Testing crash 8052f5d with {} bytes", data.len());
        // This should fail gracefully, not panic
        match GgufReader::new(&data) {
            Ok(reader) => {
                println!(
                    "8052f5d: Parsing succeeded unexpectedly with {} tensors",
                    reader.tensor_count()
                );
                // Try to trigger any secondary crashes
                let _ = reader.metadata_keys();
                for i in 0..reader.tensor_count().min(10) {
                    let _ = reader.get_tensor_info(i as usize);
                }
            }
            Err(e) => println!("8052f5d: Expected error: {}", e),
        }
    } else {
        println!("Crash file 8052f5d not found - skipping");
    }
}
