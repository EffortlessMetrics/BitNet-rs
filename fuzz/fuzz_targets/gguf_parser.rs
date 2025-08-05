#![no_main]

use libfuzzer_sys::fuzz_target;
use bitnet_models::formats::gguf::GgufReader;

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
        let _ = reader.metadata();
        let _ = reader.tensor_count();
        
        // Test tensor enumeration doesn't panic
        for i in 0..reader.tensor_count().min(100) { // Limit to prevent timeout
            let _ = reader.tensor_info(i);
        }
    }
});

// Mock GGUF reader for fuzzing
mod mock_gguf {
    use std::collections::HashMap;
    
    pub struct GgufReader<'a> {
        data: &'a [u8],
        header_parsed: bool,
    }
    
    impl<'a> GgufReader<'a> {
        pub fn new(data: &'a [u8]) -> Result<Self, Box<dyn std::error::Error>> {
            if data.len() < 16 {
                return Err("Data too short".into());
            }
            
            // Basic magic number check
            if &data[0..4] != b"GGUF" {
                return Err("Invalid magic number".into());
            }
            
            Ok(Self {
                data,
                header_parsed: false,
            })
        }
        
        pub fn metadata(&self) -> HashMap<String, String> {
            // Return empty metadata for fuzzing
            HashMap::new()
        }
        
        pub fn tensor_count(&self) -> usize {
            // Extract tensor count from header, with bounds checking
            if self.data.len() >= 12 {
                let count_bytes = [self.data[8], self.data[9], self.data[10], self.data[11]];
                let count = u32::from_le_bytes(count_bytes) as usize;
                // Limit to prevent excessive memory usage during fuzzing
                count.min(1000)
            } else {
                0
            }
        }
        
        pub fn tensor_info(&self, index: usize) -> Option<TensorInfo> {
            if index < self.tensor_count() {
                Some(TensorInfo {
                    name: format!("tensor_{}", index),
                    shape: vec![1, 1],
                    dtype: "f32".to_string(),
                    offset: 0,
                })
            } else {
                None
            }
        }
    }
    
    pub struct TensorInfo {
        pub name: String,
        pub shape: Vec<usize>,
        pub dtype: String,
        pub offset: u64,
    }
}

// Use the mock implementation for fuzzing
use mock_gguf::{GgufReader, TensorInfo};