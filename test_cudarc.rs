// Test file to understand cudarc API
use cudarc::prelude::*;

fn main() {
    println!("Testing cudarc API...");
    
    // Try to create a device
    match CudaDevice::new(0) {
        Ok(device) => {
            println!("CUDA device created successfully");
            println!("Device name: {:?}", device.name());
        }
        Err(e) => {
            println!("Failed to create CUDA device: {}", e);
        }
    }
}