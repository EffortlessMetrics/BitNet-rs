#![cfg(feature = "cuda")]

use bitnet_common::QuantizationType;
use bitnet_kernels::{cpu::FallbackKernel, gpu::{CudaKernel, is_cuda_available}, KernelProvider};

fn block_size(q: QuantizationType) -> usize {
    match q {
        QuantizationType::I2S => 32,
        QuantizationType::TL1 => 64,
        QuantizationType::TL2 => 128,
    }
}

#[test]
fn test_quantization_matches_cpu() {
    if !is_cuda_available() {
        eprintln!("CUDA not available, skipping test");
        return;
    }

    let gpu = match CudaKernel::new() {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Failed to create CUDA kernel: {}", e);
            return;
        }
    };
    let cpu = FallbackKernel;

    let size = 256;
    let input: Vec<f32> = (0..size).map(|i| i as f32 / size as f32 - 0.5).collect();

    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let block = block_size(qtype);
        let mut out_gpu = vec![0u8; size / 4];
        let mut scales_gpu = vec![0.0f32; (size + block - 1) / block];
        let mut out_cpu = vec![0u8; size / 4];
        let mut scales_cpu = vec![0.0f32; (size + block - 1) / block];

        cpu.quantize(&input, &mut out_cpu, &mut scales_cpu, qtype).unwrap();
        gpu.quantize(&input, &mut out_gpu, &mut scales_gpu, qtype).unwrap();

        assert_eq!(out_gpu, out_cpu, "Quantized bytes differ for {:?}", qtype);
        for (a, b) in scales_gpu.iter().zip(scales_cpu.iter()) {
            assert!((a - b).abs() < 1e-6, "Scale mismatch for {:?}: {} vs {}", qtype, a, b);
        }
    }
}
