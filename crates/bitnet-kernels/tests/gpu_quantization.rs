use bitnet_common::QuantizationType;
use bitnet_kernels::{KernelProvider, cpu::FallbackKernel, gpu::CudaKernel};

#[test]
fn test_gpu_quantization_matches_cpu() {
    let kernel = match CudaKernel::new() {
        Ok(k) => k,
        Err(e) => {
            eprintln!("CUDA not available: {}", e);
            return;
        }
    };

    let input: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.1).sin()).collect();
    let cpu_kernel = FallbackKernel;

    for qtype in [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2] {
        let block_size = match qtype {
            QuantizationType::I2S => 32,
            QuantizationType::TL1 => 64,
            QuantizationType::TL2 => 128,
        };
        let num_blocks = (input.len() + block_size - 1) / block_size;
        let mut cpu_out = vec![0u8; input.len() / 4];
        let mut cpu_scales = vec![0f32; num_blocks];
        cpu_kernel.quantize(&input, &mut cpu_out, &mut cpu_scales, qtype).unwrap();

        let mut gpu_out = vec![0u8; input.len() / 4];
        let mut gpu_scales = vec![0f32; num_blocks];
        kernel.quantize(&input, &mut gpu_out, &mut gpu_scales, qtype).unwrap();

        assert_eq!(cpu_out, gpu_out, "quantized output mismatch for {:?}", qtype);
        for (a, b) in cpu_scales.iter().zip(gpu_scales.iter()) {
            assert!((a - b).abs() < 1e-5, "scale mismatch for {:?}: {} vs {}", qtype, a, b);
        }
    }
}
