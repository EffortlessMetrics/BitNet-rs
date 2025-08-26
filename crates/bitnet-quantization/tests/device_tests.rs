use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer, QuantizerTrait};
use bitnet_common::{BitNetTensor, Tensor};
use candle_core::{Device, Tensor as CandleTensor};

fn sample_tensor() -> BitNetTensor {
    let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let t = CandleTensor::from_vec(data, (32,), &Device::Cpu).unwrap();
    BitNetTensor::new(t)
}

#[test]
fn test_dequantize_cpu_and_gpu_paths() {
    let tensor = sample_tensor();
    let quantizers: Vec<Box<dyn QuantizerTrait>> = vec![
        Box::new(I2SQuantizer::new()),
        Box::new(TL1Quantizer::new()),
        Box::new(TL2Quantizer::new()),
    ];

    for q in quantizers {
        let q_data = q.quantize_tensor(&tensor).unwrap();
        // CPU path
        let cpu = q.dequantize_tensor(&q_data).unwrap();
        assert_eq!(cpu.shape(), &[32]);

        // GPU path (skip if CUDA unavailable)
        #[cfg(feature = "cuda")]
        if let Ok(cuda) = Device::new_cuda(0) {
            let gpu = q.dequantize_tensor_device(&q_data, &cuda).unwrap();
            assert_eq!(gpu.shape(), &[32]);
            match gpu.inner().device() {
                Device::Cuda(_) => {},
                _ => panic!("tensor not on cuda"),
            }
        }
    }
}
