use bitnet_common::BitNetTensor;
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
use candle_core::{Device, Tensor as CandleTensor};

#[test]
fn cpu_gpu_parity_quantize_dequantize() {
    let Ok(gpu_device) = Device::cuda_if_available(0) else {
        return;
    };
    let data = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0, 0.3, -0.7, 0.9];
    let candle = CandleTensor::from_slice(&data, &[data.len()], &Device::Cpu).unwrap();
    let tensor = BitNetTensor::new(candle);

    // I2S
    let i2s = I2SQuantizer::new();
    let cpu_q = i2s.quantize_tensor(&tensor, &Device::Cpu).unwrap();
    let gpu_q = i2s.quantize_tensor(&tensor, &gpu_device).unwrap();
    assert_eq!(cpu_q.data, gpu_q.data);
    let cpu_dq = i2s.dequantize_tensor(&cpu_q, &Device::Cpu).unwrap().to_vec().unwrap();
    let gpu_dq = i2s.dequantize_tensor(&gpu_q, &gpu_device).unwrap().to_vec().unwrap();
    assert_eq!(cpu_dq, gpu_dq);

    // TL1
    let tl1 = TL1Quantizer::new();
    let cpu_q = tl1.quantize_tensor(&tensor, &Device::Cpu).unwrap();
    let gpu_q = tl1.quantize_tensor(&tensor, &gpu_device).unwrap();
    assert_eq!(cpu_q.data, gpu_q.data);
    let cpu_dq = tl1.dequantize_tensor(&cpu_q, &Device::Cpu).unwrap().to_vec().unwrap();
    let gpu_dq = tl1.dequantize_tensor(&gpu_q, &gpu_device).unwrap().to_vec().unwrap();
    assert_eq!(cpu_dq, gpu_dq);

    // TL2
    let tl2 = TL2Quantizer::new();
    let cpu_q = tl2.quantize_tensor(&tensor, &Device::Cpu).unwrap();
    let gpu_q = tl2.quantize_tensor(&tensor, &gpu_device).unwrap();
    assert_eq!(cpu_q.data, gpu_q.data);
    let cpu_dq = tl2.dequantize_tensor(&cpu_q, &Device::Cpu).unwrap().to_vec().unwrap();
    let gpu_dq = tl2.dequantize_tensor(&gpu_q, &gpu_device).unwrap().to_vec().unwrap();
    assert_eq!(cpu_dq, gpu_dq);
}
