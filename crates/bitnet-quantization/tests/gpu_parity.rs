use bitnet_common::{BitNetTensor, Result};
use bitnet_quantization::{
    I2SQuantizer, TL1Quantizer, TL2Quantizer,
    utils::{create_tensor_from_f32, extract_f32_data},
};
use candle_core::Device;

fn prepare_devices() -> Option<Device> {
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            return Device::new_cuda(0).ok();
        } else {
            return None;
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

fn sample_tensor(device: &Device) -> Result<BitNetTensor> {
    let data: Vec<f32> = (0..128).map(|i| i as f32 - 64.0).collect();
    let shape = vec![128];
    create_tensor_from_f32(data, &shape, device)
}

#[test]
fn test_i2s_cpu_gpu_parity() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(gpu) = prepare_devices() else {
        return Ok(());
    };
    let t_cpu = sample_tensor(&cpu)?;
    let t_gpu = sample_tensor(&gpu)?;
    let q = I2SQuantizer::new();
    let qt_cpu = q.quantize(&t_cpu, &cpu)?;
    let qt_gpu = q.quantize(&t_gpu, &gpu)?;
    assert_eq!(qt_cpu.data, qt_gpu.data);
    assert_eq!(qt_cpu.scales, qt_gpu.scales);
    let dq_cpu = q.dequantize(&qt_cpu, &cpu)?;
    let dq_gpu = q.dequantize(&qt_gpu, &gpu)?;
    let cpu_vals = extract_f32_data(&dq_cpu)?;
    let gpu_vals = extract_f32_data(&dq_gpu)?;
    assert_eq!(cpu_vals, gpu_vals);
    Ok(())
}

#[test]
fn test_tl1_cpu_gpu_parity() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(gpu) = prepare_devices() else {
        return Ok(());
    };
    let t_cpu = sample_tensor(&cpu)?;
    let t_gpu = sample_tensor(&gpu)?;
    let q = TL1Quantizer::new();
    let qt_cpu = q.quantize(&t_cpu, &cpu)?;
    let qt_gpu = q.quantize(&t_gpu, &gpu)?;
    assert_eq!(qt_cpu.data, qt_gpu.data);
    let dq_cpu = q.dequantize(&qt_cpu, &cpu)?;
    let dq_gpu = q.dequantize(&qt_gpu, &gpu)?;
    let cpu_vals = extract_f32_data(&dq_cpu)?;
    let gpu_vals = extract_f32_data(&dq_gpu)?;
    assert_eq!(cpu_vals, gpu_vals);
    Ok(())
}

#[test]
fn test_tl2_cpu_gpu_parity() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(gpu) = prepare_devices() else {
        return Ok(());
    };
    let t_cpu = sample_tensor(&cpu)?;
    let t_gpu = sample_tensor(&gpu)?;
    let q = TL2Quantizer::new();
    let qt_cpu = q.quantize(&t_cpu, &cpu)?;
    let qt_gpu = q.quantize(&t_gpu, &gpu)?;
    assert_eq!(qt_cpu.data, qt_gpu.data);
    let dq_cpu = q.dequantize(&qt_cpu, &cpu)?;
    let dq_gpu = q.dequantize(&qt_gpu, &gpu)?;
    let cpu_vals = extract_f32_data(&dq_cpu)?;
    let gpu_vals = extract_f32_data(&dq_gpu)?;
    assert_eq!(cpu_vals, gpu_vals);
    Ok(())
}
