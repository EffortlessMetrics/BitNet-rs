use bitnet_common::{Device, QuantizationType};
use bitnet_kernels::DeviceAwareQuantizer;

#[test]
fn memory_usage_updates_after_allocation() {
    let quant = DeviceAwareQuantizer::new(Device::Cpu).expect("create quantizer");

    let input_len = 32; // one block for I2S
    let input = vec![0f32; input_len];
    let mut output = vec![0u8; input_len / 4];
    let mut scales = vec![0f32; input_len / 32];

    // establish baseline memory usage
    quant
        .quantize(&input, &mut output, &mut scales, QuantizationType::I2S)
        .expect("quantize baseline");
    let baseline = quant.get_stats().unwrap().memory_used_bytes;

    // allocate additional memory
    let big_allocation = vec![0u8; 32 * 1024 * 1024];
    std::hint::black_box(&big_allocation);

    // trigger stats update after allocation
    quant
        .quantize(&input, &mut output, &mut scales, QuantizationType::I2S)
        .expect("quantize after alloc");
    let after = quant.get_stats().unwrap().memory_used_bytes;

    assert!(after > baseline, "memory usage should increase after allocation");
}
