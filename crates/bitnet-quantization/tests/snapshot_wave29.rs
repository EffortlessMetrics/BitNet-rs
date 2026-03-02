//! Wave-29 snapshot tests for bitnet-quantization API surface stability.
//!
//! Pins I2S quantization output, TL1 lookup table contents, QK256 block layout,
//! and dequantization output for known inputs.

use bitnet_common::{QuantizationType, Tensor};
use bitnet_quantization::QuantizedTensor;
use bitnet_quantization::i2s::I2SQuantizer;
use bitnet_quantization::i2s_qk256::{
    QK256_BLOCK, QK256_PACKED_BYTES, code_to_f32, unpack_qk256_block,
};
use bitnet_quantization::tl1::{LookupTable, TL1Config};

// ── I2S quantization of known vector ──────────────────────────────

#[test]
fn i2s_quantize_known_vector() {
    let quantizer = I2SQuantizer::new();
    // Build a tensor from known values [1.0, -0.5, 0.3, -0.8, 0.0] padded to block_size
    let mut values = vec![0.0f32; 32]; // I2S block_size = 32
    values[0] = 1.0;
    values[1] = -0.5;
    values[2] = 0.3;
    values[3] = -0.8;
    values[4] = 0.0;

    let tensor = candle_core::Tensor::from_vec(values, &[32], &candle_core::Device::Cpu).unwrap();
    let bitnet_tensor = bitnet_common::BitNetTensor::new(tensor);

    let quantized = quantizer.quantize(&bitnet_tensor, &candle_core::Device::Cpu).unwrap();
    let info = format!(
        "qtype={:?}\nblock_size={}\nshape={:?}\ndata_len={}\nscales_len={}\nscales={:.4?}\ncompression_ratio={:.2}",
        quantized.qtype,
        quantized.block_size,
        quantized.shape,
        quantized.data.len(),
        quantized.scales.len(),
        quantized.scales,
        quantized.compression_ratio(),
    );
    insta::assert_snapshot!(info);
}

// ── TL1 lookup table contents ─────────────────────────────────────

#[test]
fn tl1_lookup_table_contents() {
    let config = TL1Config::default();
    let table = LookupTable::new(-1.0, 1.0, config.precision_bits, config.use_asymmetric);

    // Sample forward lookups and reverse lookups
    let forward_sample: Vec<i8> = (0..16).map(|i| table.quantize(i as f32 * 0.1 - 0.8)).collect();
    let reverse_sample: Vec<f32> = (-2i8..=2).map(|q| table.dequantize(q)).collect();

    let info = format!(
        "config_block_size={}\nconfig_table_size={}\nconfig_asymmetric={}\nconfig_bits={}\nforward_sample={:?}\nreverse_sample={:.4?}",
        config.block_size,
        config.lookup_table_size,
        config.use_asymmetric,
        config.precision_bits,
        forward_sample,
        reverse_sample,
    );
    insta::assert_snapshot!(info);
}

// ── QK256 block layout for 256-element input ──────────────────────

#[test]
fn qk256_block_layout_256() {
    let layout_info = format!(
        "block_size={}\npacked_bytes={}\nbits_per_element=2\nelements_per_byte=4",
        QK256_BLOCK, QK256_PACKED_BYTES,
    );

    // Verify code-to-float LUT
    let lut: Vec<String> = (0u8..4).map(|c| format!("code{}={:.1}", c, code_to_f32(c))).collect();

    // Create a known packed block and unpack it
    let mut packed = [0u8; QK256_PACKED_BYTES];
    // Encode a recognizable pattern: first 4 elements = codes [0, 1, 2, 3]
    packed[0] = 0b11_10_01_00; // elem0=0, elem1=1, elem2=2, elem3=3
    packed[1] = 0b00_01_10_11; // elem4=3, elem5=2, elem6=1, elem7=0

    let mut codes = [0u8; QK256_BLOCK];
    unpack_qk256_block(&packed, &mut codes);

    let first_8_codes: Vec<u8> = codes[..8].to_vec();
    let first_8_floats: Vec<String> =
        first_8_codes.iter().map(|&c| format!("{:.1}", code_to_f32(c))).collect();

    let output = format!(
        "{}\nlut=[{}]\nfirst_8_codes={:?}\nfirst_8_floats={:?}",
        layout_info,
        lut.join(", "),
        first_8_codes,
        first_8_floats,
    );
    insta::assert_snapshot!(output);
}

// ── Dequantization output for known quantized block ───────────────

#[test]
fn dequantize_known_quantized_block() {
    let quantizer = I2SQuantizer::new();

    // Create a quantized tensor manually with known data
    let packed_data = vec![0b11_10_01_00u8; 8]; // 32 elements: repeating pattern [0,1,2,3]
    let scales = vec![0.5f32]; // single scale for one block
    let shape = vec![32usize];

    let quantized = QuantizedTensor::new_with_params(
        packed_data,
        scales,
        None,
        shape,
        QuantizationType::I2S,
        32,
    );

    let dequantized = quantizer.dequantize_tensor(&quantized).unwrap();
    let dequant_shape = dequantized.shape().to_vec();
    let dequant_data: Vec<f32> = dequantized.to_vec().unwrap();

    // Show first 8 values and summary stats
    let first_8: Vec<String> = dequant_data.iter().take(8).map(|v| format!("{v:.4}")).collect();
    let min = dequant_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = dequant_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let info = format!(
        "shape={:?}\ntotal_elements={}\nfirst_8={:?}\nmin={:.4}\nmax={:.4}",
        dequant_shape,
        dequant_data.len(),
        first_8,
        min,
        max,
    );
    insta::assert_snapshot!(info);
}
