//! Wave 24 snapshot tests — quantization configs, SIMD capabilities,
//! int4/int8 quantization params, TL1/TL2 configs, device-aware quantizer
//! types, calibration data, and nibble-packed storage.
//!
//! Covers: SimdCapabilities, QuantizationStrategy, TL1Config, TL2Config,
//! LookupTable, VectorizedLookupTable, Int8QuantConfig, CalibrationMethod,
//! Int8QuantParams, QuantError, Int4QuantConfig, Int4QuantParams, NibblePacked,
//! ToleranceConfig, QuantizedTensor (device_aware), AccuracyReport,
//! CPUQuantizer, CalibrationData, I2SLayout.

use bitnet_quantization::simd_ops::{QuantizationStrategy, SimdCapabilities};

// ============================================================================
// Section 1 — SIMD capabilities & strategy
// ============================================================================

#[test]
fn w24_simd_capabilities_detect() {
    let caps = SimdCapabilities::detect();
    // Snapshot whatever the build host reports — pins CI's view.
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn w24_quantization_strategy_all_variants() {
    let variants: Vec<QuantizationStrategy> = vec![
        QuantizationStrategy::Scalar,
        QuantizationStrategy::SSE4_1,
        QuantizationStrategy::AVX2,
        QuantizationStrategy::AVX512,
        QuantizationStrategy::NEON,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn w24_simd_best_strategy() {
    let caps = SimdCapabilities::detect();
    let strategy = caps.best_quantization_strategy();
    insta::assert_debug_snapshot!(strategy);
}

// ============================================================================
// Section 2 — TL1 config and lookup table
// ============================================================================

use bitnet_quantization::tl1::{LookupTable, TL1Config};

#[test]
fn w24_tl1_config_default() {
    let cfg = TL1Config::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_tl1_config_custom() {
    let cfg = TL1Config {
        block_size: 128,
        lookup_table_size: 512,
        use_asymmetric: true,
        precision_bits: 4,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_lookup_table_symmetric_2bit() {
    let lut = LookupTable::new(-1.0, 1.0, 2, false);
    // Test round-trip: quantize then dequantize
    let vals = [-1.0_f32, -0.5, 0.0, 0.5, 1.0];
    let round_tripped: Vec<(f32, i8, f32)> = vals
        .iter()
        .map(|&v| {
            let q = lut.quantize(v);
            let dq = lut.dequantize(q);
            (v, q, dq)
        })
        .collect();
    insta::assert_debug_snapshot!(round_tripped);
}

#[test]
fn w24_lookup_table_asymmetric_2bit() {
    let lut = LookupTable::new(-0.5, 1.5, 2, true);
    let vals = [-0.5_f32, 0.0, 0.5, 1.0, 1.5];
    let round_tripped: Vec<(f32, i8, f32)> = vals
        .iter()
        .map(|&v| {
            let q = lut.quantize(v);
            let dq = lut.dequantize(q);
            (v, q, dq)
        })
        .collect();
    insta::assert_debug_snapshot!(round_tripped);
}

// ============================================================================
// Section 3 — TL2 config and vectorized lookup table
// ============================================================================

use bitnet_quantization::tl2::{TL2Config, VectorizedLookupTable};

#[test]
fn w24_tl2_config_default() {
    let cfg = TL2Config::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_tl2_config_custom_no_avx() {
    let cfg = TL2Config {
        block_size: 64,
        lookup_table_size: 128,
        use_avx512: false,
        use_avx2: false,
        precision_bits: 2,
        vectorized_tables: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_vectorized_lut_round_trip() {
    let lut = VectorizedLookupTable::new(-1.0, 1.0, 2);
    let vals = [-1.0_f32, -0.5, 0.0, 0.5, 1.0];
    let round_tripped: Vec<(f32, i8, f32)> = vals
        .iter()
        .map(|&v| {
            let q = lut.quantize(v);
            let dq = lut.dequantize(q);
            (v, q, dq)
        })
        .collect();
    insta::assert_debug_snapshot!(round_tripped);
}

#[test]
fn w24_vectorized_lut_table_lengths() {
    let lut = VectorizedLookupTable::new(-2.0, 2.0, 4);
    insta::assert_snapshot!(format!(
        "forward_len={} reverse_len={}",
        lut.forward_len(),
        lut.reverse_len()
    ));
}

// ============================================================================
// Section 4 — Int8 quantization
// ============================================================================

use bitnet_quantization::int8_quant::{
    CalibrationMethod, Int8QuantConfig, Int8QuantParams, QuantError,
};

#[test]
fn w24_int8_config_default() {
    let cfg = Int8QuantConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_int8_config_asymmetric_percentile() {
    let cfg = Int8QuantConfig {
        per_channel: false,
        symmetric: false,
        calibration_method: CalibrationMethod::Percentile(99.9),
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_int8_config_mse() {
    let cfg = Int8QuantConfig {
        per_channel: true,
        symmetric: true,
        calibration_method: CalibrationMethod::MSE,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_calibration_method_all_variants() {
    let methods: Vec<CalibrationMethod> = vec![
        CalibrationMethod::MinMax,
        CalibrationMethod::Percentile(99.0),
        CalibrationMethod::Percentile(99.9),
        CalibrationMethod::MSE,
    ];
    insta::assert_debug_snapshot!(methods);
}

#[test]
fn w24_int8_quant_params_sample() {
    let params = Int8QuantParams {
        scales: vec![0.015, 0.023],
        zero_points: vec![0, -3],
        min_vals: vec![-1.92, -2.87],
        max_vals: vec![1.88, 2.95],
    };
    insta::assert_debug_snapshot!(params);
}

#[test]
fn w24_quant_error_sample() {
    let err = QuantError { max_abs_error: 0.032, mean_abs_error: 0.008, rmse: 0.012, snr_db: 38.5 };
    insta::assert_debug_snapshot!(err);
}

// ============================================================================
// Section 5 — Int4 quantization
// ============================================================================

use bitnet_quantization::int4_quant::{Int4QuantConfig, Int4QuantParams, NibblePacked};

#[test]
fn w24_int4_config_default() {
    let cfg = Int4QuantConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_int4_config_asymmetric() {
    let cfg = Int4QuantConfig { group_size: 64, symmetric: false, block_wise: false };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_int4_quant_params_sample() {
    let params = Int4QuantParams {
        scales: vec![0.125, 0.25],
        zero_points: vec![8, 7],
        group_size: 128,
        num_groups: 2,
        symmetric: true,
    };
    insta::assert_debug_snapshot!(params);
}

#[test]
fn w24_nibble_packed_round_trip() {
    let values: Vec<i8> = vec![1, -1, 3, -4, 7, -8, 0, 2];
    let packed = NibblePacked::pack(&values);
    let unpacked = packed.unpack();
    insta::assert_debug_snapshot!((packed.len, &packed.data, &unpacked));
}

#[test]
fn w24_nibble_packed_odd_length() {
    let values: Vec<i8> = vec![5, -3, 2];
    let packed = NibblePacked::pack(&values);
    let unpacked = packed.unpack();
    insta::assert_debug_snapshot!((packed.len, &packed.data, &unpacked));
}

// ============================================================================
// Section 6 — Device-aware quantizer types
// ============================================================================

use bitnet_quantization::device_aware_quantizer::{
    AccuracyReport, CPUQuantizer, QuantizationType as DAQType,
    QuantizedTensor as DAQuantizedTensor, ToleranceConfig,
};

#[test]
fn w24_tolerance_config_default() {
    let cfg = ToleranceConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_tolerance_config_relaxed() {
    let cfg = ToleranceConfig {
        i2s_tolerance: 1e-2,
        tl_tolerance: 5e-2,
        perplexity_tolerance: 0.01,
        strict_validation: false,
    };
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w24_da_quantization_type_all_variants() {
    let types: Vec<DAQType> =
        vec![DAQType::I2S, DAQType::TL1, DAQType::TL2, DAQType::IQ2S, DAQType::FP32];
    insta::assert_debug_snapshot!(types);
}

#[test]
fn w24_da_quantization_type_display() {
    let displays: Vec<String> =
        vec![DAQType::I2S, DAQType::TL1, DAQType::TL2, DAQType::IQ2S, DAQType::FP32]
            .into_iter()
            .map(|t| t.to_string())
            .collect();
    insta::assert_debug_snapshot!(displays);
}

#[test]
fn w24_da_quantized_tensor_i2s() {
    let t = DAQuantizedTensor::new(
        vec![0xAB, 0xCD, 0xEF, 0x01],
        DAQType::I2S,
        vec![2, 16],
        vec![0.5, 0.25],
        32,
    );
    insta::assert_debug_snapshot!(t);
}

#[test]
fn w24_da_quantized_tensor_numel_nbytes() {
    let t = DAQuantizedTensor::new(vec![0; 64], DAQType::TL1, vec![4, 8], vec![1.0], 64);
    insta::assert_snapshot!(format!("numel={} nbytes={}", t.numel(), t.nbytes()));
}

#[test]
fn w24_cpu_quantizer_debug() {
    let q = CPUQuantizer::new(ToleranceConfig::default());
    insta::assert_debug_snapshot!(q);
}

#[test]
fn w24_accuracy_report_i2s() {
    use std::collections::HashMap;
    let report = AccuracyReport {
        quantization_type: DAQType::I2S,
        device: bitnet_common::Device::Cpu,
        max_absolute_error: 0.0012,
        mean_absolute_error: 0.00035,
        relative_error: 0.0008,
        passed: true,
        tolerance: 1e-5,
        metrics: HashMap::from([("snr_db".to_string(), 42.5)]),
    };
    insta::assert_debug_snapshot!(report);
}

// ============================================================================
// Section 7 — Calibration data & I2S layout
// ============================================================================

use bitnet_quantization::i2s::I2SLayout;
use bitnet_quantization::pipeline::CalibrationData;

#[test]
fn w24_calibration_data_three_layers() {
    let cal = CalibrationData {
        min_values: vec![-1.5, -2.0, -0.8],
        max_values: vec![1.5, 2.0, 0.8],
        mean_values: vec![0.01, -0.02, 0.005],
        num_samples: 1000,
    };
    insta::assert_debug_snapshot!(cal);
}

#[test]
fn w24_calibration_data_empty() {
    let cal = CalibrationData {
        min_values: vec![],
        max_values: vec![],
        mean_values: vec![],
        num_samples: 0,
    };
    insta::assert_debug_snapshot!(cal);
}

#[test]
fn w24_i2s_layout_default() {
    let layout = I2SLayout::default();
    insta::assert_snapshot!(format!(
        "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block,
    ));
}

#[test]
fn w24_i2s_layout_block_64() {
    let layout = I2SLayout::with_block_size(64);
    insta::assert_snapshot!(format!(
        "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block,
    ));
}

#[test]
fn w24_i2s_layout_block_256() {
    let layout = I2SLayout::with_block_size(256);
    insta::assert_snapshot!(format!(
        "block_size={} bytes_per_block={} data_bytes={} scale_bytes={}",
        layout.block_size,
        layout.bytes_per_block,
        layout.data_bytes_per_block,
        layout.scale_bytes_per_block,
    ));
}
