#![no_main]

use arbitrary::Arbitrary;
use bitnet_models::conversion::{
    ConversionConfig, DType, ModelFormat, QuantMethod, QuantizationSpec, estimate_output_size,
    plan_conversion,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ConversionInput {
    source_path: String,
    source_format_sel: u8,
    target_format_sel: u8,
    target_dtype_sel: u8,
    quant_method_sel: u8,
    num_params: u64,
    has_quant_config: bool,
    per_channel: bool,
    group_size: u16,
    calibration_samples: u16,
}

fn select_format(sel: u8) -> ModelFormat {
    match sel % 4 {
        0 => ModelFormat::SafeTensors,
        1 => ModelFormat::GGUF,
        2 => ModelFormat::ONNX,
        _ => ModelFormat::PyTorch,
    }
}

fn select_dtype(sel: u8) -> DType {
    match sel % 5 {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::Int8,
        _ => DType::Int4,
    }
}

fn select_quant_method(sel: u8) -> QuantMethod {
    match sel % 2 {
        0 => QuantMethod::Symmetric,
        _ => QuantMethod::Asymmetric,
    }
}

fuzz_target!(|input: ConversionInput| {
    let source_format = select_format(input.source_format_sel);
    let target_format = select_format(input.target_format_sel);
    let target_dtype = select_dtype(input.target_dtype_sel);

    // Invariant 1: DType methods must not panic
    let _ = target_dtype.bytes_per_element();
    let _ = target_dtype.is_quantized();
    let _ = target_dtype.display_name();

    // Invariant 2: ModelFormat extension must not panic
    let _ = source_format.extension();
    let _ = target_format.extension();

    // Invariant 3: Build ConversionConfig with optional quantization spec
    let quant_config = if input.has_quant_config {
        Some(QuantizationSpec {
            method: select_quant_method(input.quant_method_sel),
            per_channel: input.per_channel,
            group_size: if input.group_size > 0 { Some(input.group_size as usize) } else { None },
            calibration_samples: input.calibration_samples as usize,
        })
    } else {
        None
    };

    let config = ConversionConfig {
        source_format,
        target_format,
        target_dtype,
        quantization_config: quant_config,
    };

    // Invariant 4: plan_conversion must never panic
    let plan = plan_conversion(&input.source_path, &config);

    // Invariant 5: plan always has steps
    assert!(!plan.steps.is_empty(), "conversion plan must have at least one step");

    // Invariant 6: plan metadata must be consistent
    assert_eq!(plan.config.source_format, source_format);
    assert_eq!(plan.config.target_format, target_format);

    // Invariant 7: estimate_output_size must not panic for any inputs
    let _ = estimate_output_size(input.num_params, &target_dtype);
    let _ = estimate_output_size(0, &target_dtype);
    let _ = estimate_output_size(u64::MAX, &target_dtype);

    // Invariant 8: All DType variants produce non-zero bytes_per_element
    let all_dtypes = [DType::F32, DType::F16, DType::BF16, DType::Int8, DType::Int4];
    for dt in &all_dtypes {
        assert!(dt.bytes_per_element() > 0, "bytes_per_element must be > 0");
    }
});
