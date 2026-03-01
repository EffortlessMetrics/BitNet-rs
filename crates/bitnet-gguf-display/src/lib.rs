//! Formatting helpers for GGUF metadata display.

use bitnet_gguf::kv::GgufValue;

/// Enhanced dtype formatting with comprehensive coverage.
pub fn format_dtype(dtype: u32) -> String {
    match dtype {
        0 => "F32".to_string(),
        1 => "F16".to_string(),
        2 => "Q4_0".to_string(),
        3 => "Q4_1".to_string(),
        4 => "Q5_0".to_string(),
        5 => "Q5_1".to_string(),
        6 => "Q8_0".to_string(),
        7 => "Q8_1".to_string(),
        8 => "Q2_K".to_string(),
        9 => "Q3_K".to_string(),
        10 => "Q4_K".to_string(),
        11 => "Q5_K".to_string(),
        12 => "Q6_K".to_string(),
        13 => "Q8_K".to_string(),
        14 => "IQ2_XXS".to_string(),
        15 => "IQ2_XS".to_string(),
        16 => "IQ3_XXS".to_string(),
        17 => "I2_S".to_string(),
        18 => "IQ2_S".to_string(),
        19 => "TL1".to_string(),
        20 => "TL2".to_string(),
        21 => "IQ1_S".to_string(),
        22 => "IQ4_NL".to_string(),
        23 => "IQ3_S".to_string(),
        24 => "IQ2_S_NEW".to_string(),
        25 => "IQ4_XS".to_string(),
        _ => format!("Unknown({})", dtype),
    }
}

/// Format GGUF values for display.
pub fn format_gguf_value(value: &GgufValue) -> String {
    match value {
        GgufValue::U8(v) => v.to_string(),
        GgufValue::I8(v) => v.to_string(),
        GgufValue::U16(v) => v.to_string(),
        GgufValue::I16(v) => v.to_string(),
        GgufValue::U32(v) => v.to_string(),
        GgufValue::I32(v) => v.to_string(),
        GgufValue::F32(v) => {
            if v.fract() == 0.0 {
                format!("{:.0}", v)
            } else {
                format!("{:.6}", v)
            }
        }
        GgufValue::Bool(v) => v.to_string(),
        GgufValue::String(v) => v.clone(),
        GgufValue::Array(arr) => {
            if arr.len() <= 3 {
                format!("[{}]", arr.iter().map(format_gguf_value).collect::<Vec<_>>().join(", "))
            } else {
                format!(
                    "[{}, ... +{} more]",
                    arr.iter().take(2).map(format_gguf_value).collect::<Vec<_>>().join(", "),
                    arr.len() - 2
                )
            }
        }
        GgufValue::U64(v) => v.to_string(),
        GgufValue::I64(v) => v.to_string(),
        GgufValue::F64(v) => {
            if v.fract() == 0.0 {
                format!("{:.0}", v)
            } else {
                format!("{:.6}", v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{format_dtype, format_gguf_value};
    use bitnet_gguf::kv::GgufValue;

    #[test]
    fn dtype_names_cover_bitnet_formats() {
        assert_eq!(format_dtype(0), "F32");
        assert_eq!(format_dtype(17), "I2_S");
        assert_eq!(format_dtype(20), "TL2");
        assert_eq!(format_dtype(999), "Unknown(999)");
    }

    #[test]
    fn gguf_value_formats_scalars_and_arrays() {
        assert_eq!(format_gguf_value(&GgufValue::U32(42)), "42");
        assert_eq!(format_gguf_value(&GgufValue::F32(std::f32::consts::PI)), "3.141593");
        assert_eq!(format_gguf_value(&GgufValue::F32(3.0)), "3");

        let arr_short = GgufValue::Array(vec![GgufValue::U32(1), GgufValue::U32(2)]);
        assert_eq!(format_gguf_value(&arr_short), "[1, 2]");

        let arr_long = GgufValue::Array(vec![
            GgufValue::U32(1),
            GgufValue::U32(2),
            GgufValue::U32(3),
            GgufValue::U32(4),
        ]);
        assert_eq!(format_gguf_value(&arr_long), "[1, 2, ... +2 more]");
    }
}
