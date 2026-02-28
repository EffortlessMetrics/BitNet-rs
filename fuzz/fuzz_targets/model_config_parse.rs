#![no_main]
use std::collections::HashMap;

use arbitrary::Arbitrary;
use bitnet_models::config::GgufModelConfig;
use bitnet_models::formats::gguf::GgufValue;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum FuzzGgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
}

impl FuzzGgufValue {
    fn into_gguf(self) -> GgufValue {
        match self {
            Self::U8(v) => GgufValue::U8(v),
            Self::I8(v) => GgufValue::I8(v),
            Self::U16(v) => GgufValue::U16(v),
            Self::I16(v) => GgufValue::I16(v),
            Self::U32(v) => GgufValue::U32(v),
            Self::I32(v) => GgufValue::I32(v),
            Self::F32(v) => GgufValue::F32(v),
            Self::Bool(v) => GgufValue::Bool(v),
            Self::Str(v) => GgufValue::String(v),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzEntry {
    key: String,
    value: FuzzGgufValue,
}

#[derive(Arbitrary, Debug)]
struct ConfigInput {
    entries: Vec<FuzzEntry>,
}

fuzz_target!(|input: ConfigInput| {
    let metadata: HashMap<String, GgufValue> =
        input.entries.into_iter().take(256).map(|e| (e.key, e.value.into_gguf())).collect();

    if let Ok(config) = GgufModelConfig::from_gguf_metadata(&metadata) {
        let _ = config.validate();
        let _ = config.memory_estimate();
        let _ = config.is_gqa();
        let _ = config.gqa_group_size();
    }
});
