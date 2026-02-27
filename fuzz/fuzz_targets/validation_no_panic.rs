#![no_main]

use arbitrary::Arbitrary;
use bitnet_validation::{
    is_ln_gamma,
    rules::{detect_rules, rules_bitnet_b158_f16, rules_bitnet_b158_i2s, rules_generic},
};
use libfuzzer_sys::fuzz_target;

/// Structured input exercising the validation layer with arbitrary
/// tensor names and RMS statistics.
#[derive(Arbitrary, Debug)]
struct ValidationInput {
    /// Simulates the GGUF `general.architecture` metadata value.
    arch: String,
    /// Simulates the GGUF `general.file_type` metadata value.
    file_type: u32,
    /// Arbitrary tensor name (e.g. "blk.0.attn_norm.weight").
    tensor_name: String,
    /// Arbitrary RMS statistic (may be NaN, ±inf, negative).
    rms: f32,
    /// A second RMS for projection weight checks.
    proj_rms: f32,
}

fuzz_target!(|input: ValidationInput| {
    // Cap string lengths to keep individual runs fast.
    if input.arch.len() > 64 || input.tensor_name.len() > 256 {
        return;
    }

    // detect_rules must not panic on arbitrary arch + file_type.
    let ruleset = detect_rules(&input.arch, input.file_type);

    // check_ln must not panic; returning false for out-of-envelope is fine.
    let _ = ruleset.check_ln(&input.tensor_name, input.rms);

    // check_proj_rms must not panic on arbitrary (possibly non-finite) RMS.
    let _ = ruleset.check_proj_rms(input.proj_rms);

    // is_ln_gamma must not panic on arbitrary tensor name.
    let _ = is_ln_gamma(&input.tensor_name);

    // Built-in rulesets must also not panic.
    let f16 = rules_bitnet_b158_f16();
    let _ = f16.check_ln(&input.tensor_name, input.rms);
    let _ = f16.check_proj_rms(input.proj_rms);

    let i2s = rules_bitnet_b158_i2s();
    let _ = i2s.check_ln(&input.tensor_name, input.rms);
    let _ = i2s.check_proj_rms(input.proj_rms);

    let generic = rules_generic();
    let _ = generic.check_ln(&input.tensor_name, input.rms);
    let _ = generic.check_proj_rms(input.proj_rms);
});
