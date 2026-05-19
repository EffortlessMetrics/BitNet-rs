pub(crate) fn requested_cpu_kernel_label() -> Option<String> {
    std::env::var("BITNET_CPU_KERNEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub(crate) fn selected_cpu_hot_path_label(
    f32_scalar: u64,
    f32_avx2: u64,
    scaled_scalar: u64,
    scaled_avx2: u64,
) -> Option<String> {
    let mut labels = Vec::new();
    if f32_scalar > 0 {
        labels.push("qk256-f32-scalar-gemv");
    }
    if f32_avx2 > 0 {
        labels.push("qk256-f32-avx2-gemv");
    }
    if scaled_scalar > 0 {
        labels.push("qk256-i2s-i8s-scaled-scalar-gemv");
    }
    if scaled_avx2 > 0 {
        labels.push("qk256-i2s-i8s-scaled-avx2-gemv");
    }
    match labels.as_slice() {
        [] => None,
        [single] => Some((*single).to_string()),
        _ => Some("mixed-qk256-cpu-hot-paths".to_string()),
    }
}

pub(crate) fn qk256_execution_path_label(
    f32_scalar: u64,
    f32_avx2: u64,
    scaled_scalar: u64,
    scaled_avx2: u64,
) -> &'static str {
    let no_scale = f32_scalar + f32_avx2;
    let scaled = scaled_scalar + scaled_avx2;
    match (no_scale > 0, scaled > 0) {
        (true, true) => "mixed_scaled_and_no_scale",
        (true, false) => "no_scale_f32",
        (false, true) => "scaled_i2s_i8s",
        (false, false) => "not_observed",
    }
}
