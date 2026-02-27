#![no_main]

use arbitrary::Arbitrary;
use bitnet_honest_compute::{
    classify_compute_path, is_mock_kernel_id, validate_compute_path, validate_kernel_ids,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct HonestComputeInput {
    /// Arbitrary bytes used as the compute path string.
    compute_path: Vec<u8>,
    /// Kernel ID bytes; each 32-byte chunk becomes one kernel ID (lossy UTF-8).
    kernel_bytes: Vec<u8>,
    /// Single kernel ID for `is_mock_kernel_id`.
    single_kernel: Vec<u8>,
}

fuzz_target!(|input: HonestComputeInput| {
    let compute_path = String::from_utf8_lossy(&input.compute_path).into_owned();

    // validate_compute_path must not panic for any string.
    let _ = validate_compute_path(&compute_path);

    // Build a list of kernel IDs from the raw bytes (≤ 64 IDs).
    let kernel_ids: Vec<String> = input
        .kernel_bytes
        .chunks(32)
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .take(64)
        .collect();

    // validate_kernel_ids and classify_compute_path must not panic for any input.
    let _ = validate_kernel_ids(kernel_ids.iter().map(String::as_str));
    let _ = classify_compute_path(kernel_ids.iter().map(String::as_str));

    // is_mock_kernel_id must not panic for any string.
    let single = String::from_utf8_lossy(&input.single_kernel).into_owned();
    let _ = is_mock_kernel_id(&single);
});
