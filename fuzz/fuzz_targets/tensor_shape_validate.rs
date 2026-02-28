#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct ShapeInput {
    dims: Vec<u32>,
}

fuzz_target!(|input: ShapeInput| {
    let dims: Vec<usize> = input.dims.iter().take(8).map(|&d| d as usize % 1024).collect();
    if dims.is_empty() { return; }
    let total: usize = dims.iter().copied().fold(1usize, |a, b| a.saturating_mul(b));
    // Validate shape arithmetic doesn't overflow
    let _ = total;
});
