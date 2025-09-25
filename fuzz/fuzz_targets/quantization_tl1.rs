#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Minimal stub for compilation
    let _ = data.len();
});
