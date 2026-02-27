#![no_main]

use bitnet_receipts::InferenceReceipt;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Deserialise arbitrary bytes as InferenceReceipt JSON – must never panic.
    let _ = serde_json::from_slice::<InferenceReceipt>(data);
});
