// Exit codes for precise CI triage
#[allow(dead_code)]
pub const EXIT_SUCCESS: i32 = 0;
#[allow(dead_code)]
pub const EXIT_GENERIC_FAIL: i32 = 1;
#[allow(dead_code)]
pub const EXIT_STRICT_MAPPING: i32 = 3;
pub const EXIT_STRICT_TOKENIZER: i32 = 4;
// Validation gate exit codes
pub const EXIT_NLL_TOO_HIGH: i32 = 5;
pub const EXIT_TAU_TOO_LOW: i32 = 6;
pub const EXIT_ARGMAX_MISMATCH: i32 = 7;
#[allow(dead_code)]
pub const EXIT_PERF_FAIL: i32 = 9;
#[allow(dead_code)]
pub const EXIT_RSS_FAIL: i32 = 10;