//! Shared process exit code contract for CLI and validation tooling.
//!
//! Keeping these constants in a dedicated microcrate makes the numeric
//! contract reusable and auditable across binaries.

/// Command completed successfully.
pub const EXIT_SUCCESS: i32 = 0;
/// Generic non-specific failure.
pub const EXIT_GENERIC_FAIL: i32 = 1;
/// Strict mapping gate failure.
pub const EXIT_STRICT_MAPPING: i32 = 3;
/// Strict tokenizer gate failure.
pub const EXIT_STRICT_TOKENIZER: i32 = 4;
/// NLL quality gate failure.
pub const EXIT_NLL_TOO_HIGH: i32 = 5;
/// Tau quality gate failure.
pub const EXIT_TAU_TOO_LOW: i32 = 6;
/// Argmax parity gate failure.
pub const EXIT_ARGMAX_MISMATCH: i32 = 7;
/// LayerNorm/projection strict validation failure.
pub const EXIT_LN_SUSPICIOUS: i32 = 8;
/// Performance benchmark gate failure.
pub const EXIT_PERF_FAIL: i32 = 9;
/// RSS memory ceiling gate failure.
pub const EXIT_RSS_FAIL: i32 = 10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_remain_stable() {
        assert_eq!(EXIT_SUCCESS, 0);
        assert_eq!(EXIT_GENERIC_FAIL, 1);
        assert_eq!(EXIT_STRICT_MAPPING, 3);
        assert_eq!(EXIT_STRICT_TOKENIZER, 4);
        assert_eq!(EXIT_NLL_TOO_HIGH, 5);
        assert_eq!(EXIT_TAU_TOO_LOW, 6);
        assert_eq!(EXIT_ARGMAX_MISMATCH, 7);
        assert_eq!(EXIT_LN_SUSPICIOUS, 8);
        assert_eq!(EXIT_PERF_FAIL, 9);
        assert_eq!(EXIT_RSS_FAIL, 10);
    }
}
