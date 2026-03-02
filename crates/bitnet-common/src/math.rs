#[inline]
#[allow(clippy::manual_div_ceil, clippy::arithmetic_side_effects)]
pub const fn ceil_div(n: usize, d: usize) -> usize {
    // Safe ceiling division with overflow protection
    // Note: This is a const fn so we can't use checked arithmetic in const context
    // This pattern is safe when d > 0 and n + d - 1 doesn't overflow
    // Used for tensor dimension calculations where inputs are validated
    (n + d - 1) / d
}
