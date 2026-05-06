#[inline]
pub const fn ceil_div(n: usize, d: usize) -> usize {
    assert!(d > 0, "ceil_div divisor must be non-zero");

    // Avoid `n + d - 1` overflow by using quotient + remainder.
    let q = n / d;
    let r = n % d;
    if r != 0 { q + 1 } else { q }
}
