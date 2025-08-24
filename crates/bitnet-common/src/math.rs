#[inline]
#[allow(clippy::manual_div_ceil)]
pub const fn ceil_div(n: usize, d: usize) -> usize {
    (n + d - 1) / d
}
