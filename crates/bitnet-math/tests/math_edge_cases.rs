//! Edge-case tests for bitnet-math utilities.
//!
//! Tests cover:
//! - ceil_div: normal cases, exact division, single element, large values
//! - ceil_div: boundary cases (d=1, n=0)

use bitnet_math::ceil_div;

// ---------------------------------------------------------------------------
// ceil_div — normal cases
// ---------------------------------------------------------------------------

#[test]
fn ceil_div_exact() {
    assert_eq!(ceil_div(10, 5), 2);
    assert_eq!(ceil_div(100, 10), 10);
    assert_eq!(ceil_div(256, 32), 8);
}

#[test]
fn ceil_div_rounds_up() {
    assert_eq!(ceil_div(11, 5), 3);
    assert_eq!(ceil_div(101, 10), 11);
    assert_eq!(ceil_div(257, 32), 9);
}

#[test]
fn ceil_div_one_remainder() {
    assert_eq!(ceil_div(6, 5), 2);
    assert_eq!(ceil_div(33, 32), 2);
}

// ---------------------------------------------------------------------------
// ceil_div — boundary cases
// ---------------------------------------------------------------------------

#[test]
fn ceil_div_n_zero() {
    assert_eq!(ceil_div(0, 1), 0);
    assert_eq!(ceil_div(0, 100), 0);
}

#[test]
fn ceil_div_d_one() {
    assert_eq!(ceil_div(42, 1), 42);
    assert_eq!(ceil_div(0, 1), 0);
    assert_eq!(ceil_div(1, 1), 1);
}

#[test]
fn ceil_div_n_equals_d() {
    assert_eq!(ceil_div(5, 5), 1);
    assert_eq!(ceil_div(1024, 1024), 1);
}

#[test]
fn ceil_div_n_less_than_d() {
    assert_eq!(ceil_div(1, 5), 1);
    assert_eq!(ceil_div(3, 100), 1);
}

#[test]
fn ceil_div_large_values() {
    // 1 billion / 256 = 3_906_250 exact
    assert_eq!(ceil_div(1_000_000_000, 256), 3_906_250);
    // 1 billion + 1 / 256 = 3_906_251 (rounds up)
    assert_eq!(ceil_div(1_000_000_001, 256), 3_906_251);
}

#[test]
fn ceil_div_is_const() {
    // Verify it can be used in const context
    const RESULT: usize = ceil_div(10, 3);
    assert_eq!(RESULT, 4);
}

#[test]
fn ceil_div_powers_of_two() {
    assert_eq!(ceil_div(4096, 32), 128);
    assert_eq!(ceil_div(16384, 64), 256);
    assert_eq!(ceil_div(65536, 256), 256);
}
