use bitnet_math::ceil_div;
use proptest::prelude::*;

// ── ceil_div mathematical invariants ─────────────────────────────────────────

proptest! {
    /// ceil_div(n, d) * d >= n for all n and d > 0.
    #[test]
    fn prop_ceil_div_covers_n(
        n in 0usize..=1_000_000,
        d in 1usize..=10_000,
    ) {
        let result = ceil_div(n, d);
        prop_assert!(result * d >= n,
            "ceil_div({n}, {d}) = {result} but {result} * {d} < {n}");
    }

    /// ceil_div(n, d) is the smallest integer k such that k * d >= n.
    #[test]
    fn prop_ceil_div_is_minimal(
        n in 0usize..=1_000_000,
        d in 1usize..=10_000,
    ) {
        let result = ceil_div(n, d);
        if result > 0 {
            let r = result - 1;
            prop_assert!(r * d < n,
                "ceil_div({n}, {d}) = {result} is not minimal: {r} * {d} >= {n}",
                n=n, d=d, result=result, r=r);
        }
    }

    /// When n is exactly divisible by d, ceil_div(n, d) == n / d.
    #[test]
    fn prop_ceil_div_exact_divisor(
        k in 0usize..=1_000,
        d in 1usize..=1_000,
    ) {
        let n = k * d;
        prop_assert_eq!(ceil_div(n, d), k);
    }

    /// ceil_div with d = 1 is the identity.
    #[test]
    fn prop_ceil_div_divisor_one_is_identity(n in 0usize..=1_000_000) {
        prop_assert_eq!(ceil_div(n, 1), n);
    }

    /// ceil_div(0, d) is always 0.
    #[test]
    fn prop_ceil_div_numerator_zero(d in 1usize..=4096) {
        prop_assert_eq!(ceil_div(0, d), 0);
    }
}
