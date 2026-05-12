//! Fallible test assertions.
//!
//! These helpers make it practical to write tests that propagate
//! failure with `?` instead of panicking through `unwrap()`,
//! `expect()`, `assert!`, or `assert_eq!`.
//!
//! See `docs/NO_PANIC_POLICY.md` for the broader rationale: tests
//! are part of the contract, and the long-term goal is for fixture
//! loading, parsing, indexing, and helper plumbing to all be
//! fallible.
//!
//! # Quick reference
//!
//! ```rust,ignore
//! use bitnet_test_support::assertions::{ensure, ensure_eq, require_some, require_ok};
//!
//! fn parse_a_thing() -> anyhow::Result<u32> { Ok(42) }
//!
//! #[test]
//! fn check_thing() -> anyhow::Result<()> {
//!     let v = require_ok(parse_a_thing(), "thing parse")?;
//!     ensure_eq(v, 42, "thing value")?;
//!     ensure(v.is_power_of_two(), "expected v to be a power of two")?;
//!     Ok(())
//! }
//! ```

use std::fmt;

/// Return `Ok(())` if `condition` is `true`, else fail with `message`.
///
/// Replaces `assert!(cond, "...")` in fallible tests.
///
/// # Errors
/// Returns an error built from `message` when `condition` is `false`.
#[inline]
pub fn ensure(condition: bool, message: impl Into<String>) -> anyhow::Result<()> {
    if condition { Ok(()) } else { Err(anyhow::anyhow!(message.into())) }
}

/// Return `Ok(())` if `actual == expected`, else fail with a message
/// that includes both values.
///
/// Replaces `assert_eq!` in fallible tests.
///
/// # Errors
/// Returns an error including the rendered `actual` and `expected`
/// values when they are not equal.
#[inline]
pub fn ensure_eq<T>(actual: T, expected: T, label: impl fmt::Display) -> anyhow::Result<()>
where
    T: fmt::Debug + PartialEq,
{
    if actual == expected {
        Ok(())
    } else {
        Err(anyhow::anyhow!("{label}: actual = {actual:?}, expected = {expected:?}"))
    }
}

/// Return `Ok(())` if `actual != unexpected`, else fail with a message.
///
/// # Errors
/// Returns an error labelled `label` when the values are equal.
#[inline]
pub fn ensure_ne<T>(actual: T, unexpected: T, label: impl fmt::Display) -> anyhow::Result<()>
where
    T: fmt::Debug + PartialEq,
{
    if actual != unexpected {
        Ok(())
    } else {
        Err(anyhow::anyhow!("{label}: did not expect {actual:?}"))
    }
}

/// Unwrap an `Option<T>` into `Result<T, anyhow::Error>` with `label` on `None`.
///
/// Replaces `option.unwrap()` and `option.expect("...")` in fallible tests.
///
/// # Errors
/// Returns an error labelled `label` when `value` is `None`.
#[inline]
pub fn require_some<T>(value: Option<T>, label: impl Into<String>) -> anyhow::Result<T> {
    value.ok_or_else(|| anyhow::anyhow!(label.into()))
}

/// Unwrap a `Result<T, E>` into `Result<T, anyhow::Error>` with the
/// original error rendered.
///
/// Replaces `result.unwrap()` and `result.expect("...")` in fallible tests.
///
/// # Errors
/// Returns an error labelled `label` (with the original error appended)
/// when `value` is `Err`.
#[inline]
pub fn require_ok<T, E>(value: Result<T, E>, label: impl Into<String>) -> anyhow::Result<T>
where
    E: fmt::Debug,
{
    value.map_err(|err| anyhow::anyhow!("{}: {err:?}", label.into()))
}

/// Unwrap a `Result<T, E>` whose error is meant to surface as a
/// failure with its `Display` representation rather than `Debug`.
///
/// # Errors
/// Returns an error labelled `label` (with the original error's `Display`
/// representation appended) when `value` is `Err`.
#[inline]
pub fn require_ok_display<T, E>(value: Result<T, E>, label: impl Into<String>) -> anyhow::Result<T>
where
    E: fmt::Display,
{
    value.map_err(|err| anyhow::anyhow!("{}: {err}", label.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_passes_when_true() -> anyhow::Result<()> {
        ensure(true, "always")
    }

    #[test]
    fn ensure_fails_with_message() -> anyhow::Result<()> {
        let Err(e) = ensure(false, "bad thing") else {
            anyhow::bail!("ensure unexpectedly passed");
        };
        ensure(e.to_string().contains("bad thing"), "error includes label")
    }

    #[test]
    fn ensure_eq_renders_values() -> anyhow::Result<()> {
        let Err(e) = ensure_eq(1u32, 2u32, "magic value") else {
            anyhow::bail!("ensure_eq unexpectedly passed");
        };
        let s = e.to_string();
        ensure(s.contains("magic value"), "error includes label")?;
        ensure(s.contains("actual = 1"), "error includes actual value")?;
        ensure(s.contains("expected = 2"), "error includes expected value")
    }

    #[test]
    fn ensure_ne_passes() -> anyhow::Result<()> {
        ensure_ne(1u32, 2u32, "different")?;
        let Err(e) = ensure_ne(1u32, 1u32, "same") else {
            anyhow::bail!("ensure_ne unexpectedly passed");
        };
        ensure(e.to_string().contains("same"), "error includes label")
    }

    #[test]
    fn require_some_unwraps_or_labels() -> anyhow::Result<()> {
        ensure_eq(require_some(Some(7u32), "no seven")?, 7, "some value")?;
        let Err(e) = require_some(Option::<u32>::None, "no value") else {
            anyhow::bail!("require_some unexpectedly passed");
        };
        ensure(e.to_string().contains("no value"), "error includes label")
    }

    #[test]
    fn require_ok_renders_inner_error_with_debug() -> anyhow::Result<()> {
        struct E(&'static str);
        impl std::fmt::Debug for E {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "debug-boom: {}", self.0)
            }
        }
        let Err(e) = require_ok::<(), E>(Err(E("boom")), "thing") else {
            anyhow::bail!("require_ok unexpectedly passed");
        };
        let s = e.to_string();
        ensure(s.contains("thing"), "error includes label")?;
        ensure(s.contains("boom"), "error includes debug detail")
    }

    #[test]
    fn require_ok_display_uses_display_format() -> anyhow::Result<()> {
        struct E(&'static str);
        impl std::fmt::Display for E {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "boom-display: {}", self.0)
            }
        }
        let Err(e) = require_ok_display::<(), E>(Err(E("X")), "thing") else {
            anyhow::bail!("require_ok_display unexpectedly passed");
        };
        ensure(e.to_string().contains("boom-display: X"), "error uses display detail")
    }
}
