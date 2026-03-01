//! Helpers for GGUF byte alignment and layout computations.

/// Align `value` up to the next multiple of `alignment`.
///
/// Returns `value` unchanged when `alignment == 0`.
#[must_use]
pub fn align_up_u64(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

/// Align `value` up to the next multiple of `alignment`.
///
/// Returns `value` unchanged when `alignment == 0`.
#[must_use]
pub fn align_up_usize(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_u64_behaves() {
        assert_eq!(align_up_u64(0, 32), 0);
        assert_eq!(align_up_u64(1, 32), 32);
        assert_eq!(align_up_u64(31, 32), 32);
        assert_eq!(align_up_u64(32, 32), 32);
        assert_eq!(align_up_u64(33, 32), 64);
    }

    #[test]
    fn align_up_usize_behaves() {
        assert_eq!(align_up_usize(0, 32), 0);
        assert_eq!(align_up_usize(1, 32), 32);
        assert_eq!(align_up_usize(31, 32), 32);
        assert_eq!(align_up_usize(32, 32), 32);
        assert_eq!(align_up_usize(33, 32), 64);
    }
}
