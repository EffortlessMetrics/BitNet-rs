//! Attention mask construction for transformer inference.
//!
//! Builds causal, padded, and sliding window attention masks.

/// Attention mask type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskType {
    /// Standard causal (lower-triangular) mask.
    Causal,
    /// No mask (full attention).
    Full,
    /// Sliding window attention.
    SlidingWindow(usize),
}

/// Build a causal attention mask (lower triangular).
/// Returns a flattened seq_len × seq_len mask where true = attend.
pub fn causal_mask(seq_len: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = true;
        }
    }
    mask
}

/// Build a sliding window causal mask.
/// Each position attends to at most `window_size` previous positions.
pub fn sliding_window_mask(seq_len: usize, window_size: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * seq_len];
    for i in 0..seq_len {
        let start = if i >= window_size { i - window_size + 1 } else { 0 };
        for j in start..=i {
            mask[i * seq_len + j] = true;
        }
    }
    mask
}

/// Build a full attention mask (no masking).
pub fn full_mask(seq_len: usize) -> Vec<bool> {
    vec![true; seq_len * seq_len]
}

/// Build a mask from a mask type.
pub fn build_mask(mask_type: MaskType, seq_len: usize) -> Vec<bool> {
    match mask_type {
        MaskType::Causal => causal_mask(seq_len),
        MaskType::Full => full_mask(seq_len),
        MaskType::SlidingWindow(w) => sliding_window_mask(seq_len, w),
    }
}

/// Apply padding to an attention mask.
/// Positions beyond `actual_len` are masked out.
pub fn apply_padding(mask: &mut [bool], seq_len: usize, actual_len: usize) {
    for i in 0..seq_len {
        for j in 0..seq_len {
            if i >= actual_len || j >= actual_len {
                mask[i * seq_len + j] = false;
            }
        }
    }
}

/// Convert bool mask to f32 (0.0 = attend, -inf = ignore).
pub fn mask_to_f32(mask: &[bool]) -> Vec<f32> {
    mask.iter().map(|&b| if b { 0.0 } else { f32::NEG_INFINITY }).collect()
}

/// Count attended positions for a query at position `pos`.
pub fn attended_count(mask: &[bool], seq_len: usize, pos: usize) -> usize {
    if pos >= seq_len {
        return 0;
    }
    (0..seq_len).filter(|&j| mask[pos * seq_len + j]).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_size() {
        let m = causal_mask(4);
        assert_eq!(m.len(), 16);
    }

    #[test]
    fn test_causal_mask_diagonal() {
        let m = causal_mask(3);
        // Row 0: [T, F, F]
        assert!(m[0]);
        assert!(!m[1]);
        assert!(!m[2]);
        // Row 1: [T, T, F]
        assert!(m[3]);
        assert!(m[4]);
        assert!(!m[5]);
        // Row 2: [T, T, T]
        assert!(m[6]);
        assert!(m[7]);
        assert!(m[8]);
    }

    #[test]
    fn test_full_mask() {
        let m = full_mask(3);
        assert!(m.iter().all(|&b| b));
    }

    #[test]
    fn test_sliding_window() {
        let m = sliding_window_mask(5, 2);
        // Position 0: attends to [0] (window doesn't go negative)
        assert_eq!(attended_count(&m, 5, 0), 1);
        // Position 1: attends to [0, 1]
        assert_eq!(attended_count(&m, 5, 1), 2);
        // Position 4: attends to [3, 4]
        assert_eq!(attended_count(&m, 5, 4), 2);
    }

    #[test]
    fn test_sliding_window_large() {
        let m = sliding_window_mask(4, 100);
        // Large window = basically causal
        let c = causal_mask(4);
        assert_eq!(m, c);
    }

    #[test]
    fn test_build_mask_causal() {
        let m = build_mask(MaskType::Causal, 3);
        assert_eq!(m, causal_mask(3));
    }

    #[test]
    fn test_build_mask_full() {
        let m = build_mask(MaskType::Full, 3);
        assert_eq!(m, full_mask(3));
    }

    #[test]
    fn test_build_mask_sliding() {
        let m = build_mask(MaskType::SlidingWindow(2), 4);
        assert_eq!(m, sliding_window_mask(4, 2));
    }

    #[test]
    fn test_apply_padding() {
        let mut m = causal_mask(4);
        apply_padding(&mut m, 4, 2);
        // Only positions 0,1 should have any true values
        assert_eq!(attended_count(&m, 4, 0), 1);
        assert_eq!(attended_count(&m, 4, 1), 2);
        assert_eq!(attended_count(&m, 4, 2), 0);
        assert_eq!(attended_count(&m, 4, 3), 0);
    }

    #[test]
    fn test_mask_to_f32() {
        let m = vec![true, false, true];
        let f = mask_to_f32(&m);
        assert_eq!(f[0], 0.0);
        assert!(f[1].is_infinite());
        assert_eq!(f[2], 0.0);
    }

    #[test]
    fn test_attended_count() {
        let m = causal_mask(4);
        assert_eq!(attended_count(&m, 4, 0), 1);
        assert_eq!(attended_count(&m, 4, 3), 4);
    }

    #[test]
    fn test_attended_count_oob() {
        let m = causal_mask(3);
        assert_eq!(attended_count(&m, 3, 5), 0);
    }

    #[test]
    fn test_empty_mask() {
        let m = causal_mask(0);
        assert!(m.is_empty());
    }

    #[test]
    fn test_single_position() {
        let m = causal_mask(1);
        assert_eq!(m, vec![true]);
    }

    #[test]
    fn test_mask_type_eq() {
        assert_eq!(MaskType::Causal, MaskType::Causal);
        assert_ne!(MaskType::Causal, MaskType::Full);
        assert_eq!(MaskType::SlidingWindow(128), MaskType::SlidingWindow(128));
    }
}
