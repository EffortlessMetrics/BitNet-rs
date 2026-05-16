//! Reusable helpers for extracting layer indices from tensor names.
//!
//! This crate centralizes low-level parsing used by model loading and GPU
//! weight handling so tensor layer-name behavior follows one implementation.

/// Parse an unsigned integer from the start of `bytes` until the first non-digit.
///
/// Returns `None` if no leading digits are found or on overflow.
pub fn parse_usize_prefix(bytes: &[u8]) -> Option<usize> {
    let mut value: usize = 0;
    let mut found_any = false;

    for &b in bytes {
        if b.is_ascii_digit() {
            value = value.checked_mul(10)?.checked_add((b - b'0') as usize)?;
            found_any = true;
        } else {
            break;
        }
    }

    found_any.then_some(value)
}

/// Extract the first numeric segment from a dot-separated tensor name.
///
/// Example: `"model.layers.12.attn.q_proj.weight" -> Some(12)`.
pub fn extract_any_layer_index(name: &str) -> Option<usize> {
    for part in name.split('.') {
        if let Ok(idx) = part.parse::<usize>() {
            return Some(idx);
        }
    }
    None
}

/// Extract a layer index from known GGUF/transformer patterns.
///
/// Supports `blk.<i>.*` and `layers.<i>.*`.
pub fn extract_structured_layer_index(name: &str) -> Option<usize> {
    if let Some(idx) = parse_prefix_layer_index(name, "blk.") {
        return Some(idx);
    }
    if name.contains("blk.") {
        return None;
    }

    parse_prefix_layer_index(name, "layers.")
}

/// Extract a layer index from `blk.<i>.*` or `layers.<i>.*` where `<i>` is a
/// complete dot-separated segment.
pub fn extract_structured_layer_index_segment(name: &str) -> Option<usize> {
    if let Some(idx) = parse_segment_layer_index(name, "blk.") {
        return Some(idx);
    }
    if name.contains("blk.") {
        return None;
    }

    parse_segment_layer_index(name, "layers.")
}

/// Extract a `blk.<i>.*` block index.
///
/// Example: `"blk.3.attn_q.weight" -> Some(3)`.
pub fn parse_block_index(name: &str) -> Option<usize> {
    let mut parts = name.split('.');
    match (parts.next(), parts.next()) {
        (Some("blk"), Some(index)) => index.parse().ok(),
        _ => None,
    }
}

fn parse_prefix_layer_index(name: &str, prefix: &str) -> Option<usize> {
    name.find(prefix)
        .and_then(|pos| name.as_bytes().get(pos + prefix.len()..))
        .and_then(parse_usize_prefix)
}

fn parse_segment_layer_index(name: &str, prefix: &str) -> Option<usize> {
    let start = name.find(prefix)? + prefix.len();
    let rest = name.get(start..)?;
    let dot_pos = rest.find('.')?;
    rest.get(..dot_pos)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::{
        extract_any_layer_index, extract_structured_layer_index,
        extract_structured_layer_index_segment, parse_block_index, parse_usize_prefix,
    };

    #[test]
    fn parse_prefix_digits() {
        assert_eq!(parse_usize_prefix(b"123.attn"), Some(123));
        assert_eq!(parse_usize_prefix(b"001"), Some(1));
        assert_eq!(parse_usize_prefix(b"x1"), None);
    }

    #[test]
    fn parse_prefix_digits_checks_overflow() {
        let too_large = format!("{}0", usize::MAX);
        assert_eq!(parse_usize_prefix(too_large.as_bytes()), None);
    }

    #[test]
    fn extract_any_index_from_dotted_name() {
        assert_eq!(extract_any_layer_index("model.layers.39.mlp"), Some(39));
        assert_eq!(extract_any_layer_index("layers.0.wq"), Some(0));
        assert_eq!(extract_any_layer_index("embed_tokens.weight"), None);
    }

    #[test]
    fn extract_structured_index() {
        assert_eq!(extract_structured_layer_index("blk.123.attn_q.weight"), Some(123));
        assert_eq!(extract_structured_layer_index("model.layers.7.mlp.gate"), Some(7));
        assert_eq!(extract_structured_layer_index("token_embd.weight"), None);
    }

    #[test]
    fn extract_structured_index_preserves_blk_precedence() {
        assert_eq!(extract_structured_layer_index("x.blk.foo.layers.7.weight"), None);
    }

    #[test]
    fn extract_structured_segment_index() {
        assert_eq!(extract_structured_layer_index_segment("blk.123.attn_q.weight"), Some(123));
        assert_eq!(extract_structured_layer_index_segment("model.layers.7.mlp.gate"), Some(7));
        assert_eq!(extract_structured_layer_index_segment("layers.3x.weight"), None);
        assert_eq!(extract_structured_layer_index_segment("layers.3"), None);
        assert_eq!(extract_structured_layer_index_segment("x.blk.foo.layers.7.weight"), None);
    }

    #[test]
    fn parse_block_only() {
        assert_eq!(parse_block_index("blk.3.attn_q.weight"), Some(3));
        assert_eq!(parse_block_index("blk.x.attn_q.weight"), None);
        assert_eq!(parse_block_index("layers.3.attn_q.weight"), None);
        assert_eq!(parse_block_index("blk.3x.attn_q.weight"), None);
    }

    #[test]
    fn parse_prefix_digits_empty_returns_none() {
        assert_eq!(parse_usize_prefix(b""), None);
    }

    #[test]
    fn parse_prefix_digits_zero_and_single_digit() {
        assert_eq!(parse_usize_prefix(b"0"), Some(0));
        assert_eq!(parse_usize_prefix(b"7"), Some(7));
        assert_eq!(parse_usize_prefix(b"0.foo"), Some(0));
    }

    #[test]
    fn parse_prefix_digits_stops_at_first_non_digit() {
        assert_eq!(parse_usize_prefix(b"42x99"), Some(42));
        assert_eq!(parse_usize_prefix(b"1_0"), Some(1));
    }

    #[test]
    fn extract_any_index_empty_or_no_dots() {
        assert_eq!(extract_any_layer_index(""), None);
        assert_eq!(extract_any_layer_index("weight"), None);
    }

    #[test]
    fn extract_any_index_returns_first_numeric_segment() {
        // First numeric segment wins, not the structurally meaningful one.
        assert_eq!(extract_any_layer_index("model.7.layers.39.mlp"), Some(7));
    }

    #[test]
    fn extract_any_index_segment_must_be_pure_numeric() {
        // "39x" is not a valid usize so the iterator moves on.
        assert_eq!(extract_any_layer_index("model.layers.39x.mlp"), None);
        // Empty inputs split into a single empty segment which does not parse.
        assert_eq!(extract_any_layer_index("."), None);
    }

    #[test]
    fn extract_structured_index_blk_without_digits_returns_none() {
        // `blk.` is present but not followed by digits — prefix match yields
        // no index and the `blk` precedence rule short-circuits before
        // `layers.` is considered.
        assert_eq!(extract_structured_layer_index("blk.foo.weight"), None);
    }

    #[test]
    fn extract_structured_index_prefix_form_accepts_trailing_dot() {
        // Prefix form does not require a trailing dot delimiter after the
        // numeric prefix: `blk.5weight` is parsed as block index 5.
        assert_eq!(extract_structured_layer_index("blk.5weight"), Some(5));
    }

    #[test]
    fn extract_structured_segment_index_blk_without_following_dot() {
        // Segment form requires a `.` after the numeric segment; missing → None.
        assert_eq!(extract_structured_layer_index_segment("blk.5"), None);
        assert_eq!(extract_structured_layer_index_segment("blk.5weight"), None);
    }

    #[test]
    fn extract_structured_segment_index_layers_without_following_dot() {
        assert_eq!(extract_structured_layer_index_segment("layers.7"), None);
        assert_eq!(extract_structured_layer_index_segment("layers.7x"), None);
    }

    #[test]
    fn parse_block_index_rejects_partial_names() {
        assert_eq!(parse_block_index(""), None);
        assert_eq!(parse_block_index("blk"), None);
        assert_eq!(parse_block_index("blk."), None);
        assert_eq!(parse_block_index("notblk.3"), None);
    }
}
