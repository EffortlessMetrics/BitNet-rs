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
    fn parse_usize_prefix_rejects_empty_input() {
        assert_eq!(parse_usize_prefix(b""), None);
    }

    #[test]
    fn parse_usize_prefix_stops_at_first_non_digit() {
        assert_eq!(parse_usize_prefix(b"4096abc"), Some(4096));
        assert_eq!(parse_usize_prefix(b"7."), Some(7));
    }

    #[test]
    fn parse_block_index_requires_pure_numeric_segment() {
        // "blk.3" with no trailing dot still parses the block index — the suffix
        // string is consumed by the int parser.
        assert_eq!(parse_block_index("blk.3"), Some(3));
        // But ".3.x" with a non-numeric tail in the same segment must fail.
        assert_eq!(parse_block_index("blk.3a"), None);
    }

    #[test]
    fn extract_any_layer_index_picks_first_numeric_segment() {
        // When several segments parse as numbers, only the first wins.
        assert_eq!(extract_any_layer_index("model.layers.3.attn.4.weight"), Some(3));
    }

    #[test]
    fn extract_any_layer_index_handles_empty_string() {
        assert_eq!(extract_any_layer_index(""), None);
    }

    #[test]
    fn extract_structured_layer_index_handles_layers_with_numeric_suffix() {
        // The "prefix" variant accepts a digit prefix even when the segment is
        // not purely numeric.
        assert_eq!(extract_structured_layer_index("layers.7x.weight"), Some(7));
    }

    #[test]
    fn extract_structured_layer_index_segment_rejects_numeric_with_suffix() {
        // The "segment" variant rejects the same input because "7x" is not a
        // valid usize on its own.
        assert_eq!(extract_structured_layer_index_segment("layers.7x.weight"), None);
    }

    #[test]
    fn extract_structured_returns_none_for_unrelated_names() {
        for name in ["output.weight", "tok_embd.bias", "norm.weight"] {
            assert_eq!(extract_structured_layer_index(name), None);
            assert_eq!(extract_structured_layer_index_segment(name), None);
        }
    }
}
