//! SRP helpers for extracting transformer layer indices from tensor names.

/// Extract the first numeric segment in a dot-separated tensor name.
///
/// Examples:
/// - `"model.layers.12.attn.q_proj"` -> `Some(12)`
/// - `"layers.0.wq"` -> `Some(0)`
/// - `"embed_tokens.weight"` -> `None`
#[must_use]
pub fn extract_first_numeric_segment(name: &str) -> Option<usize> {
    name.split('.').find_map(|part| part.parse::<usize>().ok())
}

/// Extract a layer index immediately following one of the provided prefixes.
///
/// This is useful for formats that encode layer IDs directly after tokens such
/// as `"blk."` or `"layers."`.
#[must_use]
pub fn extract_prefixed_layer_index(name: &str, prefixes: &[&str]) -> Option<usize> {
    prefixes.iter().find_map(|prefix| {
        name.find(prefix).and_then(|pos| parse_usize_prefix(name[pos + prefix.len()..].as_bytes()))
    })
}

/// Parse an unsigned integer from the beginning of `bytes` until the first non-digit.
#[must_use]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_numeric_segment_works() {
        assert_eq!(extract_first_numeric_segment("model.layers.12.mlp.gate"), Some(12));
        assert_eq!(extract_first_numeric_segment("layers.0.wq"), Some(0));
        assert_eq!(extract_first_numeric_segment("embed_tokens.weight"), None);
    }

    #[test]
    fn prefixed_layer_index_works() {
        let prefixes = ["blk.", "layers."];
        assert_eq!(extract_prefixed_layer_index("blk.39.attn_q.weight", &prefixes), Some(39));
        assert_eq!(extract_prefixed_layer_index("model.layers.5.attention.wq", &prefixes), Some(5));
        assert_eq!(extract_prefixed_layer_index("model.lx.5.attention.wq", &prefixes), None);
    }

    #[test]
    fn parse_usize_prefix_rejects_overflow() {
        let huge = format!("{}tail", usize::MAX);
        assert!(parse_usize_prefix(huge.as_bytes()).is_some());
        let overflowing = format!("{}0tail", usize::MAX);
        assert_eq!(parse_usize_prefix(overflowing.as_bytes()), None);
    }
}
