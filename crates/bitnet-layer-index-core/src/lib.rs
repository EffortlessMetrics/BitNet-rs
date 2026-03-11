//! Layer-index extraction helpers for tensor names.
//!
//! This crate centralizes the small but repeated parsing rule used across
//! multiple crates: return the first dot-delimited segment that parses as
//! `usize`.

/// Extracts the first dot-delimited numeric segment from a tensor/weight name.
///
/// # Examples
///
/// ```
/// use bitnet_layer_index_core::extract_layer_index;
///
/// assert_eq!(extract_layer_index("layers.5.attention.wq"), Some(5));
/// assert_eq!(extract_layer_index("model.layers.12.mlp.gate"), Some(12));
/// assert_eq!(extract_layer_index("embed_tokens.weight"), None);
/// ```
#[must_use]
pub fn extract_layer_index(name: &str) -> Option<usize> {
    for part in name.split('.') {
        if let Ok(idx) = part.parse::<usize>() {
            return Some(idx);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::extract_layer_index;

    #[test]
    fn extracts_first_numeric_segment() {
        assert_eq!(extract_layer_index("layers.5.attention.wq"), Some(5));
        assert_eq!(extract_layer_index("model.layers.12.mlp.gate"), Some(12));
    }

    #[test]
    fn returns_none_when_no_numeric_segment_exists() {
        assert_eq!(extract_layer_index("embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("layers.attention.wq"), None);
    }

    #[test]
    fn handles_zero_and_leading_tokens() {
        assert_eq!(extract_layer_index("layers.0.wq"), Some(0));
        assert_eq!(extract_layer_index("0.layers.wq"), Some(0));
    }
}
