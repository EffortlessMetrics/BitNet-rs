pub(crate) fn normalize_label(input: &str) -> String {
    input.trim().to_ascii_lowercase().replace(['_', ' '], "-")
}
