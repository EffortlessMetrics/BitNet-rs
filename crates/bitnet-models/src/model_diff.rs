//! Human-readable model difference reports.
//!
//! Compare two model configurations and produce a structured diff
//! showing what changed, added, or removed.

use std::collections::BTreeMap;

/// Type of difference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffKind {
    Added,
    Removed,
    Changed,
    Unchanged,
}

/// A single difference entry.
#[derive(Debug, Clone)]
pub struct DiffEntry {
    pub field: String,
    pub kind: DiffKind,
    pub left: Option<String>,
    pub right: Option<String>,
}

impl DiffEntry {
    pub fn added(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self { field: field.into(), kind: DiffKind::Added, left: None, right: Some(value.into()) }
    }

    pub fn removed(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self { field: field.into(), kind: DiffKind::Removed, left: Some(value.into()), right: None }
    }

    pub fn changed(
        field: impl Into<String>,
        from: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        Self {
            field: field.into(),
            kind: DiffKind::Changed,
            left: Some(from.into()),
            right: Some(to.into()),
        }
    }

    pub fn unchanged(field: impl Into<String>, value: impl Into<String>) -> Self {
        let v = value.into();
        Self {
            field: field.into(),
            kind: DiffKind::Unchanged,
            left: Some(v.clone()),
            right: Some(v),
        }
    }
}

/// A complete diff report.
#[derive(Debug, Clone)]
pub struct DiffReport {
    pub left_name: String,
    pub right_name: String,
    pub entries: Vec<DiffEntry>,
}

impl DiffReport {
    pub fn new(left: impl Into<String>, right: impl Into<String>) -> Self {
        Self { left_name: left.into(), right_name: right.into(), entries: Vec::new() }
    }

    pub fn add(&mut self, entry: DiffEntry) {
        self.entries.push(entry);
    }

    pub fn has_changes(&self) -> bool {
        self.entries.iter().any(|e| e.kind != DiffKind::Unchanged)
    }

    pub fn change_count(&self) -> usize {
        self.entries.iter().filter(|e| e.kind != DiffKind::Unchanged).count()
    }

    pub fn added_count(&self) -> usize {
        self.entries.iter().filter(|e| e.kind == DiffKind::Added).count()
    }

    pub fn removed_count(&self) -> usize {
        self.entries.iter().filter(|e| e.kind == DiffKind::Removed).count()
    }

    pub fn changed_entries(&self) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| e.kind == DiffKind::Changed).collect()
    }

    /// Render as plain text.
    pub fn render_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("--- {}", self.left_name));
        lines.push(format!("+++ {}", self.right_name));
        lines.push(String::new());

        for entry in &self.entries {
            match &entry.kind {
                DiffKind::Added => {
                    lines.push(format!(
                        "+ {}: {}",
                        entry.field,
                        entry.right.as_deref().unwrap_or("")
                    ));
                }
                DiffKind::Removed => {
                    lines.push(format!(
                        "- {}: {}",
                        entry.field,
                        entry.left.as_deref().unwrap_or("")
                    ));
                }
                DiffKind::Changed => {
                    lines.push(format!(
                        "~ {}: {} -> {}",
                        entry.field,
                        entry.left.as_deref().unwrap_or(""),
                        entry.right.as_deref().unwrap_or(""),
                    ));
                }
                DiffKind::Unchanged => {
                    lines.push(format!(
                        "  {}: {}",
                        entry.field,
                        entry.left.as_deref().unwrap_or("")
                    ));
                }
            }
        }
        lines.join("\n")
    }
}

/// Compare two key-value maps.
pub fn diff_maps(
    left_name: &str,
    right_name: &str,
    left: &BTreeMap<String, String>,
    right: &BTreeMap<String, String>,
) -> DiffReport {
    let mut report = DiffReport::new(left_name, right_name);

    for (key, lval) in left {
        if let Some(rval) = right.get(key) {
            if lval == rval {
                report.add(DiffEntry::unchanged(key, lval));
            } else {
                report.add(DiffEntry::changed(key, lval, rval));
            }
        } else {
            report.add(DiffEntry::removed(key, lval));
        }
    }

    for (key, rval) in right {
        if !left.contains_key(key) {
            report.add(DiffEntry::added(key, rval));
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn test_identical_maps() {
        let m = make_map(&[("a", "1"), ("b", "2")]);
        let report = diff_maps("left", "right", &m, &m);
        assert!(!report.has_changes());
        assert_eq!(report.change_count(), 0);
    }

    #[test]
    fn test_added_field() {
        let left = make_map(&[("a", "1")]);
        let right = make_map(&[("a", "1"), ("b", "2")]);
        let report = diff_maps("L", "R", &left, &right);
        assert_eq!(report.added_count(), 1);
    }

    #[test]
    fn test_removed_field() {
        let left = make_map(&[("a", "1"), ("b", "2")]);
        let right = make_map(&[("a", "1")]);
        let report = diff_maps("L", "R", &left, &right);
        assert_eq!(report.removed_count(), 1);
    }

    #[test]
    fn test_changed_field() {
        let left = make_map(&[("a", "1")]);
        let right = make_map(&[("a", "2")]);
        let report = diff_maps("L", "R", &left, &right);
        assert_eq!(report.changed_entries().len(), 1);
    }

    #[test]
    fn test_render_text() {
        let left = make_map(&[("layers", "32")]);
        let right = make_map(&[("layers", "40")]);
        let report = diff_maps("model_a", "model_b", &left, &right);
        let text = report.render_text();
        assert!(text.contains("--- model_a"));
        assert!(text.contains("+++ model_b"));
        assert!(text.contains("~ layers: 32 -> 40"));
    }

    #[test]
    fn test_diff_entry_constructors() {
        let a = DiffEntry::added("x", "1");
        assert_eq!(a.kind, DiffKind::Added);

        let r = DiffEntry::removed("y", "2");
        assert_eq!(r.kind, DiffKind::Removed);

        let c = DiffEntry::changed("z", "3", "4");
        assert_eq!(c.kind, DiffKind::Changed);

        let u = DiffEntry::unchanged("w", "5");
        assert_eq!(u.kind, DiffKind::Unchanged);
    }

    #[test]
    fn test_report_counts() {
        let mut report = DiffReport::new("a", "b");
        report.add(DiffEntry::added("x", "1"));
        report.add(DiffEntry::removed("y", "2"));
        report.add(DiffEntry::changed("z", "3", "4"));
        report.add(DiffEntry::unchanged("w", "5"));
        assert_eq!(report.change_count(), 3);
        assert_eq!(report.added_count(), 1);
        assert_eq!(report.removed_count(), 1);
    }

    #[test]
    fn test_empty_maps() {
        let left = BTreeMap::new();
        let right = BTreeMap::new();
        let report = diff_maps("L", "R", &left, &right);
        assert!(!report.has_changes());
        assert_eq!(report.entries.len(), 0);
    }

    #[test]
    fn test_render_added() {
        let left = BTreeMap::new();
        let right = make_map(&[("new_field", "value")]);
        let report = diff_maps("L", "R", &left, &right);
        let text = report.render_text();
        assert!(text.contains("+ new_field: value"));
    }

    #[test]
    fn test_render_removed() {
        let left = make_map(&[("old_field", "value")]);
        let right = BTreeMap::new();
        let report = diff_maps("L", "R", &left, &right);
        let text = report.render_text();
        assert!(text.contains("- old_field: value"));
    }

    #[test]
    fn test_has_changes_false() {
        let m = make_map(&[("a", "1")]);
        let report = diff_maps("L", "R", &m, &m);
        assert!(!report.has_changes());
    }

    #[test]
    fn test_complex_diff() {
        let left = make_map(&[("arch", "bitnet"), ("layers", "30"), ("old", "x")]);
        let right = make_map(&[("arch", "phi"), ("layers", "40"), ("new", "y")]);
        let report = diff_maps("bitnet", "phi", &left, &right);
        assert!(report.has_changes());
        assert_eq!(report.change_count(), 4); // 2 changed + 1 removed + 1 added
    }
}
