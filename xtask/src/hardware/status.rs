pub(super) fn is_known_status(status: &str) -> bool {
    matches!(
        status,
        "unsupported"
            | "missing"
            | "diagnostic"
            | "load_proven"
            | "parity_proven"
            | "quality_proven"
            | "performance_proven"
            | "resident_proven"
            | "complete"
    )
}

pub(super) fn is_claimable_status(status: &str) -> bool {
    matches!(status, "quality_proven" | "performance_proven" | "resident_proven" | "complete")
}

pub(super) fn status_requires_receipts(status: &str) -> bool {
    !matches!(status, "unsupported" | "missing" | "diagnostic")
}
