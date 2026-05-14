//! Guards for the A770 BitNet claim-boundary spec.
//!
//! These tests keep the product boundary from drifting while the implementation
//! rails land PR by PR. They intentionally check only documentation/source
//! policy, not runtime receipts.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read_repo_file(path: &str) -> String {
    let full_path = repo_root().join(path);
    std::fs::read_to_string(&full_path).unwrap_or_default()
}

fn assert_contains(contents: &str, needle: &str, context: &str) {
    assert!(contents.contains(needle), "{} must contain `{}`", context, needle);
}

fn assert_file_exists(path: &str) {
    let full_path = repo_root().join(path);
    assert!(full_path.exists(), "missing required file {}", full_path.display());
}

#[test]
fn a770_claim_boundary_spec_files_exist() {
    assert_file_exists("docs/specs/a770-bitnet-claim-boundary.md");
    assert_file_exists("docs/specs/intel-arc-a770-gpu-roadmap.md");
    assert_file_exists("docs/hardware/intel-arc-a770-validation.md");
    assert_file_exists("plans/a770-bitnet-claim-boundary-implementation.md");
}

#[test]
fn a770_claim_boundary_docs_link_to_the_spec() {
    let roadmap = read_repo_file("docs/specs/intel-arc-a770-gpu-roadmap.md");
    let validation = read_repo_file("docs/hardware/intel-arc-a770-validation.md");
    let index = read_repo_file("docs/specs/INDEX.md");

    assert_contains(&roadmap, "docs/specs/a770-bitnet-claim-boundary.md", "A770 roadmap");
    assert_contains(&roadmap, "plans/a770-bitnet-claim-boundary-implementation.md", "A770 roadmap");
    assert_contains(
        &validation,
        "docs/specs/a770-bitnet-claim-boundary.md",
        "A770 validation profile",
    );
    assert_contains(&index, "a770-bitnet-claim-boundary.md", "spec index");
}

#[test]
fn a770_claim_boundary_preserves_first_claim_target() {
    let spec = read_repo_file("docs/specs/a770-bitnet-claim-boundary.md");
    let plan = read_repo_file("plans/a770-bitnet-claim-boundary-implementation.md");

    for contents in [&spec, &plan] {
        assert_contains(
            contents,
            "BitNet b1.58 i2_s trusted partial A770 acceleration",
            "A770 claim boundary",
        );
    }
}

#[test]
fn a770_claim_boundary_preserves_critical_not_claims() {
    let spec = read_repo_file("docs/specs/a770-bitnet-claim-boundary.md");
    let plan = read_repo_file("plans/a770-bitnet-claim-boundary-implementation.md");
    let validation = read_repo_file("docs/hardware/intel-arc-a770-validation.md");

    let required = [
        "selected_attention_residency",
        "resident_kv_decode",
        "attention_scores_residency",
        "softmax_residency",
        "attention_value_mix_residency",
        "full_support_op_residency",
        "full_device_residency",
        "completion",
    ];

    for not_claim in required {
        assert_contains(&spec, not_claim, "A770 claim-boundary spec");
        assert_contains(&plan, not_claim, "A770 implementation plan");
    }

    for not_claim in [
        "selected-attention",
        "resident KV decode",
        "attention scores",
        "softmax",
        "attention value",
        "full support-op residency",
        "full device residency",
        "completion",
    ] {
        assert_contains(&validation, not_claim, "A770 validation profile");
    }
}

#[test]
fn a770_claim_boundary_keeps_selected_attention_separate() {
    let spec = read_repo_file("docs/specs/a770-bitnet-claim-boundary.md");
    let roadmap = read_repo_file("docs/specs/intel-arc-a770-gpu-roadmap.md");

    assert_contains(
        &spec,
        "Selected attention is a separate research lane",
        "A770 claim-boundary spec",
    );
    assert_contains(&spec, "It is not promoted by trusted", "A770 claim-boundary spec");
    assert_contains(&roadmap, "Selected attention, resident KV", "A770 roadmap");
}

#[test]
fn a770_claim_boundary_keeps_benchmarks_diagnostic_until_claim_gated() {
    let roadmap = read_repo_file("docs/specs/intel-arc-a770-gpu-roadmap.md");

    assert_contains(
        &roadmap,
        "| Benchmark beats CPU baseline with artifact | Diagnostic performance evidence |",
        "A770 roadmap",
    );
    assert_contains(
        &roadmap,
        "performance claims also require the stricter claim-boundary",
        "A770 roadmap",
    );
}

#[test]
fn a770_claim_boundary_requires_clean_parent_benchmark_and_real_history() {
    let spec = read_repo_file("docs/specs/a770-bitnet-claim-boundary.md");
    let plan = read_repo_file("plans/a770-bitnet-claim-boundary-implementation.md");

    for contents in [&spec, &plan] {
        assert_contains(contents, "repo.dirty=false", "A770 claim boundary");
    }

    for needle in [
        "two distinct receipts",
        "distinct non-empty run IDs",
        "distinct receipt paths",
        "same device",
        "same backend",
        "same benchmark profile",
        "same kernel route",
    ] {
        assert_contains(&spec, needle, "A770 claim-boundary spec");
    }

    assert_contains(
        &spec,
        "Comparing a receipt to itself is not history",
        "A770 claim-boundary spec",
    );
    assert_contains(
        &plan,
        "two distinct same-device, same-route history receipts",
        "A770 implementation plan",
    );
}
