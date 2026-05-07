//! CI control plane subcommands.
//!
//! `xtask ci plan` replaces the inline Python in
//! `.github/workflows/pr-plan.yml`. It computes the changed-files
//! posture, touched areas, expected CI lanes, and an estimated LEM
//! budget band, then emits both `ci-plan.json` and a markdown
//! step-summary block.

pub mod plan;
