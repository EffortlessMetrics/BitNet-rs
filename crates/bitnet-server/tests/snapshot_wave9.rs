//! Snapshot wave 9 — server model registry, canary, and config Display formatting.
//!
//! Pins the human-readable output of server infrastructure types to
//! catch accidental regressions in diagnostics and error messages.

use bitnet_server::canary::{BackendChoice, CanaryConfig, compute_divergence};
use bitnet_server::config::DeviceConfig;
use bitnet_server::model_registry::{ModelState, RegistryError};

// ── ModelState Display ─────────────────────────────────────────────

#[test]
fn model_state_display_all_variants() {
    let states =
        [ModelState::Loading, ModelState::Ready, ModelState::Serving, ModelState::Unloading];
    let output: Vec<String> = states.iter().map(|s| format!("{s}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn model_state_debug_all_variants() {
    let states =
        [ModelState::Loading, ModelState::Ready, ModelState::Serving, ModelState::Unloading];
    let output: Vec<String> = states.iter().map(|s| format!("{s:?}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

// ── RegistryError Display ──────────────────────────────────────────

#[test]
fn registry_error_already_loaded_display() {
    let err =
        RegistryError::AlreadyLoaded { model_id: "bitnet-2b".into(), device_id: "gpu:0".into() };
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn registry_error_not_found_display() {
    let err = RegistryError::NotFound { model_id: "bitnet-2b".into(), device_id: "gpu:1".into() };
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn registry_error_invalid_transition_display() {
    let err =
        RegistryError::InvalidTransition { from: ModelState::Loading, to: ModelState::Unloading };
    insta::assert_snapshot!(format!("{err}"));
}

#[test]
fn registry_error_insufficient_memory_display() {
    let err = RegistryError::InsufficientMemory {
        device_id: "gpu:0".into(),
        required: 4_000_000_000,
        available: 2_000_000_000,
    };
    insta::assert_snapshot!(format!("{err}"));
}

// ── BackendChoice Display ──────────────────────────────────────────

#[test]
fn backend_choice_display_variants() {
    let output = format!("{}\n{}", BackendChoice::Baseline, BackendChoice::Canary);
    insta::assert_snapshot!(output);
}

// ── CanaryConfig defaults ──────────────────────────────────────────

#[test]
fn canary_config_default_debug() {
    insta::assert_debug_snapshot!(CanaryConfig::default());
}

// ── compute_divergence ─────────────────────────────────────────────

#[test]
fn compute_divergence_identical_vectors() {
    let v = vec![1.0_f32, 2.0, 3.0];
    insta::assert_snapshot!(format!("{:.6}", compute_divergence(&v, &v)));
}

#[test]
fn compute_divergence_known_difference() {
    let a = vec![1.0_f32, 0.0, 0.0];
    let b = vec![0.0_f32, 0.0, 0.0];
    insta::assert_snapshot!(format!("{:.6}", compute_divergence(&a, &b)));
}

#[test]
fn compute_divergence_mismatched_lengths() {
    let a = vec![1.0_f32, 2.0];
    let b = vec![1.0_f32];
    insta::assert_snapshot!(format!("{}", compute_divergence(&a, &b)));
}

// ── DeviceConfig parsing ───────────────────────────────────────────

#[test]
fn device_config_parse_auto() {
    let cfg: DeviceConfig = "auto".parse().unwrap();
    insta::assert_snapshot!(format!("{cfg:?}"));
}

#[test]
fn device_config_parse_cpu() {
    let cfg: DeviceConfig = "cpu".parse().unwrap();
    insta::assert_snapshot!(format!("{cfg:?}"));
}

#[test]
fn device_config_parse_gpu_with_id() {
    let cfg: DeviceConfig = "gpu:2".parse().unwrap();
    insta::assert_snapshot!(format!("{cfg:?}"));
}
