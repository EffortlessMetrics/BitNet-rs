#![cfg(feature = "metal")]

use std::error::Error;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use bitnet_device_probe::{AppleBackendReceipt, AppleResolvedDevice};
use bitnet_kernels::metal::dense_prefill_attention_scores::{
    dense_prefill_attention_scores_runtime_api_available,
    run_dense_prefill_attention_scores_blocking,
};
use bitnet_kernels::metal::smoke::{
    DENSE_KERNEL_FAMILY, DENSE_METAL_PREFILL_ATTENTION_SCORES_KERNEL_ID, DENSE_MODEL_FAMILY,
    DENSE_PREFILL_ATTENTION_SCORES_EXECUTION_PHASE,
    DENSE_PREFILL_ATTENTION_SCORES_KV_CACHE_BEHAVIOR, DENSE_PREFILL_ATTENTION_SCORES_LAYOUT_SOURCE,
    DENSE_PREFILL_ATTENTION_SCORES_PHASE_SCOPE, DENSE_PREFILL_ATTENTION_SCORES_TIMING_SCOPE,
    DENSE_PREFILL_LINEAR_REST_OF_PIPELINE_BACKEND, DENSE_TRANSPORT_LAYOUT,
    DenseMetalPrefillAttentionScoresFixture, DenseMetalPrefillAttentionScoresReceipt,
    DenseMetalPrefillAttentionScoresTiming, MACHINE_ID, REFERENCE_BACKEND, REQUESTED_BACKEND,
    RUNTIME_API, SELECTED_BACKEND, SLM_APPLE_M4_METAL_PHASE_ARTIFACT_KIND, SmokeComparison,
    argmax_index, compare_tiny_add_outputs, dense_metal_prefill_attention_scores_fixture,
    dense_prefill_attention_scores_expected, dense_prefill_attention_scores_kv_head,
    dense_prefill_attention_scores_shape_words, is_apple_m4_adapter_name,
    metal_dense_prefill_attention_scores_artifact_path,
};
use serde_json::json;

const RUN_DENSE_PREFILL_ATTENTION_SCORES_ENV: &str =
    "BITNET_RUN_M4_METAL_DENSE_PREFILL_ATTENTION_SCORES";
const DENSE_PREFILL_ATTENTION_SCORES_RECEIPT_ENV: &str =
    "BITNET_M4_METAL_DENSE_PREFILL_ATTENTION_SCORES_RECEIPT";
const DENSE_PREFILL_ATTENTION_SCORES_ARTIFACT_PATH_ENV: &str =
    "BITNET_M4_METAL_DENSE_PREFILL_ATTENTION_SCORES_ARTIFACT_PATH";
const ATTENTION_SCORE_TOLERANCE: f32 = 1e-5;

#[test]
fn dense_prefill_attention_scores_fixture_matches_cpu_reference() -> Result<(), Box<dyn Error>> {
    let fixture = dense_metal_prefill_attention_scores_fixture();
    let expected = dense_prefill_attention_scores_expected(
        &fixture.q,
        &fixture.k,
        fixture.prefill_tokens,
        fixture.attention_heads,
        fixture.kv_heads,
        fixture.head_dim,
    );

    assert_eq!(fixture.prefill_tokens, 4);
    assert_eq!(fixture.attention_heads, 14);
    assert_eq!(fixture.kv_heads, 2);
    assert_eq!(fixture.head_dim, 64);
    assert_eq!(
        fixture.q.len(),
        fixture.prefill_tokens * fixture.attention_heads * fixture.head_dim
    );
    assert_eq!(fixture.k.len(), fixture.prefill_tokens * fixture.kv_heads * fixture.head_dim);
    assert_eq!(
        fixture.expected_scores.len(),
        fixture.attention_heads * fixture.prefill_tokens * fixture.prefill_tokens
    );
    assert_eq!(dense_prefill_attention_scores_shape_words(&fixture), [4, 14, 2, 64]);
    assert_eq!(dense_prefill_attention_scores_kv_head(0, 14, 2), 0);
    assert_eq!(dense_prefill_attention_scores_kv_head(6, 14, 2), 0);
    assert_eq!(dense_prefill_attention_scores_kv_head(7, 14, 2), 1);
    assert_eq!(dense_prefill_attention_scores_kv_head(13, 14, 2), 1);
    assert!((fixture.scale - 0.125).abs() <= f32::EPSILON);
    assert!(fixture.expected_scores.iter().any(|value| *value != 0.0));

    compare_tiny_add_outputs(&fixture.expected_scores, &expected, 0.0)?;
    Ok(())
}

#[test]
fn dense_prefill_attention_scores_receipt_records_phase_scope_and_claim_boundary()
-> Result<(), Box<dyn Error>> {
    let fixture = dense_metal_prefill_attention_scores_fixture();
    let zero = SmokeComparison { max_abs_error: 0.0, mean_abs_error: 0.0 };
    let receipt = DenseMetalPrefillAttentionScoresReceipt::passed(
        metal_dense_prefill_attention_scores_artifact_path("2026-05-22"),
        zero,
        &fixture,
        DenseMetalPrefillAttentionScoresTiming::measured(0.25, 0.75),
    );

    assert_eq!(receipt.machine_id, MACHINE_ID);
    assert_eq!(receipt.artifact_kind, SLM_APPLE_M4_METAL_PHASE_ARTIFACT_KIND);
    assert_eq!(receipt.requested_backend, REQUESTED_BACKEND);
    assert_eq!(receipt.selected_backend, SELECTED_BACKEND);
    assert_eq!(receipt.runtime_api, RUNTIME_API);
    assert_eq!(receipt.reference_backend, REFERENCE_BACKEND);
    assert_eq!(receipt.target_backend, SELECTED_BACKEND);
    assert_eq!(receipt.rest_of_pipeline_backend, DENSE_PREFILL_LINEAR_REST_OF_PIPELINE_BACKEND);
    assert_eq!(receipt.kernel_id, DENSE_METAL_PREFILL_ATTENTION_SCORES_KERNEL_ID);
    assert_eq!(receipt.model_family, DENSE_MODEL_FAMILY);
    assert_eq!(receipt.kernel_family, DENSE_KERNEL_FAMILY);
    assert_eq!(receipt.execution_phase, DENSE_PREFILL_ATTENTION_SCORES_EXECUTION_PHASE);
    assert_eq!(receipt.phase_scope, DENSE_PREFILL_ATTENTION_SCORES_PHASE_SCOPE);
    assert_eq!(receipt.layout_source, DENSE_PREFILL_ATTENTION_SCORES_LAYOUT_SOURCE);
    assert_eq!(receipt.transport_layout, DENSE_TRANSPORT_LAYOUT);
    assert_eq!(receipt.kv_cache_behavior, DENSE_PREFILL_ATTENTION_SCORES_KV_CACHE_BEHAVIOR);
    assert!(!receipt.fallback_used);
    assert!(receipt.scores_match_cpu_reference);
    assert!(receipt.score_shape_matches_cpu_reference);
    assert!(receipt.head_mapping_matches_cpu_reference);
    assert_eq!(receipt.score_argmax_index, argmax_index(&fixture.expected_scores));
    assert_eq!(receipt.timing.cpu_reference_ms, 0.25);
    assert_eq!(receipt.timing.metal_phase_ms, 0.75);
    assert_eq!(receipt.timing.dispatch_readback_ms, 0.75);
    assert_eq!(receipt.timing.timing_delta_ms, 0.5);
    assert_eq!(receipt.timing.timing_scope, DENSE_PREFILL_ATTENTION_SCORES_TIMING_SCOPE);
    assert!(!receipt.timing.speedup_claim);
    assert_eq!(
        receipt.artifact_path,
        "ci/hardware/apple-m4-mac-mini/2026-05-22/slm-metal-phases/metal-dense-prefill-attention-scores.json"
    );

    let mut receipt_json = apple_backend_receipt_json(
        receipt.machine_id,
        receipt.artifact_kind,
        receipt.requested_backend,
        Some(receipt.selected_backend),
        receipt.runtime_api,
        "Apple M4 test adapter".to_string(),
        receipt.fallback_used,
        receipt.artifact_path.clone(),
        Some(receipt.kernel_id),
        receipt.result,
    )?;
    extend_dense_prefill_attention_scores_metrics(&mut receipt_json, &receipt, &fixture)?;

    assert_eq!(
        receipt_json["metal_phase"]["execution_phase"],
        DENSE_PREFILL_ATTENTION_SCORES_EXECUTION_PHASE
    );
    assert_eq!(receipt_json["layout"]["source"], DENSE_PREFILL_ATTENTION_SCORES_LAYOUT_SOURCE);
    assert_eq!(receipt_json["parity"]["scores_match_cpu_reference"], true);
    assert_eq!(receipt_json["parity"]["score_shape_matches_cpu_reference"], true);
    assert_eq!(receipt_json["parity"]["head_mapping_matches_cpu_reference"], true);
    assert_eq!(receipt_json["timing"]["speedup_claim"], false);
    assert_eq!(receipt_json["claim_boundary"]["metal_phase_only"], true);
    assert_eq!(receipt_json["claim_boundary"]["full_metal_inference"], false);
    assert_eq!(receipt_json["claim_boundary"]["bitnet_inference"], false);
    Ok(())
}

#[test]
fn dense_prefill_attention_scores_match_cpu_reference_when_enabled() -> Result<(), Box<dyn Error>> {
    if std::env::var(RUN_DENSE_PREFILL_ATTENTION_SCORES_ENV).as_deref() != Ok("1") {
        eprintln!(
            "skipping live M4 Metal dense prefill attention-score phase; set {RUN_DENSE_PREFILL_ATTENTION_SCORES_ENV}=1 to run it"
        );
        return Ok(());
    }
    if !dense_prefill_attention_scores_runtime_api_available() {
        return Err(io_error(
            "live M4 Metal dense prefill attention-score phase requires --features metal-runtime on Apple Silicon",
        ));
    }

    let fixture = dense_metal_prefill_attention_scores_fixture();
    let cpu_reference_start = Instant::now();
    let cpu_scores = dense_prefill_attention_scores_expected(
        &fixture.q,
        &fixture.k,
        fixture.prefill_tokens,
        fixture.attention_heads,
        fixture.kv_heads,
        fixture.head_dim,
    );
    let cpu_reference_duration = cpu_reference_start.elapsed();
    let metal_phase_start = Instant::now();
    let metal_output = run_dense_prefill_attention_scores_blocking(&fixture)?;
    let metal_phase_duration = metal_phase_start.elapsed();

    if !is_apple_m4_adapter_name(&metal_output.adapter_name) {
        return Err(io_error(format!(
            "M4-METAL-EX-002 dense prefill attention-score phase requires an Apple M4-family Metal adapter; found '{}'",
            metal_output.adapter_name
        )));
    }

    compare_tiny_add_outputs(&fixture.expected_scores, &cpu_scores, 0.0)?;
    let comparison =
        compare_tiny_add_outputs(&cpu_scores, &metal_output.scores, ATTENTION_SCORE_TOLERANCE)?;
    let artifact_path = std::env::var(DENSE_PREFILL_ATTENTION_SCORES_ARTIFACT_PATH_ENV)
        .or_else(|_| std::env::var(DENSE_PREFILL_ATTENTION_SCORES_RECEIPT_ENV))
        .unwrap_or_else(|_| {
            "ci/hardware/apple-m4-mac-mini/<date>/slm-metal-phases/metal-dense-prefill-attention-scores.json"
                .to_string()
        });
    let receipt = DenseMetalPrefillAttentionScoresReceipt::passed(
        artifact_path.clone(),
        comparison,
        &fixture,
        DenseMetalPrefillAttentionScoresTiming::measured(
            duration_ms(cpu_reference_duration),
            duration_ms(metal_phase_duration),
        ),
    );

    let mut receipt_json = apple_backend_receipt_json(
        receipt.machine_id,
        receipt.artifact_kind,
        receipt.requested_backend,
        Some(receipt.selected_backend),
        receipt.runtime_api,
        metal_output.adapter_name,
        receipt.fallback_used,
        receipt.artifact_path.clone(),
        Some(receipt.kernel_id),
        receipt.result,
    )?;
    extend_dense_prefill_attention_scores_metrics(&mut receipt_json, &receipt, &fixture)?;

    if let Ok(path) = std::env::var(DENSE_PREFILL_ATTENTION_SCORES_RECEIPT_ENV) {
        let output_path = receipt_output_path(&path);
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(output_path, serde_json::to_string_pretty(&receipt_json)?)?;
    }

    println!("{}", serde_json::to_string_pretty(&receipt_json)?);
    Ok(())
}

fn apple_backend_receipt_json(
    machine_id: &str,
    artifact_kind: &str,
    requested_backend: &str,
    selected_backend: Option<&str>,
    runtime_api: &str,
    chip: String,
    fallback_used: bool,
    artifact_path: String,
    kernel_id: Option<&str>,
    result: &str,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut receipt = AppleBackendReceipt::new(
        machine_id,
        artifact_kind,
        requested_backend,
        selected_backend,
        runtime_api,
        AppleResolvedDevice::new(chip).with_unified_memory(true),
        fallback_used,
        artifact_path,
    )
    .with_result(result);

    if let Some(kernel_id) = kernel_id {
        receipt = receipt.with_kernel_id(kernel_id);
    }

    receipt.validate()?;
    Ok(serde_json::to_value(receipt)?)
}

fn extend_dense_prefill_attention_scores_metrics(
    receipt_json: &mut serde_json::Value,
    receipt: &DenseMetalPrefillAttentionScoresReceipt,
    fixture: &DenseMetalPrefillAttentionScoresFixture,
) -> Result<(), Box<dyn Error>> {
    let Some(object) = receipt_json.as_object_mut() else {
        return Err(io_error("Apple receipt JSON is not an object"));
    };
    object.insert(
        "model".to_string(),
        json!({
            "family": receipt.model_family,
            "artifact": null,
            "source": "deterministic_dense_qwen_attention_score_fixture",
            "full_model_inference": false
        }),
    );
    object.insert(
        "slm_pipeline".to_string(),
        json!({
            "requested_backend": DENSE_PREFILL_LINEAR_REST_OF_PIPELINE_BACKEND,
            "selected_backend": DENSE_PREFILL_LINEAR_REST_OF_PIPELINE_BACKEND,
            "runtime_api": "cpu",
            "remaining_phases_backend": receipt.rest_of_pipeline_backend,
            "cpu_pipeline_for_remaining_phases": true,
            "full_inference_exercised": false
        }),
    );
    object.insert(
        "metal_phase".to_string(),
        json!({
            "requested_backend": receipt.requested_backend,
            "selected_backend": receipt.selected_backend,
            "runtime_api": receipt.runtime_api,
            "fallback_used": receipt.fallback_used,
            "kernel_id": receipt.kernel_id,
            "kernel_family": receipt.kernel_family,
            "execution_phase": receipt.execution_phase,
            "phase_scope": receipt.phase_scope,
            "prefill_tokens": receipt.prefill_tokens,
            "kv_cache_behavior": receipt.kv_cache_behavior,
            "timing_recorded": true,
            "full_autoregressive_decode": false,
            "full_metal_inference": false
        }),
    );
    object.insert(
        "dimensions".to_string(),
        json!({
            "attention_heads": receipt.attention_heads,
            "kv_heads": receipt.kv_heads,
            "head_dim": receipt.head_dim,
            "prefill_tokens": receipt.prefill_tokens,
            "q_shape": [receipt.prefill_tokens, receipt.attention_heads, receipt.head_dim],
            "k_shape": [receipt.prefill_tokens, receipt.kv_heads, receipt.head_dim],
            "score_shape": [receipt.attention_heads, receipt.prefill_tokens, receipt.prefill_tokens],
            "score_count": receipt.score_count,
            "scale": receipt.scale
        }),
    );
    object.insert(
        "layout".to_string(),
        json!({
            "source": receipt.layout_source,
            "transport_layout": receipt.transport_layout,
            "q_layout": "row_major_f32_tokens_by_attention_heads_by_head_dim",
            "k_layout": "row_major_f32_tokens_by_kv_heads_by_head_dim",
            "score_layout": "row_major_f32_attention_heads_by_query_token_by_key_token",
            "q_elements": fixture.q.len(),
            "k_elements": fixture.k.len(),
            "score_elements": fixture.expected_scores.len(),
            "softmax_applied": false,
            "causal_mask_applied": false,
            "consumes_dense_f32_directly": true,
            "dequantizes_before_compute": false
        }),
    );
    object.insert(
        "parity".to_string(),
        json!({
            "reference_backend": receipt.reference_backend,
            "target_backend": receipt.target_backend,
            "kernel_id": receipt.kernel_id,
            "kernel_family": receipt.kernel_family,
            "scores_match_cpu_reference": receipt.scores_match_cpu_reference,
            "score_shape_matches_cpu_reference": receipt.score_shape_matches_cpu_reference,
            "head_mapping_matches_cpu_reference": receipt.head_mapping_matches_cpu_reference,
            "max_abs_error": receipt.max_abs_error,
            "mean_abs_error": receipt.mean_abs_error,
            "tolerance": ATTENTION_SCORE_TOLERANCE,
            "score_argmax_index": receipt.score_argmax_index,
            "token_agreement_for_greedy": null
        }),
    );
    object.insert(
        "timing".to_string(),
        json!({
            "cpu_reference_ms": receipt.timing.cpu_reference_ms,
            "metal_phase_ms": receipt.timing.metal_phase_ms,
            "dispatch_readback_ms": receipt.timing.dispatch_readback_ms,
            "timing_delta_ms": receipt.timing.timing_delta_ms,
            "timing_scope": receipt.timing.timing_scope,
            "phase_local_only": true,
            "speedup_claim": receipt.timing.speedup_claim
        }),
    );
    object.insert(
        "claim_boundary".to_string(),
        json!({
            "metal_phase_only": true,
            "cpu_pipeline_for_remaining_phases": true,
            "no_mask_claim": true,
            "no_softmax_claim": true,
            "no_value_mix_claim": true,
            "no_output_projection_claim": true,
            "no_kv_cache_claim": true,
            "no_decode_claim": true,
            "no_sampling_claim": true,
            "no_detokenization_claim": true,
            "full_model_inference": false,
            "full_metal_inference": false,
            "bitnet_inference": false,
            "qk256_inference": false,
            "neural_engine_inference": false,
            "mpsgraph_inference": false,
            "broad_apple_silicon_claim": false,
            "speedup_claim": false
        }),
    );
    Ok(())
}

fn receipt_output_path(path: &str) -> std::path::PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        return path.to_path_buf();
    }

    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(path)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn io_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::other(message.into()))
}
