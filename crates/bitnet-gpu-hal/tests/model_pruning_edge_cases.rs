//! Edge-case tests for model_pruning module.
//!
//! Covers: PruningMethod, PruningGranularity, ScheduleKind,
//! PruningConfig, PruningSchedule, PruneMaskManager, MagnitudePruner,
//! StructuredPruner, MovementPruner, LotteryTicket, SensitivityAnalyzer,
//! PruningReport, PruningEngine.

use bitnet_gpu_hal::model_pruning::*;

// ── PruningMethod ───────────────────────────────────────────────

#[test]
fn pruning_method_all_variants() {
    let v = vec![
        PruningMethod::Magnitude,
        PruningMethod::Structured,
        PruningMethod::Movement,
        PruningMethod::LotteryTicket,
    ];
    assert_eq!(v.len(), 4);
}

#[test]
fn pruning_method_clone_eq() {
    let a = PruningMethod::Magnitude;
    let b = a;
    assert_eq!(a, b);
}

#[test]
fn pruning_method_debug() {
    let dbg = format!("{:?}", PruningMethod::Movement);
    assert!(dbg.contains("Movement"));
}

// ── PruningGranularity ──────────────────────────────────────────

#[test]
fn pruning_granularity_all_variants() {
    let v = vec![
        PruningGranularity::Weight,
        PruningGranularity::Channel,
        PruningGranularity::Head,
        PruningGranularity::Layer,
    ];
    assert_eq!(v.len(), 4);
}

#[test]
fn pruning_granularity_clone_eq() {
    let a = PruningGranularity::Channel;
    let b = a;
    assert_eq!(a, b);
}

// ── ScheduleKind ────────────────────────────────────────────────

#[test]
fn schedule_kind_all_variants() {
    let v = vec![ScheduleKind::OneShot, ScheduleKind::Linear, ScheduleKind::Cubic];
    assert_eq!(v.len(), 3);
}

// ── PruningConfig ───────────────────────────────────────────────

#[test]
fn pruning_config_magnitude() {
    let c = PruningConfig::magnitude(0.5);
    assert_eq!(c.method, PruningMethod::Magnitude);
    assert!((c.target_sparsity - 0.5).abs() < 1e-6);
    assert!(c.validate().is_ok());
}

#[test]
fn pruning_config_structured() {
    let c = PruningConfig::structured(0.3, PruningGranularity::Channel);
    assert_eq!(c.method, PruningMethod::Structured);
    assert_eq!(c.granularity, PruningGranularity::Channel);
    assert!(c.validate().is_ok());
}

#[test]
fn pruning_config_movement() {
    let c = PruningConfig::movement(0.5, 1000);
    assert_eq!(c.method, PruningMethod::Movement);
    assert_eq!(c.total_steps, 1000);
    assert!(c.validate().is_ok());
}

#[test]
fn pruning_config_lottery_ticket() {
    let c = PruningConfig::lottery_ticket(0.8);
    assert_eq!(c.method, PruningMethod::LotteryTicket);
    assert!(c.validate().is_ok());
}

#[test]
fn pruning_config_clone() {
    let c = PruningConfig::magnitude(0.5);
    let c2 = c.clone();
    assert_eq!(c2.method, PruningMethod::Magnitude);
}

// ── PruningSchedule ─────────────────────────────────────────────

#[test]
fn schedule_one_shot() {
    let cfg = PruningConfig::magnitude(0.5);
    let sched = PruningSchedule::from_config(&cfg);
    assert!((sched.target_sparsity() - 0.5).abs() < 1e-6);
}

#[test]
fn schedule_sparsity_at_step_one_shot() {
    let sched = PruningSchedule::new(ScheduleKind::OneShot, 0.5, 0, 100);
    // One-shot should be full target at any step >= begin
    let s = sched.sparsity_at_step(50);
    assert!((s - 0.5).abs() < 1e-6);
}

#[test]
fn schedule_linear() {
    let sched = PruningSchedule::new(ScheduleKind::Linear, 0.5, 0, 100);
    let s0 = sched.sparsity_at_step(0);
    let s50 = sched.sparsity_at_step(50);
    let s100 = sched.sparsity_at_step(100);
    // Linear: 0→0.5 over steps 0..100
    assert!(s0 <= s50);
    assert!(s50 <= s100);
    assert!((s100 - 0.5).abs() < 1e-5);
}

#[test]
fn schedule_cubic() {
    let sched = PruningSchedule::new(ScheduleKind::Cubic, 0.5, 0, 100);
    let s0 = sched.sparsity_at_step(0);
    let s100 = sched.sparsity_at_step(100);
    assert!(s0 <= s100);
    assert!((s100 - 0.5).abs() < 1e-5);
}

// ── PruneMaskManager ────────────────────────────────────────────

#[test]
fn mask_manager_default() {
    let mgr = PruneMaskManager::new();
    assert_eq!(mgr.tensor_count(), 0);
    assert!((mgr.overall_sparsity() - 0.0).abs() < 1e-6 || mgr.overall_sparsity().is_nan());
}

#[test]
fn mask_manager_register_and_set() {
    let mut mgr = PruneMaskManager::new();
    mgr.register("layer.0.weight", 100);
    mgr.set_mask("layer.0.weight", vec![true; 100]);
    let mask = mgr.get_mask("layer.0.weight").unwrap();
    assert_eq!(mask.len(), 100);
    assert!(mask.iter().all(|&m| m));
}

#[test]
fn mask_manager_apply() {
    let mut mgr = PruneMaskManager::new();
    mgr.register("w", 4);
    mgr.set_mask("w", vec![true, false, true, false]);
    let mut weights = vec![1.0, 2.0, 3.0, 4.0];
    mgr.apply("w", &mut weights);
    assert_eq!(weights[0], 1.0);
    assert_eq!(weights[1], 0.0);
    assert_eq!(weights[2], 3.0);
    assert_eq!(weights[3], 0.0);
}

#[test]
fn mask_manager_sparsity() {
    let mut mgr = PruneMaskManager::new();
    mgr.register("w", 4);
    mgr.set_mask("w", vec![true, false, true, false]);
    let s = mgr.sparsity("w").unwrap();
    assert!((s - 0.5).abs() < 1e-6);
}

#[test]
fn mask_manager_overall_sparsity() {
    let mut mgr = PruneMaskManager::new();
    mgr.register("a", 4);
    mgr.register("b", 4);
    mgr.set_mask("a", vec![true, false, true, false]);
    mgr.set_mask("b", vec![true, true, true, true]);
    let s = mgr.overall_sparsity();
    // (2 zeros / 8 total) = 0.25
    assert!((s - 0.25).abs() < 1e-6);
}

#[test]
fn mask_manager_tensor_names() {
    let mut mgr = PruneMaskManager::new();
    mgr.register("x", 10);
    mgr.register("y", 20);
    let names: Vec<&str> = mgr.tensor_names().collect();
    assert_eq!(names.len(), 2);
}

#[test]
fn mask_manager_get_mask_missing() {
    let mgr = PruneMaskManager::new();
    assert!(mgr.get_mask("nonexistent").is_none());
}

// ── MagnitudePruner ─────────────────────────────────────────────

#[test]
fn magnitude_pruner_compute_mask() {
    let cfg = PruningConfig::magnitude(0.5);
    let pruner = MagnitudePruner::new(cfg);
    let weights = vec![0.1, -0.5, 0.3, -0.8, 0.01, 0.9, -0.2, 0.4];
    let mask = pruner.compute_mask(&weights, 0.5);
    assert_eq!(mask.len(), 8);
    // ~50% should be false (pruned)
    let pruned_count = mask.iter().filter(|&&m| !m).count();
    assert_eq!(pruned_count, 4);
}

#[test]
fn magnitude_pruner_prune() {
    let cfg = PruningConfig::magnitude(0.25);
    let pruner = MagnitudePruner::new(cfg);
    let mut weights = vec![0.1, 0.5, 0.3, 0.8];
    let mask = pruner.prune(&mut weights);
    assert_eq!(mask.len(), 4);
    // Some weights should be zeroed
    let zeroed = weights.iter().filter(|&&w| w == 0.0).count();
    assert_eq!(zeroed, 1); // 25% of 4 = 1
}

#[test]
fn magnitude_pruner_target_sparsity() {
    let cfg = PruningConfig::magnitude(0.7);
    let pruner = MagnitudePruner::new(cfg);
    assert!((pruner.target_sparsity() - 0.7).abs() < 1e-6);
}

// ── StructuredPruner ────────────────────────────────────────────

#[test]
fn structured_pruner_channels() {
    let cfg = PruningConfig::structured(0.5, PruningGranularity::Channel);
    let pruner = StructuredPruner::new(cfg);
    // 4 output channels, 3 input features
    let weights = vec![
        0.1, 0.2, 0.3, // channel 0 (small)
        0.5, 0.6, 0.7, // channel 1 (medium)
        0.01, 0.02, 0.03, // channel 2 (smallest)
        0.8, 0.9, 1.0, // channel 3 (large)
    ];
    let mask = pruner.prune_channels(&weights, 4, 3);
    assert_eq!(mask.len(), 4);
    // 50% pruned → 2 channels
    let kept = mask.iter().filter(|&&m| m).count();
    assert_eq!(kept, 2);
}

#[test]
fn structured_pruner_heads() {
    let cfg = PruningConfig::structured(0.5, PruningGranularity::Head);
    let pruner = StructuredPruner::new(cfg);
    let head_norms = vec![0.1, 0.8, 0.3, 0.9]; // 4 heads
    let mask = pruner.prune_heads(&head_norms);
    assert_eq!(mask.len(), 4);
    let kept = mask.iter().filter(|&&m| m).count();
    assert_eq!(kept, 2);
}

#[test]
fn structured_pruner_layers() {
    let cfg = PruningConfig::structured(0.25, PruningGranularity::Layer);
    let pruner = StructuredPruner::new(cfg);
    let layer_scores = vec![0.5, 0.1, 0.8, 0.3]; // 4 layers
    let mask = pruner.prune_layers(&layer_scores);
    assert_eq!(mask.len(), 4);
    let kept = mask.iter().filter(|&&m| m).count();
    assert_eq!(kept, 3); // prune 25% = 1
}

#[test]
fn structured_pruner_granularity() {
    let cfg = PruningConfig::structured(0.5, PruningGranularity::Head);
    let pruner = StructuredPruner::new(cfg);
    assert_eq!(pruner.granularity(), PruningGranularity::Head);
}

// ── MovementPruner ──────────────────────────────────────────────

#[test]
fn movement_pruner_scores() {
    let weights = vec![0.1, -0.5, 0.3, -0.8];
    let grads = vec![0.2, 0.1, -0.1, -0.3];
    let scores = MovementPruner::movement_scores(&weights, &grads);
    assert_eq!(scores.len(), 4);
}

#[test]
fn movement_pruner_step() {
    let cfg = PruningConfig::movement(0.5, 10);
    let mut pruner = MovementPruner::new(cfg);
    let weights = vec![0.1, -0.5, 0.3, -0.8];
    let grads = vec![0.2, 0.1, -0.1, -0.3];
    let mask = pruner.step(&weights, &grads);
    assert_eq!(mask.len(), 4);
    assert_eq!(pruner.current_step(), 1);
}

#[test]
fn movement_pruner_current_sparsity() {
    let cfg = PruningConfig::movement(0.5, 100);
    let pruner = MovementPruner::new(cfg);
    let s = pruner.current_sparsity();
    assert!(s >= 0.0 && s <= 1.0);
}

// ── LotteryTicket ───────────────────────────────────────────────

#[test]
fn lottery_ticket_basic() {
    let cfg = PruningConfig::lottery_ticket(0.8);
    let lt = LotteryTicket::new(cfg, 5);
    assert_eq!(lt.round(), 0);
    assert_eq!(lt.max_rounds(), 5);
    assert!(!lt.is_complete());
}

#[test]
fn lottery_ticket_save_and_rewind() {
    let cfg = PruningConfig::lottery_ticket(0.5);
    let mut lt = LotteryTicket::new(cfg, 3);
    let initial = vec![1.0, 2.0, 3.0, 4.0];
    lt.save_initial_weights("w", &initial);
    assert_eq!(lt.initial_weights("w").unwrap(), &initial);

    // Simulate training
    let trained = vec![0.5, 0.1, 2.5, 0.05];
    let mask = lt.prune_round("w", &trained);
    assert_eq!(mask.len(), 4);

    // Rewind
    let mut weights = vec![0.0; 4];
    lt.rewind("w", &mut weights, &mask);
    // Active weights should be restored to initial values
    for (i, (&m, &w)) in mask.iter().zip(weights.iter()).enumerate() {
        if m {
            assert_eq!(w, initial[i]);
        } else {
            assert_eq!(w, 0.0);
        }
    }
}

#[test]
fn lottery_ticket_sparsity_at_round() {
    let cfg = PruningConfig::lottery_ticket(0.8);
    let lt = LotteryTicket::new(cfg, 5);
    let s0 = lt.sparsity_at_round(0);
    let s5 = lt.sparsity_at_round(5);
    assert!(s0 <= s5);
}

// ── SensitivityAnalyzer ─────────────────────────────────────────

#[test]
fn sensitivity_analyzer_default() {
    let sa = SensitivityAnalyzer::new();
    assert_eq!(sa.record_count(), 0);
}

#[test]
fn sensitivity_analyzer_record() {
    let mut sa = SensitivityAnalyzer::new();
    sa.record("layer.0", 0.1, 0.01);
    sa.record("layer.0", 0.5, 0.05);
    sa.record("layer.0", 0.9, 0.20);
    assert_eq!(sa.record_count(), 3);
}

#[test]
fn sensitivity_analyzer_curve() {
    let mut sa = SensitivityAnalyzer::new();
    sa.record("layer.0", 0.1, 0.01);
    sa.record("layer.0", 0.5, 0.05);
    let curve = sa.sensitivity_curve("layer.0");
    assert_eq!(curve.len(), 2);
}

#[test]
fn sensitivity_analyzer_recommend() {
    let mut sa = SensitivityAnalyzer::new();
    sa.record("layer.0", 0.5, 0.02);
    sa.record("layer.1", 0.5, 0.10);
    let recs = sa.recommend(0.5, 0.05);
    // layer.0 is less sensitive, should get higher sparsity recommendation
    assert!(!recs.is_empty());
}

// ── PruningReport ───────────────────────────────────────────────

#[test]
fn pruning_report_from_mask_manager() {
    let mut mgr = PruneMaskManager::new();
    mgr.register("w", 10);
    mgr.set_mask("w", vec![true, false, true, false, true, false, true, false, true, false]);
    let report = PruningReport::from_mask_manager(&mgr, PruningMethod::Magnitude);
    assert_eq!(report.total_params, 10);
    assert_eq!(report.pruned_params, 5);
    assert_eq!(report.remaining_params, 5);
    assert!((report.overall_sparsity - 0.5).abs() < 1e-6);
    assert_eq!(report.method, PruningMethod::Magnitude);
}

#[test]
fn pruning_report_from_weights() {
    let w1 = vec![0.0, 1.0, 0.0, 2.0];
    let w2 = vec![3.0, 0.0, 0.0, 4.0];
    let report =
        PruningReport::from_weights(&[("a", &w1[..]), ("b", &w2[..])], PruningMethod::Structured);
    assert_eq!(report.total_params, 8);
    // 4 zeros out of 8
    assert_eq!(report.pruned_params, 4);
}

#[test]
fn pruning_report_with_quality_metric() {
    let mut mgr = PruneMaskManager::new();
    mgr.register("w", 4);
    mgr.set_mask("w", vec![true; 4]);
    let report =
        PruningReport::from_mask_manager(&mgr, PruningMethod::Magnitude).with_quality_metric(0.95);
    assert_eq!(report.quality_metric, Some(0.95));
}

// ── PruningEngine ───────────────────────────────────────────────

#[test]
fn pruning_engine_magnitude() {
    let cfg = PruningConfig::magnitude(0.5);
    let mut engine = PruningEngine::new(cfg);
    engine.register_tensor("w", 8);
    let mut weights = vec![0.1, -0.5, 0.3, -0.8, 0.01, 0.9, -0.2, 0.4];
    engine.prune_step("w", &mut weights, None);
    let zeroed = weights.iter().filter(|&&w| w == 0.0).count();
    assert!(zeroed > 0);
}

#[test]
fn pruning_engine_advance_step() {
    let cfg = PruningConfig::magnitude(0.5);
    let mut engine = PruningEngine::new(cfg);
    assert_eq!(engine.current_step(), 0);
    engine.advance_step();
    assert_eq!(engine.current_step(), 1);
}

#[test]
fn pruning_engine_report() {
    let cfg = PruningConfig::magnitude(0.5);
    let mut engine = PruningEngine::new(cfg);
    engine.register_tensor("w", 4);
    let mut weights = vec![0.1, 0.5, 0.01, 0.8];
    engine.prune_step("w", &mut weights, None);
    let report = engine.report();
    assert_eq!(report.total_params, 4);
    assert!(report.pruned_params > 0);
}

#[test]
fn pruning_engine_config() {
    let cfg = PruningConfig::magnitude(0.3);
    let engine = PruningEngine::new(cfg);
    assert_eq!(engine.config().method, PruningMethod::Magnitude);
}
