//! Integration tests for the async GPU execution engine.
//!
//! All tests use the **mock backend** (no `oneapi` feature) so they run
//! without GPU hardware.

use bitnet_opencl::{
    AsyncGpuExecutor, ExecutionPlan, GpuError, GpuEvent, GpuFuture, PipelineScheduler,
    PipelineStage,
};

// =========================================================================
// AsyncGpuExecutor
// =========================================================================

#[test]
fn single_operation_submit_and_wait() {
    let exec = AsyncGpuExecutor::new(3);
    let fut = exec.submit_kernel(0, "matmul").unwrap();
    assert!(fut.event().is_complete());
    exec.wait_all().unwrap();
}

#[test]
fn submit_transfer_completes() {
    let exec = AsyncGpuExecutor::new(3);
    let fut = exec.submit_transfer(0, "upload_embeddings").unwrap();
    assert!(fut.event().is_complete());
    assert_eq!(exec.total_submitted(), 1);
}

#[test]
fn submit_readback_completes() {
    let exec = AsyncGpuExecutor::new(3);
    let fut = exec.submit_readback(2, "download_logits").unwrap();
    assert!(fut.event().is_complete());
}

#[test]
fn invalid_queue_returns_error() {
    let exec = AsyncGpuExecutor::new(2);
    let res = exec.submit_kernel(10, "bad_queue");
    assert!(res.is_err());
    match res.unwrap_err() {
        GpuError::InvalidQueue { requested, available } => {
            assert_eq!(requested, 10);
            assert_eq!(available, 2);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn flush_drains_all_pending() {
    let exec = AsyncGpuExecutor::new(1);
    exec.submit_kernel(0, "a").unwrap();
    exec.submit_kernel(0, "b").unwrap();
    exec.submit_kernel(0, "c").unwrap();
    assert_eq!(exec.pending_count(), 3);
    let flushed = exec.flush();
    assert_eq!(flushed, 3);
    assert_eq!(exec.pending_count(), 0);
}

#[test]
fn double_buffer_alternates() {
    let exec = AsyncGpuExecutor::new(2);
    assert_eq!(exec.active_buffer(), 0);
    exec.swap_buffer();
    assert_eq!(exec.active_buffer(), 1);
    exec.swap_buffer();
    assert_eq!(exec.active_buffer(), 0);
    // Swap several times.
    for i in 0..6 {
        exec.swap_buffer();
        assert_eq!(exec.active_buffer(), u64::from(i % 2 == 0));
    }
}

#[test]
fn total_submitted_tracks_lifetime() {
    let exec = AsyncGpuExecutor::new(3);
    assert_eq!(exec.total_submitted(), 0);
    exec.submit_kernel(0, "a").unwrap();
    exec.submit_transfer(1, "b").unwrap();
    exec.submit_readback(2, "c").unwrap();
    assert_eq!(exec.total_submitted(), 3);
    exec.flush();
    // Flush doesn't reset the counter.
    assert_eq!(exec.total_submitted(), 3);
}

#[test]
fn multiple_queues_independent() {
    let exec = AsyncGpuExecutor::new(3);
    // Submit to different queues concurrently.
    let f0 = exec.submit_transfer(0, "upload").unwrap();
    let f1 = exec.submit_kernel(1, "compute").unwrap();
    let f2 = exec.submit_readback(2, "download").unwrap();
    assert!(f0.event().is_complete());
    assert!(f1.event().is_complete());
    assert!(f2.event().is_complete());
    exec.wait_all().unwrap();
}

// =========================================================================
// GpuEvent
// =========================================================================

#[test]
fn event_error_state() {
    let e = GpuEvent::new();
    assert!(!e.is_done());
    e.signal_error();
    assert!(e.is_done());
    assert!(e.is_error());
    assert!(!e.is_complete());
}

// =========================================================================
// GpuFuture
// =========================================================================

#[test]
fn future_error_propagation() {
    let f = GpuFuture::<i32>::pending();
    f.resolve(Err(GpuError::OperationFailed { op: "bad_kernel".into() }));
    assert!(f.event().is_error());
    let result = f.wait();
    assert!(result.is_err());
}

#[test]
fn mock_backend_completes_synchronously() {
    let exec = AsyncGpuExecutor::new(1);
    for i in 0..10 {
        let fut = exec.submit_kernel(0, format!("kernel_{i}")).unwrap();
        // Every operation completes immediately in mock mode.
        assert!(fut.event().is_complete(), "mock operation {i} should be immediately complete");
    }
}

// =========================================================================
// ExecutionPlan
// =========================================================================

#[test]
fn empty_plan_is_noop() {
    let exec = AsyncGpuExecutor::new(3);
    let plan = ExecutionPlan::new();
    assert!(plan.is_empty());
    let events = exec.execute_plan(&plan).unwrap();
    assert!(events.is_empty());
}

#[test]
fn plan_three_stages_complete_in_order() {
    let exec = AsyncGpuExecutor::new(3);
    let mut plan = ExecutionPlan::new();
    let t = plan.add_op(PipelineStage::Transfer, "upload", &[]);
    let c = plan.add_op(PipelineStage::Compute, "matmul", &[t]);
    let _r = plan.add_op(PipelineStage::Readback, "download", &[c]);

    assert_eq!(plan.len(), 3);
    let events = exec.execute_plan(&plan).unwrap();
    assert_eq!(events.len(), 3);
    for e in &events {
        assert!(e.is_complete());
    }
}

#[test]
fn plan_invalid_forward_dependency() {
    let exec = AsyncGpuExecutor::new(3);
    let mut plan = ExecutionPlan::new();
    // Op 0 depends on op 1, which doesn't exist yet → error.
    plan.add_op(PipelineStage::Compute, "bad", &[1]);
    let result = exec.execute_plan(&plan);
    assert!(result.is_err());
}

// =========================================================================
// PipelineScheduler
// =========================================================================

#[test]
fn scheduler_respects_depth() {
    let mut sched = PipelineScheduler::new(2).unwrap();
    assert_eq!(sched.schedule_token(), Some(0));
    assert_eq!(sched.schedule_token(), Some(1));
    assert_eq!(sched.schedule_token(), None); // at capacity
}

#[test]
fn scheduler_drain_and_reschedule() {
    let mut sched = PipelineScheduler::new(1).unwrap();
    sched.schedule_token();
    assert_eq!(sched.drain_completed(), Some(0));
    assert_eq!(sched.completed_tokens(), 1);
    assert_eq!(sched.schedule_token(), Some(1));
}

#[test]
fn scheduler_dag_dependency_structure() {
    let mut sched = PipelineScheduler::new(2).unwrap();
    sched.schedule_token();
    sched.schedule_token();

    // 6 nodes: (T0, C0, R0, T1, C1, R1).
    assert_eq!(sched.dag_len(), 6);
    // Build plan and execute to verify validity.
    let exec = AsyncGpuExecutor::new(3);
    let events = sched.execute(&exec).unwrap();
    assert_eq!(events.len(), 6);
    for e in &events {
        assert!(e.is_complete());
    }
}

#[test]
fn scheduler_reset_clears_state() {
    let mut sched = PipelineScheduler::new(3).unwrap();
    sched.schedule_token();
    sched.schedule_token();
    sched.reset();
    assert!(sched.is_idle());
    assert_eq!(sched.dag_len(), 0);
    assert_eq!(sched.completed_tokens(), 0);
    // Can schedule again after reset.
    assert_eq!(sched.schedule_token(), Some(0));
}

#[test]
fn scheduler_invalid_depth() {
    assert!(PipelineScheduler::new(0).is_err());
    assert!(PipelineScheduler::new(5).is_err());
}

#[test]
fn full_pipeline_four_tokens() {
    let mut sched = PipelineScheduler::new(4).unwrap();
    for i in 0..4 {
        assert_eq!(sched.schedule_token(), Some(i));
    }
    assert_eq!(sched.schedule_token(), None);

    let exec = AsyncGpuExecutor::new(3);
    let events = sched.execute(&exec).unwrap();
    // 4 tokens × 3 stages = 12 nodes.
    assert_eq!(events.len(), 12);

    // Drain all.
    for i in 0..4 {
        assert_eq!(sched.drain_completed(), Some(i));
    }
    assert!(sched.is_idle());
    assert_eq!(sched.completed_tokens(), 4);
}
