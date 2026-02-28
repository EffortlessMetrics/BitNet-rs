//! Integration tests for model sharding and layer partitioning.

use bitnet_opencl::{
    CrossDeviceTransfer, DeviceBackend, DeviceDescriptor, LayerPartitioner, ModelArchitecture,
    ShardPlanner, ShardingError, ShardingStrategy,
};

// ── Helpers ─────────────────────────────────────────────────────────────

fn cuda_device(id: usize, mem_gb: u64) -> DeviceDescriptor {
    DeviceDescriptor {
        id,
        label: format!("CUDA:{id}"),
        memory_bytes: mem_gb * 1_073_741_824,
        backend: DeviceBackend::Cuda,
    }
}

fn ocl_device(id: usize, mem_gb: u64) -> DeviceDescriptor {
    DeviceDescriptor {
        id,
        label: format!("OpenCL:{id}"),
        memory_bytes: mem_gb * 1_073_741_824,
        backend: DeviceBackend::OpenCL,
    }
}

fn model(num_layers: usize) -> ModelArchitecture {
    ModelArchitecture {
        num_layers,
        memory_per_layer_bytes: 100 * 1_048_576,
        hidden_dim: 4096,
        num_attention_heads: 32,
        fixed_overhead_bytes: 50 * 1_048_576,
    }
}

// ── ShardPlanner tests ──────────────────────────────────────────────────

#[test]
fn single_device_all_layers() {
    let plan = ShardPlanner::new(vec![cuda_device(0, 8)], ShardingStrategy::LayerWise)
        .plan(&model(10))
        .unwrap();

    assert_eq!(plan.shards.len(), 1);
    assert_eq!(plan.shards[0].assignment.start_layer, 0);
    assert_eq!(plan.shards[0].assignment.end_layer, 10);
    assert!(plan.transfers.is_empty());
}

#[test]
fn two_device_even_split() {
    let plan =
        ShardPlanner::new(vec![cuda_device(0, 8), cuda_device(1, 8)], ShardingStrategy::LayerWise)
            .plan(&model(10))
            .unwrap();

    assert_eq!(plan.shards[0].assignment.num_layers(), 5);
    assert_eq!(plan.shards[1].assignment.num_layers(), 5);
}

#[test]
fn three_device_uneven_layer_count() {
    let plan = ShardPlanner::new(
        vec![cuda_device(0, 8), cuda_device(1, 8), cuda_device(2, 8)],
        ShardingStrategy::LayerWise,
    )
    .plan(&model(10))
    .unwrap();

    assert_eq!(plan.shards[0].assignment.num_layers(), 4);
    assert_eq!(plan.shards[1].assignment.num_layers(), 3);
    assert_eq!(plan.shards[2].assignment.num_layers(), 3);
}

#[test]
fn memory_constrained_placement() {
    let plan = ShardPlanner::new(vec![cuda_device(0, 8)], ShardingStrategy::LayerWise)
        .plan(&model(10))
        .unwrap();

    // 10 × 100 MB + 50 MB overhead = 1050 MB fits in 8 GB.
    assert!(plan.shards[0].estimated_memory_bytes < 8 * 1_073_741_824);
}

#[test]
fn heterogeneous_device_memory_limits() {
    let devices = vec![cuda_device(0, 16), ocl_device(1, 4)];
    let plan = ShardPlanner::new(devices, ShardingStrategy::LayerWise)
        .with_activation_bytes(4096)
        .plan(&model(10))
        .unwrap();

    assert_eq!(plan.shards.len(), 2);
    assert!(!plan.transfers[0].same_backend);
}

#[test]
fn cross_device_transfer_cost_same_backend() {
    let xfer = CrossDeviceTransfer {
        source_device_id: 0,
        target_device_id: 1,
        tensor_bytes: 24_000,
        same_backend: true,
    };
    assert_eq!(xfer.estimated_cost_us(), 2_000);
}

#[test]
fn cross_device_transfer_cost_different_backend() {
    let xfer = CrossDeviceTransfer {
        source_device_id: 0,
        target_device_id: 1,
        tensor_bytes: 24_000,
        same_backend: false,
    };
    assert_eq!(xfer.estimated_cost_us(), 4_000);
}

#[test]
fn empty_model_zero_layers() {
    let plan = ShardPlanner::new(vec![cuda_device(0, 8)], ShardingStrategy::LayerWise)
        .plan(&model(0))
        .unwrap();

    assert_eq!(plan.shards[0].assignment.num_layers(), 0);
}

#[test]
fn very_large_model_1000_layers() {
    let devices: Vec<DeviceDescriptor> = (0..10).map(|i| cuda_device(i, 128)).collect();
    let plan = ShardPlanner::new(devices, ShardingStrategy::LayerWise).plan(&model(1000)).unwrap();

    let total_layers: usize = plan.shards.iter().map(|s| s.assignment.num_layers()).sum();
    assert_eq!(total_layers, 1000);
    assert_eq!(plan.shards.len(), 10);
}

#[test]
fn pipeline_parallel_stage_assignment() {
    let plan = ShardPlanner::new(
        vec![cuda_device(0, 8), cuda_device(1, 8), cuda_device(2, 8)],
        ShardingStrategy::PipelineParallel,
    )
    .plan(&model(9))
    .unwrap();

    assert_eq!(plan.strategy, ShardingStrategy::PipelineParallel);
    for shard in &plan.shards {
        assert_eq!(shard.assignment.num_layers(), 3);
    }
}

#[test]
fn tensor_parallel_split_all_devices_see_all_layers() {
    let plan = ShardPlanner::new(
        vec![cuda_device(0, 8), cuda_device(1, 8)],
        ShardingStrategy::TensorParallel,
    )
    .with_activation_bytes(4096)
    .plan(&model(4))
    .unwrap();

    for shard in &plan.shards {
        assert_eq!(shard.assignment.start_layer, 0);
        assert_eq!(shard.assignment.end_layer, 4);
    }
}

#[test]
fn tensor_parallel_generates_transfers() {
    let plan = ShardPlanner::new(
        vec![cuda_device(0, 8), cuda_device(1, 8), cuda_device(2, 8)],
        ShardingStrategy::TensorParallel,
    )
    .with_activation_bytes(4096)
    .plan(&model(4))
    .unwrap();

    // C(3,2) = 3 pairs.
    assert_eq!(plan.num_transfers(), 3);
}

#[test]
fn no_devices_error() {
    let result = ShardPlanner::new(vec![], ShardingStrategy::LayerWise).plan(&model(10));
    assert_eq!(result, Err(ShardingError::NoDevices));
}

#[test]
fn insufficient_memory_error() {
    let result =
        ShardPlanner::new(vec![cuda_device(0, 1)], ShardingStrategy::LayerWise).plan(&model(20));

    assert!(matches!(result, Err(ShardingError::InsufficientMemory { .. })));
}

#[test]
fn total_memory_equals_shard_sum() {
    let plan =
        ShardPlanner::new(vec![cuda_device(0, 8), cuda_device(1, 8)], ShardingStrategy::LayerWise)
            .plan(&model(10))
            .unwrap();

    let sum: u64 = plan.shards.iter().map(|s| s.estimated_memory_bytes).sum();
    assert_eq!(plan.total_memory_bytes(), sum);
}

// ── LayerPartitioner tests ──────────────────────────────────────────────

#[test]
fn partitioner_even_two_devices() {
    let plan = LayerPartitioner::new(vec![cuda_device(0, 8), cuda_device(1, 8)])
        .partition(&model(10))
        .unwrap();

    assert_eq!(plan.entries[0].assignment.num_layers(), 5);
    assert_eq!(plan.entries[1].assignment.num_layers(), 5);
}

#[test]
fn partitioner_memory_constrained_heterogeneous() {
    let devices = vec![
        DeviceDescriptor {
            id: 0,
            label: "small".into(),
            memory_bytes: 350 * 1_048_576,
            backend: DeviceBackend::Cuda,
        },
        cuda_device(1, 8),
    ];
    let plan =
        LayerPartitioner::new(devices).memory_constrained(true).partition(&model(10)).unwrap();

    // 350 MB − 50 MB overhead = 300 MB → 3 layers on device 0.
    assert_eq!(plan.entries[0].assignment.num_layers(), 3);
    assert_eq!(plan.entries[1].assignment.num_layers(), 7);
}

#[test]
fn partitioner_compute_cost_per_layer() {
    let plan = LayerPartitioner::new(vec![cuda_device(0, 8)])
        .with_cost_per_layer(42)
        .partition(&model(10))
        .unwrap();
    assert_eq!(plan.total_compute_cost, 420);
}

#[test]
fn partitioner_transfer_cost_multi_device() {
    let plan = LayerPartitioner::new(vec![cuda_device(0, 8), cuda_device(1, 8)])
        .with_activation_bytes(12_000)
        .partition(&model(10))
        .unwrap();

    assert_eq!(plan.total_transfer_cost_us, 1_000);
}

#[test]
fn layers_contiguous_across_all_shards() {
    let plan = ShardPlanner::new(
        vec![cuda_device(0, 8), cuda_device(1, 8), cuda_device(2, 8), cuda_device(3, 8)],
        ShardingStrategy::LayerWise,
    )
    .plan(&model(17))
    .unwrap();

    // Verify contiguity: each shard starts where previous ended.
    for pair in plan.shards.windows(2) {
        assert_eq!(pair[0].assignment.end_layer, pair[1].assignment.start_layer,);
    }
    // Total layers add up.
    let total: usize = plan.shards.iter().map(|s| s.assignment.num_layers()).sum();
    assert_eq!(total, 17);
}
