//! Snapshot tests for OpenCL kernel configurations, sources, and outputs.
//!
//! These tests pin the stable surface of the OpenCL kernel modules so that
//! accidental changes to kernel sources, config defaults, work-size
//! computations, error messages, and output shapes are caught in review.

// ── Kernel source snapshots ─────────────────────────────────────────

mod kernel_sources {
    use bitnet_kernels::opencl_kernel_sources::{KernelProgramId, KernelSourceRegistry};

    #[test]
    fn matmul_kernel_source() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Matmul).unwrap();
        insta::assert_snapshot!(src.source);
    }

    #[test]
    fn matmul_tiled_kernel_source() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::MatmulTiled).unwrap();
        insta::assert_snapshot!(src.source);
    }

    #[test]
    fn softmax_kernel_source() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Softmax).unwrap();
        insta::assert_snapshot!(src.source);
    }

    #[test]
    fn rms_norm_kernel_source() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::RmsNorm).unwrap();
        insta::assert_snapshot!(src.source);
    }

    #[test]
    fn rope_kernel_source() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::RoPE).unwrap();
        insta::assert_snapshot!(src.source);
    }

    #[test]
    fn elementwise_kernel_source() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Elementwise).unwrap();
        insta::assert_snapshot!(src.source);
    }

    #[test]
    fn quantized_kernel_source() {
        let reg = KernelSourceRegistry::new();
        let src = reg.get(&KernelProgramId::Quantized).unwrap();
        insta::assert_snapshot!(src.source);
    }

    #[test]
    fn embedding_cl_source() {
        insta::assert_snapshot!(bitnet_kernels::opencl_embedding::EMBEDDING_CL);
    }

    #[test]
    fn attention_cl_source() {
        insta::assert_snapshot!(bitnet_kernels::opencl_attention::ATTENTION_CL);
    }

    #[test]
    fn ffn_cl_source() {
        insta::assert_snapshot!(bitnet_kernels::opencl_ffn::FFN_CL);
    }

    #[test]
    fn kv_cache_cl_source() {
        insta::assert_snapshot!(bitnet_kernels::opencl_kv_cache::KV_CACHE_CL);
    }

    #[test]
    fn quantized_matvec_cl_source() {
        insta::assert_snapshot!(bitnet_kernels::opencl_quantized::QUANTIZED_MATVEC_CL);
    }

    #[test]
    fn quantized_matvec_subgroup_cl_source() {
        insta::assert_snapshot!(bitnet_kernels::opencl_quantized::QUANTIZED_MATVEC_SUBGROUP_CL);
    }

    #[test]
    fn registry_all_entry_points() {
        let reg = KernelSourceRegistry::new();
        let mut pts = reg.all_entry_points();
        pts.sort();
        insta::assert_yaml_snapshot!(pts);
    }

    #[test]
    fn registry_kernel_count() {
        let reg = KernelSourceRegistry::new();
        insta::assert_snapshot!(format!("kernel_count={}", reg.len()));
    }
}

// ── Configuration snapshots ─────────────────────────────────────────

mod config_snapshots {
    use bitnet_kernels::opencl_work_size::{
        IntelArcWorkSizeHints, WorkSizeConfig, WorkSizeOptimizer,
    };

    #[test]
    fn intel_arc_work_size_hints_default() {
        let hints = IntelArcWorkSizeHints::default();
        insta::assert_debug_snapshot!(hints);
    }

    #[test]
    fn work_size_config_default() {
        let cfg = WorkSizeConfig::default();
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn work_size_1d_1024_elements() {
        let opt = WorkSizeOptimizer::intel_arc();
        let res = opt.optimize_1d(1024);
        insta::assert_debug_snapshot!(res);
    }

    #[test]
    fn work_size_2d_64x128() {
        let opt = WorkSizeOptimizer::intel_arc();
        let res = opt.optimize_2d(64, 128);
        insta::assert_debug_snapshot!(res);
    }

    #[test]
    fn work_size_3d_4x32x64() {
        let opt = WorkSizeOptimizer::intel_arc();
        let res = opt.optimize_3d(4, 32, 64);
        insta::assert_debug_snapshot!(res);
    }

    #[test]
    fn pipeline_config_validation_error_zero_layers() {
        use bitnet_kernels::opencl_pipeline::PipelineConfig;
        let cfg = PipelineConfig {
            num_layers: 0,
            hidden_dim: 2048,
            num_heads: 32,
            head_dim: 64,
            intermediate_dim: 5632,
            vocab_size: 32000,
            max_seq_len: 4096,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        let err = cfg.validate().unwrap_err();
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn pipeline_config_total_parameters_estimate() {
        use bitnet_kernels::opencl_pipeline::PipelineConfig;
        let cfg = PipelineConfig {
            num_layers: 24,
            hidden_dim: 2048,
            num_heads: 32,
            head_dim: 64,
            intermediate_dim: 5632,
            vocab_size: 32000,
            max_seq_len: 4096,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        insta::assert_snapshot!(format!("total_params={}", cfg.total_parameters_estimate()));
    }

    #[test]
    fn transformer_layer_config_default() {
        use bitnet_kernels::opencl_transformer::TransformerLayerConfig;
        let cfg = TransformerLayerConfig::default();
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn opencl_context_config_default() {
        use bitnet_kernels::opencl_context::OpenClContextConfig;
        let cfg = OpenClContextConfig::default();
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn device_constraints_a770_defaults() {
        use bitnet_kernels::opencl_registry::DeviceConstraints;
        let dc = DeviceConstraints::a770_defaults();
        insta::assert_debug_snapshot!(dc);
    }

    #[test]
    fn i2s_packed_format_default() {
        use bitnet_kernels::opencl_quantized::I2sPackedFormat;
        let fmt = I2sPackedFormat::default();
        insta::assert_debug_snapshot!(fmt);
    }
}

// ── Registry snapshots ──────────────────────────────────────────────

mod registry_snapshots {
    use bitnet_kernels::opencl_registry::{KernelOp, KernelRegistry, KernelVariant};

    #[test]
    fn kernel_op_all_list() {
        let ops: Vec<String> = KernelOp::ALL.iter().map(|o| o.to_string()).collect();
        insta::assert_yaml_snapshot!(ops);
    }

    #[test]
    fn kernel_variant_priorities() {
        let variants = [
            KernelVariant::OpenClTiled,
            KernelVariant::OpenClVectorized,
            KernelVariant::OpenClScalar,
            KernelVariant::CpuSimd,
            KernelVariant::CpuScalar,
        ];
        let prios: Vec<String> =
            variants.iter().map(|v| format!("{}={}", v, v.priority())).collect();
        insta::assert_yaml_snapshot!(prios);
    }

    #[test]
    fn a770_registry_summary() {
        let reg = KernelRegistry::with_default_a770_kernels();
        insta::assert_snapshot!(reg.summary());
    }

    #[test]
    fn a770_registry_gpu_coverage() {
        let reg = KernelRegistry::with_default_a770_kernels();
        insta::assert_snapshot!(format!("gpu_coverage={:.2}", reg.gpu_coverage()));
    }

    #[test]
    fn a770_registry_available_ops() {
        let reg = KernelRegistry::with_default_a770_kernels();
        let ops: Vec<String> = reg.available_ops().iter().map(|o| o.to_string()).collect();
        insta::assert_yaml_snapshot!(ops);
    }
}

// ── Output shape snapshots ──────────────────────────────────────────

mod output_shapes {
    use bitnet_kernels::opencl_embedding::EmbeddingConfig;

    #[test]
    fn embedding_config_dimensions() {
        let cfg = EmbeddingConfig::new(32000, 2048);
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn embedding_config_with_padding() {
        let cfg = EmbeddingConfig::new(32000, 2048).with_padding_idx(0);
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn attention_config_standard() {
        let cfg =
            bitnet_kernels::opencl_attention::AttentionConfig::new(32, 64, 4096, true).unwrap();
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn attention_config_gqa() {
        let cfg = bitnet_kernels::opencl_attention::AttentionConfig::new_gqa(32, 8, 64, 4096, true)
            .unwrap();
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn ffn_config_silu() {
        use bitnet_kernels::opencl_ffn::{ActivationType, FfnConfig};
        let cfg = FfnConfig::new(2048, 5632, ActivationType::SiLU).unwrap();
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn kv_cache_config_memory() {
        use bitnet_kernels::opencl_kv_cache::KvCacheConfig;
        let cfg = KvCacheConfig {
            max_seq_len: 4096,
            num_heads: 32,
            head_dim: 64,
            num_layers: 24,
            dtype_bytes: 4,
        };
        insta::assert_snapshot!(format!(
            "per_layer={} total={}",
            cfg.memory_per_layer(),
            cfg.total_memory()
        ));
    }

    #[test]
    fn quantized_matvec_config_qk256() {
        use bitnet_kernels::opencl_quantized::{I2sBlockLayout, QuantizedMatVecConfig};
        let cfg = QuantizedMatVecConfig::new(2048, 2048, I2sBlockLayout::Qk256);
        insta::assert_snapshot!(format!(
            "packed_cols={} scales_per_row={} total_weight_bytes={} block_size={}",
            cfg.packed_cols(),
            cfg.scales_per_row(),
            cfg.total_weight_bytes(),
            cfg.block_size(),
        ));
    }

    #[test]
    fn optimal_block_config_for_2048_cols() {
        use bitnet_kernels::opencl_quantized::OptimalBlockConfig;
        let opt = OptimalBlockConfig::for_cols(2048);
        insta::assert_debug_snapshot!(opt);
    }
}

// ── Error message snapshots ─────────────────────────────────────────

mod error_messages {
    #[test]
    fn kv_cache_error_layer_out_of_bounds() {
        use bitnet_kernels::opencl_kv_cache::KvCacheError;
        let err = KvCacheError::LayerOutOfBounds { requested: 25, available: 24 };
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn kv_cache_error_cache_full() {
        use bitnet_kernels::opencl_kv_cache::KvCacheError;
        let err = KvCacheError::CacheFull { max_len: 4096 };
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn kv_cache_error_dimension_mismatch() {
        use bitnet_kernels::opencl_kv_cache::KvCacheError;
        let err = KvCacheError::DimensionMismatch { expected: 2048, got: 1024 };
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn opencl_error_no_platform() {
        use bitnet_kernels::opencl_context::OpenClError;
        insta::assert_snapshot!(OpenClError::NoPlatform.to_string());
    }

    #[test]
    fn opencl_error_kernel_compile_failed() {
        use bitnet_kernels::opencl_context::OpenClError;
        let err = OpenClError::KernelCompileFailed {
            kernel_name: "matmul_tiled".into(),
            log: "undeclared identifier 'TILE_SIZE'".into(),
        };
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn opencl_error_buffer_allocation_failed() {
        use bitnet_kernels::opencl_context::OpenClError;
        let err = OpenClError::BufferAllocationFailed {
            size_bytes: 1_073_741_824,
            reason: "out of device memory".into(),
        };
        insta::assert_snapshot!(err.to_string());
    }

    #[test]
    fn pipeline_error_gpu_unavailable() {
        use bitnet_kernels::opencl_pipeline::PipelineError;
        insta::assert_snapshot!(PipelineError::GpuUnavailable.to_string());
    }

    #[test]
    fn pipeline_error_stage_failure() {
        use bitnet_kernels::opencl_pipeline::{PipelineError, PipelineStage};
        let err = PipelineError::StageFailure {
            stage: PipelineStage::Attention,
            reason: "workspace buffer too small".into(),
        };
        insta::assert_snapshot!(err.to_string());
    }
}
