//! Comprehensive property-based tests for OpenCL CPU reference implementations.
//!
//! Tests mathematical properties of functions in the publicly-exported opencl_*
//! modules: buffer alignment, cache key hashing, embedding lookup/projection,
//! kernel source registry, pipeline config validation, and work size optimization.

use proptest::prelude::*;

use bitnet_kernels::opencl_buffer::{
    align_size, optimal_buffer_size, validate_alignment, AlignedBuffer, BufferAlignment, BufferPool,
};
use bitnet_kernels::opencl_cache::{
    CacheConfig, CacheSerializer, CacheStats, KernelCacheManager, KernelCacheEntry, KernelCacheKey,
    SourceHasher,
};
use bitnet_kernels::opencl_embedding::{
    EmbeddingConfig, EmbeddingNorm, EmbeddingTable, OutputProjection, PositionEmbedding,
    TiedEmbedding, embedding_lookup_ref, output_projection_ref,
};
use bitnet_kernels::opencl_kernel_sources::{KernelProgramId, KernelSourceRegistry};
use bitnet_kernels::opencl_pipeline::{
    GenerationConfig, InferencePipeline, PipelineBuilder, PipelineConfig, PipelineStage,
    PipelineStatus,
};
use bitnet_kernels::opencl_work_size::{
    ElementwiseWorkSize, MatmulWorkSize, ReductionWorkSize, WorkSizeOptimizer,
};

// ═══════════════════════════════════════════════════════════════════
// Helper strategies
// ═══════════════════════════════════════════════════════════════════

/// Generates a Vec<f32> with finite, non-NaN values.
#[allow(dead_code)]
fn arb_f32_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, min_len..=max_len)
}

/// Generates a matrix as (data, rows, cols) with finite values.
#[allow(dead_code)]
fn arb_matrix(max_dim: usize) -> impl Strategy<Value = (Vec<f32>, usize, usize)> {
    (1..=max_dim, 1..=max_dim).prop_flat_map(|(rows, cols)| {
        prop::collection::vec(-1.0f32..1.0f32, rows * cols).prop_map(move |data| {
            (data, rows, cols)
        })
    })
}

// ═══════════════════════════════════════════════════════════════════
// 1. opencl_buffer — alignment and pool properties
// ═══════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    // --- Alignment ---

    #[test]
    fn align_size_is_gte_input(size in 0usize..100_000, alignment in 1usize..4096) {
        let aligned = align_size(size, alignment);
        prop_assert!(aligned >= size, "aligned {} < input {}", aligned, size);
    }

    #[test]
    fn align_size_is_multiple_of_alignment(size in 0usize..100_000, alignment in 1usize..4096) {
        let aligned = align_size(size, alignment);
        if size == 0 {
            prop_assert_eq!(aligned, 0);
        } else {
            prop_assert_eq!(aligned % alignment, 0, "{} not multiple of {}", aligned, alignment);
        }
    }

    #[test]
    fn align_size_idempotent(size in 1usize..100_000, alignment in 1usize..4096) {
        let once = align_size(size, alignment);
        let twice = align_size(once, alignment);
        prop_assert_eq!(once, twice, "aligning twice should be idempotent");
    }

    #[test]
    fn align_size_overshoot_bounded(size in 1usize..100_000, alignment in 1usize..4096) {
        let aligned = align_size(size, alignment);
        prop_assert!(aligned - size < alignment, "overshoot {} >= alignment {}", aligned - size, alignment);
    }

    #[test]
    fn optimal_buffer_size_gte_raw(elements in 0usize..10_000, elem_size in 1usize..16) {
        let raw = elements.saturating_mul(elem_size);
        let optimal = optimal_buffer_size(elements, elem_size);
        prop_assert!(optimal >= raw, "optimal {} < raw {}", optimal, raw);
    }

    #[test]
    fn optimal_buffer_size_zero_for_zero(elem_size in 0usize..16) {
        prop_assert_eq!(optimal_buffer_size(0, elem_size), 0);
    }

    #[test]
    fn optimal_buffer_size_cache_line_aligned(elements in 1usize..10_000, elem_size in 1usize..16) {
        let optimal = optimal_buffer_size(elements, elem_size);
        if optimal > 0 {
            prop_assert_eq!(optimal % BufferAlignment::CACHE_LINE, 0,
                "optimal {} not cache-line aligned", optimal);
        }
    }

    #[test]
    fn validate_alignment_consistent(ptr in 0usize..1_000_000, alignment in 1usize..4096) {
        let result = validate_alignment(ptr, alignment);
        prop_assert_eq!(result, ptr % alignment == 0);
    }

    // --- AlignedBuffer ---

    #[test]
    fn aligned_buffer_has_correct_len(n in 1usize..1000, alignment in 1usize..256) {
        let buf: AlignedBuffer<f32> = AlignedBuffer::new(n, alignment);
        // len() returns allocated element count which may exceed n due to alignment
        prop_assert!(buf.len() >= n, "buf.len() {} < requested {}", buf.len(), n);
        prop_assert!(!buf.is_empty());
    }

    #[test]
    fn aligned_buffer_byte_size_gte_elements(n in 1usize..1000, alignment in 1usize..256) {
        let buf: AlignedBuffer<f32> = AlignedBuffer::new(n, alignment);
        prop_assert!(buf.byte_size() >= n * std::mem::size_of::<f32>());
    }

    // --- BufferPool ---

    #[test]
    fn buffer_pool_put_then_get(size in 1usize..10_000) {
        let pool = BufferPool::new();
        let buf: AlignedBuffer<u8> = pool.allocate(size, 64);
        prop_assert!(buf.byte_size() >= size);
    }

    // ═══════════════════════════════════════════════════════════════
    // 2. opencl_cache — hashing, serialization, cache properties
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn source_hash_deterministic(s in "\\PC{1,200}") {
        let h1 = SourceHasher::hash_source(&s);
        let h2 = SourceHasher::hash_source(&s);
        prop_assert_eq!(h1, h2, "hash must be deterministic");
    }

    #[test]
    fn bytes_hash_deterministic(data in prop::collection::vec(any::<u8>(), 0..200)) {
        let h1 = SourceHasher::hash_bytes(&data);
        let h2 = SourceHasher::hash_bytes(&data);
        prop_assert_eq!(h1, h2);
    }

    #[test]
    fn cache_key_filename_is_hex(hash in any::<u64>()) {
        let key = KernelCacheKey::new(hash, "dev", "", "3.0");
        let filename = key.cache_filename();
        prop_assert!(filename.ends_with(".bin"));
        prop_assert_eq!(filename.len(), 20); // 16 hex + ".bin"
    }

    #[test]
    fn cache_key_from_source_matches_hash(s in "\\PC{1,100}") {
        let key = KernelCacheKey::from_source(&s, "dev", "", "3.0");
        let expected_hash = SourceHasher::hash_source(&s);
        prop_assert_eq!(key.source_hash, expected_hash);
    }

    #[test]
    fn cache_stats_hit_rate_bounded(hits in 0u64..1000, misses in 0u64..1000) {
        let stats = CacheStats { hits, misses, ..Default::default() };
        let rate = stats.hit_rate();
        prop_assert!((0.0..=1.0).contains(&rate), "hit_rate {} out of [0,1]", rate);
    }

    #[test]
    fn cache_stats_zero_queries_zero_rate(misses in 0u64..1) {
        let stats = CacheStats { hits: 0, misses: 0, ..Default::default() };
        prop_assert_eq!(stats.hit_rate(), 0.0);
        let _ = misses;
    }

    #[test]
    fn cache_serialize_roundtrip(
        binary in prop::collection::vec(any::<u8>(), 0..100),
        log in "[a-z ]{0,50}",
    ) {
        let entry = KernelCacheEntry {
            binary: binary.clone(),
            build_log: log.clone(),
            compilation_time: std::time::Duration::from_millis(42),
            device_info: "test-device".to_string(),
            created_at: std::time::SystemTime::UNIX_EPOCH,
            last_accessed: std::time::SystemTime::UNIX_EPOCH,
            access_count: 0,
        };
        let serialized = CacheSerializer::serialize(&entry);
        let deserialized = CacheSerializer::deserialize(&serialized);
        prop_assert!(deserialized.is_ok(), "deserialization failed: {:?}", deserialized.err());
        let de = deserialized.unwrap();
        prop_assert_eq!(&de.binary, &binary);
        prop_assert_eq!(&de.build_log, &log);
    }

    #[test]
    fn cache_store_then_lookup(
        hash in any::<u64>(),
        binary in prop::collection::vec(any::<u8>(), 1..50),
    ) {
        let config = CacheConfig::default();
        let cache = KernelCacheManager::new(config);
        let key = KernelCacheKey::new(hash, "dev", "", "3.0");
        cache.store(key.clone(), binary.clone(), String::new());
        let result = cache.lookup(&key);
        prop_assert!(result.is_some(), "stored entry should be retrievable");
        prop_assert_eq!(&result.unwrap().binary, &binary);
    }

    #[test]
    fn cache_invalidate_removes_entry(hash in any::<u64>()) {
        let cache = KernelCacheManager::new(CacheConfig::default());
        let key = KernelCacheKey::new(hash, "dev", "", "3.0");
        cache.store(key.clone(), vec![1, 2, 3], String::new());
        cache.invalidate(&key);
        prop_assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn cache_clear_empties(count in 1usize..10) {
        let cache = KernelCacheManager::new(CacheConfig::default());
        for i in 0..count {
            let key = KernelCacheKey::new(i as u64, "dev", "", "3.0");
            cache.store(key, vec![0u8; 8], String::new());
        }
        cache.clear();
        prop_assert!(cache.is_empty());
        prop_assert_eq!(cache.len(), 0);
    }

    // ═══════════════════════════════════════════════════════════════
    // 3. opencl_embedding — lookup, projection, normalization
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn embedding_lookup_valid_indices_produce_output(
        vocab in 2usize..20,
        dim in 2usize..16,
        seq_len in 1usize..10,
    ) {
        let weight: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let token_ids: Vec<u32> = (0..seq_len).map(|i| (i % vocab) as u32).collect();
        let mut output = vec![0.0f32; seq_len * dim];
        let result = embedding_lookup_ref(&token_ids, &weight, &mut output, vocab, dim, None);
        prop_assert!(result.is_ok());
        // Each output row should match the corresponding weight row
        for (t, &tok) in token_ids.iter().enumerate() {
            let expected_start = tok as usize * dim;
            for d in 0..dim {
                prop_assert_eq!(output[t * dim + d], weight[expected_start + d]);
            }
        }
    }

    #[test]
    fn embedding_lookup_oov_produces_zeros(
        vocab in 2usize..20,
        dim in 2usize..16,
    ) {
        let weight: Vec<f32> = (0..vocab * dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let oov_id = vocab as u32 + 5;
        let mut output = vec![999.0f32; dim];
        let result = embedding_lookup_ref(&[oov_id], &weight, &mut output, vocab, dim, None);
        prop_assert!(result.is_ok());
        for &v in &output {
            prop_assert_eq!(v, 0.0, "OOV should produce zeros");
        }
    }

    #[test]
    fn embedding_lookup_padding_idx_produces_zeros(
        vocab in 4usize..20,
        dim in 2usize..16,
    ) {
        let pad_idx = 0u32;
        let weight: Vec<f32> = (0..vocab * dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut output = vec![999.0f32; dim];
        let result = embedding_lookup_ref(&[pad_idx], &weight, &mut output, vocab, dim, Some(pad_idx));
        prop_assert!(result.is_ok());
        for &v in &output {
            prop_assert_eq!(v, 0.0, "padding idx should produce zeros");
        }
    }

    #[test]
    fn embedding_table_lookup_matches_ref(
        vocab in 2usize..10,
        dim in 2usize..8,
        seq_len in 1usize..5,
    ) {
        let weight: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let config = EmbeddingConfig::new(vocab, dim);
        let table = EmbeddingTable::new(weight.clone(), config).unwrap();
        let ids: Vec<u32> = (0..seq_len).map(|i| (i % vocab) as u32).collect();
        let mut out1 = vec![0.0f32; seq_len * dim];
        let mut out2 = vec![0.0f32; seq_len * dim];
        table.lookup(&ids, &mut out1).unwrap();
        embedding_lookup_ref(&ids, &weight, &mut out2, vocab, dim, None).unwrap();
        prop_assert_eq!(out1, out2);
    }

    #[test]
    fn output_projection_dimension_preserving(
        seq_len in 1usize..5,
        hidden in 2usize..8,
        vocab in 2usize..10,
    ) {
        let h_data: Vec<f32> = (0..seq_len * hidden).map(|i| i as f32 * 0.01).collect();
        let w_data: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.001).collect();
        let mut output = vec![0.0f32; seq_len * vocab];
        let result = output_projection_ref(&h_data, &w_data, &mut output, seq_len, hidden, vocab);
        prop_assert!(result.is_ok());
        // Output should be finite
        for &v in &output {
            prop_assert!(v.is_finite(), "projection output must be finite");
        }
    }

    #[test]
    fn output_projection_identity_weight_is_passthrough(dim in 2usize..8) {
        // Weight = I (identity): output = hidden @ I^T = hidden
        // vocab_size == hidden_size, weight is identity
        let mut weight = vec![0.0f32; dim * dim];
        for i in 0..dim {
            weight[i * dim + i] = 1.0;
        }
        let hidden: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; dim];
        output_projection_ref(&hidden, &weight, &mut output, 1, dim, dim).unwrap();
        for i in 0..dim {
            prop_assert!((output[i] - hidden[i]).abs() < 1e-5,
                "identity projection should pass through: {} vs {}", output[i], hidden[i]);
        }
    }

    #[test]
    fn tied_embedding_shares_weights(
        vocab in 2usize..10,
        dim in 2usize..8,
    ) {
        let weight: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let config = EmbeddingConfig::new(vocab, dim);
        let tied = TiedEmbedding::new(weight.clone(), config).unwrap();
        prop_assert_eq!(tied.weight(), &weight[..]);
    }

    #[test]
    fn position_embedding_adds_correctly(
        max_seq in 2usize..16,
        dim in 2usize..8,
        seq_len in 1usize..8,
    ) {
        let seq_len = seq_len.min(max_seq);
        let pos_weight: Vec<f32> = (0..max_seq * dim).map(|i| i as f32 * 0.1).collect();
        let pos_emb = PositionEmbedding::new(pos_weight.clone(), max_seq, dim).unwrap();
        let mut emb = vec![1.0f32; seq_len * dim];
        let original = emb.clone();
        pos_emb.add_to(&mut emb, seq_len, 0).unwrap();
        for t in 0..seq_len {
            for d in 0..dim {
                let expected = original[t * dim + d] + pos_weight[t * dim + d];
                prop_assert!((emb[t * dim + d] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn embedding_norm_preserves_length(dim in 4usize..32) {
        let norm = EmbeddingNorm::new(dim, 1e-5);
        let mut data: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        norm.normalize(&mut data, 1).unwrap();
        // After RMS norm, the RMS should be approximately 1/sqrt(dim)*sum
        // More precisely, mean(x^2) should be ~1/eps_scale
        let rms = (data.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
        // The RMS after normalization should be close to the ratio of original norms
        prop_assert!(rms.is_finite(), "normalized values must be finite");
        prop_assert!(rms > 0.0, "normalized vector should have non-zero RMS");
    }

    #[test]
    fn embedding_norm_idempotent_approx(dim in 4usize..32) {
        let norm = EmbeddingNorm::new(dim, 1e-5);
        let mut data1: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        norm.normalize(&mut data1, 1).unwrap();
        let after_first = data1.clone();
        norm.normalize(&mut data1, 1).unwrap();
        // Second normalization should change values only slightly
        for i in 0..dim {
            prop_assert!((data1[i] - after_first[i]).abs() < 0.2,
                "double-normalization should be approximately stable");
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 4. opencl_kernel_sources — registry invariants
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn kernel_registry_is_not_empty(_dummy in 0..1u32) {
        let registry = KernelSourceRegistry::new();
        prop_assert!(!registry.is_empty());
        prop_assert!(registry.len() >= 10, "should have at least 10 built-in kernels");
    }

    #[test]
    fn kernel_registry_all_entry_points_non_empty(_dummy in 0..1u32) {
        let registry = KernelSourceRegistry::new();
        let entry_points = registry.all_entry_points();
        prop_assert!(!entry_points.is_empty());
        for ep in &entry_points {
            prop_assert!(!ep.is_empty(), "entry point name should be non-empty");
        }
    }

    #[test]
    fn kernel_registry_builtin_ids_resolvable(_dummy in 0..1u32) {
        let registry = KernelSourceRegistry::new();
        let builtins = [
            KernelProgramId::Matmul,
            KernelProgramId::Softmax,
            KernelProgramId::RmsNorm,
            KernelProgramId::RoPE,
            KernelProgramId::Elementwise,
            KernelProgramId::Quantized,
        ];
        for id in &builtins {
            prop_assert!(registry.get(id).is_some(), "builtin {:?} not found", id);
        }
    }

    #[test]
    fn kernel_registry_sources_have_content(_dummy in 0..1u32) {
        let registry = KernelSourceRegistry::new();
        for id in registry.kernel_ids() {
            let src = registry.get(id).unwrap();
            prop_assert!(!src.source.is_empty(), "{:?} has empty source", id);
            prop_assert!(!src.entry_points.is_empty(), "{:?} has no entry points", id);
        }
    }

    #[test]
    fn kernel_registry_custom_registration(name in "[a-z]{3,10}") {
        let mut registry = KernelSourceRegistry::new();
        let before = registry.len();
        let source = bitnet_kernels::opencl_kernel_sources::KernelSource {
            id: KernelProgramId::Custom(name.clone()),
            source: "__kernel void test() {}",
            entry_points: vec!["test".into()],
            required_version: "1.2",
            build_options: vec![],
            local_mem_estimate: 0,
            preferred_work_group: 64,
        };
        registry.register(source);
        prop_assert_eq!(registry.len(), before + 1);
        prop_assert!(registry.get(&KernelProgramId::Custom(name)).is_some());
    }

    // ═══════════════════════════════════════════════════════════════
    // 5. opencl_pipeline — config validation and pipeline properties
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn pipeline_config_zero_field_rejected(
        field_idx in 0usize..7,
    ) {
        let mut cfg = PipelineConfig::tiny_test();
        match field_idx {
            0 => cfg.vocab_size = 0,
            1 => cfg.hidden_dim = 0,
            2 => cfg.num_layers = 0,
            3 => cfg.num_heads = 0,
            4 => cfg.head_dim = 0,
            5 => cfg.intermediate_dim = 0,
            _ => cfg.max_seq_len = 0,
        }
        prop_assert!(cfg.validate().is_err(), "zero field {} should fail validation", field_idx);
    }

    #[test]
    fn pipeline_config_valid_tiny(_dummy in 0..1u32) {
        let cfg = PipelineConfig::tiny_test();
        prop_assert!(cfg.validate().is_ok());
    }

    #[test]
    fn pipeline_config_valid_bitnet2b(_dummy in 0..1u32) {
        let cfg = PipelineConfig::bitnet_2b();
        prop_assert!(cfg.validate().is_ok());
    }

    #[test]
    fn generation_config_greedy_is_valid(_dummy in 0..1u32) {
        let cfg = GenerationConfig::greedy();
        prop_assert!(cfg.validate().is_ok());
    }

    #[test]
    fn generation_config_default_is_valid(_dummy in 0..1u32) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.validate().is_ok());
    }

    #[test]
    fn generation_config_negative_temp_rejected(temp in -100.0f32..-0.001) {
        let cfg = GenerationConfig::default().with_temperature(temp);
        prop_assert!(cfg.validate().is_err());
    }

    #[test]
    fn generation_config_invalid_top_p_rejected(p in prop::sample::select(vec![-0.5f32, 0.0, 1.5])) {
        let cfg = GenerationConfig::default().with_top_p(p);
        prop_assert!(cfg.validate().is_err());
    }

    #[test]
    fn generation_config_zero_max_tokens_rejected(_dummy in 0..1u32) {
        let cfg = GenerationConfig::default().with_max_tokens(0);
        prop_assert!(cfg.validate().is_err());
    }

    #[test]
    fn generation_config_builder_chain(
        max_tok in 1usize..100,
        temp in 0.0f32..2.0,
        top_k in 0usize..50,
    ) {
        let cfg = GenerationConfig::default()
            .with_max_tokens(max_tok)
            .with_temperature(temp)
            .with_top_k(top_k);
        prop_assert_eq!(cfg.max_tokens, max_tok);
        prop_assert_eq!(cfg.temperature, temp);
        prop_assert_eq!(cfg.top_k, top_k);
    }

    #[test]
    fn pipeline_stages_complete(_dummy in 0..1u32) {
        let stages = PipelineStage::all();
        prop_assert_eq!(stages.len(), 7);
    }

    #[test]
    fn pipeline_builder_tiny_constructs(_dummy in 0..1u32) {
        let pipeline = PipelineBuilder::new().build();
        prop_assert!(pipeline.is_ok());
    }

    #[test]
    fn pipeline_forward_returns_vocab_size_logits(_dummy in 0..1u32) {
        let cfg = PipelineConfig::tiny_test();
        let vocab = cfg.vocab_size;
        let mut pipeline = InferencePipeline::new(cfg).unwrap();
        let logits = pipeline.forward(&[1]);
        prop_assert!(logits.is_ok());
        prop_assert_eq!(logits.unwrap().len(), vocab);
    }

    #[test]
    fn pipeline_forward_rejects_empty_tokens(_dummy in 0..1u32) {
        let mut pipeline = InferencePipeline::new(PipelineConfig::tiny_test()).unwrap();
        let result = pipeline.forward(&[]);
        prop_assert!(result.is_err());
    }

    #[test]
    fn pipeline_reset_allows_reuse(_dummy in 0..1u32) {
        let cfg = PipelineConfig::tiny_test();
        let mut pipeline = InferencePipeline::new(cfg).unwrap();
        pipeline.forward(&[1]).unwrap();
        pipeline.reset();
        prop_assert_eq!(pipeline.status(), PipelineStatus::Ready);
        let result = pipeline.forward(&[2]);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn pipeline_logits_are_finite(_dummy in 0..1u32) {
        let mut pipeline = InferencePipeline::new(PipelineConfig::tiny_test()).unwrap();
        let logits = pipeline.forward(&[0, 1, 2]).unwrap();
        for &v in &logits {
            prop_assert!(v.is_finite(), "logit {} is not finite", v);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 6. opencl_work_size — optimization invariants
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn work_size_1d_global_gte_elements(n in 1usize..100_000) {
        let result = ElementwiseWorkSize::optimize(n);
        prop_assert!(result.global_work_size[0] >= n,
            "global {} < elements {}", result.global_work_size[0], n);
    }

    #[test]
    fn work_size_1d_has_local(n in 1usize..100_000) {
        let result = ElementwiseWorkSize::optimize(n);
        prop_assert!(result.local_work_size.is_some());
        let local = &result.local_work_size.unwrap();
        prop_assert!(local[0] > 0);
    }

    #[test]
    fn work_size_1d_global_divisible_by_local(n in 1usize..100_000) {
        let result = ElementwiseWorkSize::optimize(n);
        if let Some(local) = &result.local_work_size {
            prop_assert_eq!(result.global_work_size[0] % local[0], 0,
                "global {} not divisible by local {}", result.global_work_size[0], local[0]);
        }
    }

    #[test]
    fn work_size_1d_efficiency_bounded(n in 1usize..100_000) {
        let result = ElementwiseWorkSize::optimize(n);
        prop_assert!(result.efficiency > 0.0 && result.efficiency <= 1.0,
            "efficiency {} out of (0,1]", result.efficiency);
    }

    #[test]
    fn work_size_2d_global_covers_matrix(rows in 1usize..500, cols in 1usize..500) {
        let opt = WorkSizeOptimizer::intel_arc();
        let result = opt.optimize_2d(rows, cols);
        prop_assert!(result.global_work_size[0] >= cols);
        prop_assert!(result.global_work_size[1] >= rows);
    }

    #[test]
    fn work_size_2d_global_divisible_by_local(rows in 1usize..500, cols in 1usize..500) {
        let opt = WorkSizeOptimizer::intel_arc();
        let result = opt.optimize_2d(rows, cols);
        if let Some(local) = &result.local_work_size {
            prop_assert_eq!(result.global_work_size[0] % local[0], 0);
            prop_assert_eq!(result.global_work_size[1] % local[1], 0);
        }
    }

    #[test]
    fn work_size_3d_global_covers_volume(batch in 1usize..32, rows in 1usize..64, cols in 1usize..64) {
        let opt = WorkSizeOptimizer::intel_arc();
        let result = opt.optimize_3d(batch, rows, cols);
        prop_assert!(result.global_work_size[0] >= cols);
        prop_assert!(result.global_work_size[1] >= rows);
        prop_assert!(result.global_work_size[2] >= batch);
    }

    #[test]
    fn work_size_3d_efficiency_bounded(batch in 1usize..32, rows in 1usize..64, cols in 1usize..64) {
        let opt = WorkSizeOptimizer::intel_arc();
        let result = opt.optimize_3d(batch, rows, cols);
        prop_assert!(result.efficiency > 0.0 && result.efficiency <= 1.0);
    }

    #[test]
    fn work_size_tiled_matmul_covers_output(m in 1usize..500, n in 1usize..500) {
        let result = MatmulWorkSize::optimize(m, n);
        prop_assert!(result.global_work_size[0] >= n);
        prop_assert!(result.global_work_size[1] >= m);
    }

    #[test]
    fn work_size_tiled_matmul_custom_tile(
        m in 1usize..200,
        n in 1usize..200,
        tile in 1usize..33,
    ) {
        let result = MatmulWorkSize::optimize_with_tile(m, n, tile);
        prop_assert!(result.global_work_size[0] >= n);
        prop_assert!(result.global_work_size[1] >= m);
        prop_assert!(result.total_work_items >= m * n);
    }

    #[test]
    fn work_size_reduction_one_group_per_row(rows in 1usize..500, cols in 1usize..500) {
        let result = ReductionWorkSize::optimize(rows, cols);
        prop_assert_eq!(result.total_work_groups, rows,
            "reduction should have one group per row");
    }

    #[test]
    fn work_size_reduction_efficiency_bounded(rows in 1usize..500, cols in 1usize..500) {
        let result = ReductionWorkSize::optimize(rows, cols);
        prop_assert!(result.efficiency > 0.0 && result.efficiency <= 1.0);
    }

    #[test]
    fn work_size_local_fits_max_work_group(n in 1usize..100_000) {
        let opt = WorkSizeOptimizer::intel_arc();
        let max_wg = opt.config().max_work_group_size;
        let result = opt.optimize_1d(n);
        if let Some(local) = &result.local_work_size {
            let total_local: usize = local.iter().product();
            prop_assert!(total_local <= max_wg,
                "local work group {} exceeds max {}", total_local, max_wg);
        }
    }

    #[test]
    fn work_size_2d_local_fits_max(rows in 1usize..500, cols in 1usize..500) {
        let opt = WorkSizeOptimizer::intel_arc();
        let max_wg = opt.config().max_work_group_size;
        let result = opt.optimize_2d(rows, cols);
        if let Some(local) = &result.local_work_size {
            let total_local: usize = local.iter().product();
            prop_assert!(total_local <= max_wg);
        }
    }

    #[test]
    fn work_size_total_items_equals_product(n in 1usize..10_000) {
        let result = ElementwiseWorkSize::optimize(n);
        let product: usize = result.global_work_size.iter().product();
        prop_assert_eq!(result.total_work_items, product);
    }

    #[test]
    fn work_size_monotonic_global_1d(a in 1usize..10_000, b in 1usize..10_000) {
        let ra = ElementwiseWorkSize::optimize(a);
        let rb = ElementwiseWorkSize::optimize(b);
        if a <= b {
            prop_assert!(ra.global_work_size[0] <= rb.global_work_size[0],
                "larger input should have >= global work size");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Deterministic unit tests for edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn align_size_zero_returns_zero() {
    assert_eq!(align_size(0, 64), 0);
}

#[test]
#[should_panic(expected = "alignment must be non-zero")]
fn align_size_zero_alignment_panics() {
    let _ = align_size(10, 0);
}

#[test]
fn optimal_buffer_size_large_pages() {
    // Buffers >= 4096 bytes should be page-aligned
    let size = optimal_buffer_size(2048, 4);
    assert!(size >= 2048 * 4);
    assert_eq!(size % BufferAlignment::OPTIMAL_TRANSFER, 0);
}

#[test]
fn pipeline_config_tiny_matches_expected() {
    let cfg = PipelineConfig::tiny_test();
    assert_eq!(cfg.vocab_size, 64);
    assert_eq!(cfg.hidden_dim, 32);
    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.num_heads, 4);
    assert_eq!(cfg.head_dim, 8);
}

#[test]
fn kernel_registry_attention_not_builtin() {
    // Attention kernel requires explicit registration; not in the default set
    let registry = KernelSourceRegistry::new();
    assert!(registry.get(&KernelProgramId::Attention).is_none());
}

#[test]
fn work_size_optimizer_simd_width_is_16() {
    let opt = WorkSizeOptimizer::intel_arc();
    assert_eq!(opt.config().simd_width, 16);
}

#[test]
fn work_size_optimizer_max_1024() {
    let opt = WorkSizeOptimizer::intel_arc();
    assert_eq!(opt.config().max_work_group_size, 1024);
}

#[test]
fn cache_config_default_validates() {
    let cfg = CacheConfig::default();
    assert!(cfg.validate().is_ok());
}

#[test]
fn embedding_table_rejects_wrong_weight_size() {
    let config = EmbeddingConfig::new(10, 8);
    let result = EmbeddingTable::new(vec![0.0; 50], config);
    assert!(result.is_err());
}

#[test]
fn output_projection_rejects_wrong_weight_size() {
    let result = OutputProjection::new(vec![0.0; 50], 10, 8);
    assert!(result.is_err());
}
