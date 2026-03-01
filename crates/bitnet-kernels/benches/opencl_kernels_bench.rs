//! Criterion benchmarks for OpenCL CPU reference implementations.
//!
//! Establishes performance baselines for all OpenCL module CPU paths:
//! attention, FFN, embedding, normalization, work-size calculation,
//! activation functions, and buffer alignment utilities.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use bitnet_kernels::opencl_attention::{
    AttentionConfig, AttentionMask, AttentionScores, GroupedQueryAttention,
    MultiHeadAttentionRef, scaled_dot_product_attention_ref,
};
use bitnet_kernels::opencl_buffer::{
    AlignedBuffer, BufferPool, align_size, optimal_buffer_size, validate_alignment,
};
use bitnet_kernels::opencl_embedding::{
    EmbeddingConfig, EmbeddingNorm, EmbeddingTable, PositionEmbedding,
    embedding_lookup_ref, output_projection_ref,
};
use bitnet_kernels::opencl_ffn::{ActivationType, ffn_forward_ref, gated_ffn_forward_ref};
use bitnet_kernels::opencl_pipeline::{InferencePipeline, PipelineConfig};
use bitnet_kernels::opencl_transformer::{
    KvCacheState, LayerWeights, ResidualConnection, TransformerLayerConfig,
    transformer_layer_forward_ref,
};
use bitnet_kernels::opencl_work_size::{
    ElementwiseWorkSize, MatmulWorkSize, ReductionWorkSize, WorkSizeOptimizer,
};

// ── Deterministic data generators ────────────────────────────────

fn make_f32(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i % 127) as f32 - 63.0) * 0.01).collect()
}

fn make_f32_positive(len: usize) -> Vec<f32> {
    (0..len).map(|i| (i % 127) as f32 * 0.01 + 0.001).collect()
}

fn make_token_ids(n: usize, vocab_size: usize) -> Vec<u32> {
    (0..n).map(|i| (i % vocab_size) as u32).collect()
}

// ── 1. Attention benchmarks ──────────────────────────────────────

fn bench_scaled_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/scaled_dot_product");
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for &seq_len in &[32, 64, 128, 256] {
        let kv_len = seq_len;
        let q = make_f32(seq_len * head_dim);
        let k = make_f32(kv_len * head_dim);
        let v = make_f32(kv_len * head_dim);
        let mut output = vec![0.0f32; seq_len * head_dim];

        group.throughput(Throughput::Elements((seq_len * kv_len * head_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("causal", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    scaled_dot_product_attention_ref(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        black_box(&mut output),
                        seq_len,
                        kv_len,
                        head_dim,
                        scale,
                        true,
                    );
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("no_mask", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    scaled_dot_product_attention_ref(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        black_box(&mut output),
                        seq_len,
                        kv_len,
                        head_dim,
                        scale,
                        false,
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_multi_head_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/multi_head");
    let num_heads = 8;
    let head_dim = 64;

    for &seq_len in &[16, 32, 64] {
        let kv_len = seq_len;
        let config = AttentionConfig::new(num_heads, head_dim, 512, true).unwrap();
        let total = num_heads * head_dim;
        let q = make_f32(seq_len * total);
        let k = make_f32(kv_len * total);
        let v = make_f32(kv_len * total);
        let mut output = vec![0.0f32; seq_len * total];

        group.throughput(Throughput::Elements(
            (seq_len * kv_len * num_heads * head_dim) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("mha", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    MultiHeadAttentionRef::forward(
                        black_box(&config),
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        black_box(&mut output),
                        seq_len,
                        kv_len,
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_gqa(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/gqa");
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = 64;

    for &seq_len in &[16, 32, 64] {
        let kv_len = seq_len;
        let q = make_f32(seq_len * num_heads * head_dim);
        let k = make_f32(kv_len * num_kv_heads * head_dim);
        let v = make_f32(kv_len * num_kv_heads * head_dim);
        let mut output = vec![0.0f32; seq_len * num_heads * head_dim];

        group.bench_with_input(
            BenchmarkId::new("gqa_32h_8kv", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    GroupedQueryAttention::forward(
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        black_box(&mut output),
                        seq_len,
                        kv_len,
                        true,
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_attention_scores(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/scores");
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for &seq_len in &[64, 128, 256] {
        let kv_len = seq_len;
        let q = make_f32(seq_len * head_dim);
        let k = make_f32(kv_len * head_dim);

        group.bench_with_input(
            BenchmarkId::new("compute_raw", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    AttentionScores::compute_raw(
                        black_box(&q),
                        black_box(&k),
                        seq_len,
                        kv_len,
                        head_dim,
                        scale,
                    );
                });
            },
        );

        let mask = AttentionMask::causal(seq_len, kv_len, 0);
        let mut scores =
            AttentionScores::compute_raw(&q, &k, seq_len, kv_len, head_dim, scale);
        group.bench_with_input(
            BenchmarkId::new("apply_mask", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    scores.apply_mask(black_box(&mask));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("softmax", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let mut s = AttentionScores::compute_raw(
                        &q, &k, seq_len, kv_len, head_dim, scale,
                    );
                    s.softmax();
                    black_box(&s);
                });
            },
        );
    }
    group.finish();
}

// ── 2. FFN benchmarks ────────────────────────────────────────────

fn bench_ffn(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffn/standard");
    let activation = ActivationType::SiLU;

    for &(hidden, inter) in &[(64, 176), (128, 352), (256, 704)] {
        let seq_len = 1;
        let x = make_f32(seq_len * hidden);
        let w_up = make_f32(hidden * inter);
        let w_down = make_f32(inter * hidden);
        let mut output = vec![0.0f32; seq_len * hidden];

        group.throughput(Throughput::Elements((2 * hidden * inter) as u64));
        group.bench_with_input(
            BenchmarkId::new("silu", format!("{hidden}x{inter}")),
            &hidden,
            |b, _| {
                b.iter(|| {
                    ffn_forward_ref(
                        black_box(&x),
                        black_box(&w_up),
                        black_box(&w_down),
                        black_box(&mut output),
                        seq_len,
                        hidden,
                        inter,
                        activation,
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_gated_ffn(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffn/gated");
    let activation = ActivationType::SiLU;

    for &(hidden, inter) in &[(64, 176), (128, 352), (256, 704)] {
        let seq_len = 1;
        let x = make_f32(seq_len * hidden);
        let w_gate = make_f32(hidden * inter);
        let w_up = make_f32(hidden * inter);
        let w_down = make_f32(inter * hidden);
        let mut output = vec![0.0f32; seq_len * hidden];

        group.throughput(Throughput::Elements((3 * hidden * inter) as u64));
        group.bench_with_input(
            BenchmarkId::new("silu", format!("{hidden}x{inter}")),
            &hidden,
            |b, _| {
                b.iter(|| {
                    gated_ffn_forward_ref(
                        black_box(&x),
                        black_box(&w_gate),
                        black_box(&w_up),
                        black_box(&w_down),
                        black_box(&mut output),
                        seq_len,
                        hidden,
                        inter,
                        activation,
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

// ── 3. Normalization benchmarks ──────────────────────────────────

fn bench_embedding_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization/rms_norm");

    for &dim in &[64, 256, 1024, 4096] {
        let n_tokens = 4;
        let mut data = make_f32(n_tokens * dim);
        let norm = EmbeddingNorm::new(dim, 1e-5);

        group.throughput(Throughput::Elements((n_tokens * dim) as u64));
        group.bench_with_input(BenchmarkId::new("embedding_norm", dim), &dim, |b, _| {
            b.iter(|| {
                let mut d = data.clone();
                norm.normalize(black_box(&mut d), n_tokens).unwrap();
                black_box(&d);
            });
        });

        // Also benchmark single-token normalization
        let single = make_f32(dim);
        let single_norm = EmbeddingNorm::new(dim, 1e-5);
        group.bench_with_input(
            BenchmarkId::new("single_token", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    let mut d = single.clone();
                    single_norm.normalize(black_box(&mut d), 1).unwrap();
                    black_box(&d);
                });
            },
        );

        let _ = (&mut data, &single); // suppress unused-mut
    }
    group.finish();
}

// ── 4. Embedding benchmarks ──────────────────────────────────────

fn bench_embedding_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding/lookup");
    let vocab_size = 32000;
    let embedding_dim = 256;
    let weight = make_f32(vocab_size * embedding_dim);
    let config = EmbeddingConfig::new(vocab_size, embedding_dim);
    let table = EmbeddingTable::new(weight.clone(), config).unwrap();

    for &batch in &[1, 8, 32, 128] {
        let tokens = make_token_ids(batch, vocab_size);
        let mut output = vec![0.0f32; batch * embedding_dim];

        group.throughput(Throughput::Elements((batch * embedding_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("table", batch),
            &batch,
            |b, _| {
                b.iter(|| {
                    table
                        .lookup(black_box(&tokens), black_box(&mut output))
                        .unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ref_fn", batch),
            &batch,
            |b, _| {
                b.iter(|| {
                    embedding_lookup_ref(
                        black_box(&tokens),
                        black_box(&weight),
                        black_box(&mut output),
                        vocab_size,
                        embedding_dim,
                        None,
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_output_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding/output_projection");

    for &(hidden, vocab) in &[(64, 1024), (128, 4096), (256, 8192)] {
        let seq_len = 1;
        let hidden_data = make_f32(seq_len * hidden);
        let weight = make_f32(vocab * hidden);
        let mut output = vec![0.0f32; seq_len * vocab];

        group.throughput(Throughput::Elements((seq_len * hidden * vocab) as u64));
        group.bench_with_input(
            BenchmarkId::new("matmul_transpose", format!("{hidden}x{vocab}")),
            &hidden,
            |b, _| {
                b.iter(|| {
                    output_projection_ref(
                        black_box(&hidden_data),
                        black_box(&weight),
                        black_box(&mut output),
                        seq_len,
                        hidden,
                        vocab,
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_position_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding/position");
    let embedding_dim = 256;
    let max_seq_len = 512;
    let pos_weight = make_f32(max_seq_len * embedding_dim);
    let pos_emb = PositionEmbedding::new(pos_weight, max_seq_len, embedding_dim).unwrap();

    for &seq_len in &[1, 8, 32, 128] {
        let mut embeddings = make_f32(seq_len * embedding_dim);

        group.throughput(Throughput::Elements((seq_len * embedding_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("add_to", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    let mut emb = embeddings.clone();
                    pos_emb
                        .add_to(black_box(&mut emb), seq_len, 0)
                        .unwrap();
                    black_box(&emb);
                });
            },
        );
        let _ = &mut embeddings; // suppress unused-mut
    }
    group.finish();
}

// ── 5. Softmax benchmarks (via AttentionScores) ──────────────────

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    let head_dim = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for &len in &[128, 512, 2048] {
        let seq_len = 1;
        let kv_len = len;
        let q = make_f32(seq_len * head_dim);
        let k = make_f32(kv_len * head_dim);

        group.throughput(Throughput::Elements(kv_len as u64));
        group.bench_with_input(
            BenchmarkId::new("row_softmax", len),
            &len,
            |b, _| {
                b.iter(|| {
                    let mut scores = AttentionScores::compute_raw(
                        &q, &k, seq_len, kv_len, head_dim, scale,
                    );
                    scores.softmax();
                    black_box(&scores);
                });
            },
        );
    }

    // Multi-row softmax
    for &rows in &[16, 64, 256] {
        let kv_len = 128;
        let q = make_f32(rows * head_dim);
        let k = make_f32(kv_len * head_dim);

        group.throughput(Throughput::Elements((rows * kv_len) as u64));
        group.bench_with_input(
            BenchmarkId::new("multi_row", format!("{rows}x{kv_len}")),
            &rows,
            |b, _| {
                b.iter(|| {
                    let mut scores = AttentionScores::compute_raw(
                        &q, &k, rows, kv_len, head_dim, scale,
                    );
                    scores.softmax();
                    black_box(&scores);
                });
            },
        );
    }
    group.finish();
}

// ── 6. Activation function benchmarks ────────────────────────────

fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");

    let activations = [
        ("silu", ActivationType::SiLU),
        ("gelu", ActivationType::GELU),
        ("gelu_approx", ActivationType::GELUApprox),
        ("relu", ActivationType::ReLU),
    ];

    for &len in &[256, 1024, 4096] {
        let data = make_f32(len);

        for &(name, act) in &activations {
            group.throughput(Throughput::Elements(len as u64));
            group.bench_with_input(
                BenchmarkId::new(name, len),
                &len,
                |b, _| {
                    b.iter(|| {
                        let result: f32 = data.iter().map(|&x| act.apply(x)).sum();
                        black_box(result);
                    });
                },
            );
        }
    }
    group.finish();
}

// ── 7. Work size calculation benchmarks ──────────────────────────

fn bench_work_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_size");
    let optimizer = WorkSizeOptimizer::intel_arc();

    // 1D optimization
    for &elements in &[1024, 65536, 1_048_576, 16_777_216] {
        group.bench_with_input(
            BenchmarkId::new("optimize_1d", elements),
            &elements,
            |b, &n| {
                b.iter(|| black_box(optimizer.optimize_1d(n)));
            },
        );
    }

    // 2D optimization
    for &(rows, cols) in &[(64, 64), (256, 256), (1024, 1024)] {
        group.bench_with_input(
            BenchmarkId::new("optimize_2d", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, &(r, c)| {
                b.iter(|| black_box(optimizer.optimize_2d(r, c)));
            },
        );
    }

    // 3D optimization
    for &(batch, rows, cols) in &[(4, 64, 64), (8, 128, 128)] {
        group.bench_with_input(
            BenchmarkId::new("optimize_3d", format!("{batch}x{rows}x{cols}")),
            &(batch, rows, cols),
            |b, &(ba, r, c)| {
                b.iter(|| black_box(optimizer.optimize_3d(ba, r, c)));
            },
        );
    }

    // Tiled matmul
    for &(m, n) in &[(128, 128), (512, 512), (1024, 1024)] {
        group.bench_with_input(
            BenchmarkId::new("tiled_matmul", format!("{m}x{n}")),
            &(m, n),
            |b, &(m, n)| {
                b.iter(|| black_box(optimizer.optimize_tiled_matmul(m, n, 16)));
            },
        );
    }

    // Reduction
    for &(rows, cols) in &[(64, 1024), (256, 4096)] {
        group.bench_with_input(
            BenchmarkId::new("reduction", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, &(r, c)| {
                b.iter(|| black_box(optimizer.optimize_reduction(r, c)));
            },
        );
    }

    // Convenience wrappers
    group.bench_function("MatmulWorkSize_512x512", |b| {
        b.iter(|| black_box(MatmulWorkSize::optimize(512, 512)));
    });
    group.bench_function("ReductionWorkSize_256x4096", |b| {
        b.iter(|| black_box(ReductionWorkSize::optimize(256, 4096)));
    });
    group.bench_function("ElementwiseWorkSize_1M", |b| {
        b.iter(|| black_box(ElementwiseWorkSize::optimize(1_000_000)));
    });

    group.finish();
}

// ── 8. Buffer utility benchmarks ─────────────────────────────────

fn bench_buffer_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer");

    // align_size
    group.bench_function("align_size_64", |b| {
        b.iter(|| black_box(align_size(black_box(1000), 64)));
    });
    group.bench_function("align_size_4096", |b| {
        b.iter(|| black_box(align_size(black_box(1000), 4096)));
    });

    // optimal_buffer_size
    for &elements in &[1024, 65536, 1_048_576] {
        group.bench_with_input(
            BenchmarkId::new("optimal_buffer_size", elements),
            &elements,
            |b, &n| {
                b.iter(|| black_box(optimal_buffer_size(n, 4)));
            },
        );
    }

    // validate_alignment
    group.bench_function("validate_alignment", |b| {
        b.iter(|| black_box(validate_alignment(black_box(4096), 64)));
    });

    // AlignedBuffer allocation
    for &size in &[1024, 65536, 1_048_576] {
        group.bench_with_input(
            BenchmarkId::new("aligned_buffer_new", size),
            &size,
            |b, &n| {
                b.iter(|| {
                    let buf: AlignedBuffer<f32> = AlignedBuffer::new(n, 64);
                    black_box(buf.len());
                });
            },
        );
    }

    // BufferPool allocate + return
    let pool = BufferPool::new();
    group.bench_function("pool_alloc_return_4K", |b| {
        b.iter(|| {
            let buf = pool.allocate(4096, 64);
            pool.return_buffer(buf);
        });
    });

    group.finish();
}

// ── 9. Transformer building-block benchmarks ─────────────────────

fn bench_residual(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer/residual");

    for &size in &[256, 1024, 4096] {
        let x = make_f32(size);
        let sublayer = make_f32(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("add", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        ResidualConnection::forward(black_box(&x), black_box(&sublayer))
                            .unwrap(),
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_transformer_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer/layer");
    // Use a smaller config to keep benchmark runtime reasonable.
    let config = TransformerLayerConfig {
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 4,
        head_dim: 16,
        intermediate_size: 176,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 256,
        use_gated_ffn: true,
    };
    let weights = LayerWeights::ones(&config);
    let x = make_f32_positive(config.hidden_size);

    for &position in &[0, 16, 64] {
        let label = format!("pos_{position}");
        group.bench_with_input(
            BenchmarkId::new("forward_ref", &label),
            &position,
            |b, &pos| {
                b.iter(|| {
                    let mut kv = KvCacheState::new(&config);
                    // Pre-fill some cache entries for realistic benchmarking
                    for p in 0..pos {
                        let dummy = make_f32_positive(config.hidden_size);
                        let dummy_weights = LayerWeights::ones(&config);
                        let _ = transformer_layer_forward_ref(
                            &dummy, &dummy_weights, &mut kv, p, &config,
                        );
                    }
                    black_box(
                        transformer_layer_forward_ref(
                            black_box(&x),
                            &weights,
                            &mut kv,
                            pos,
                            &config,
                        )
                        .unwrap(),
                    );
                });
            },
        );
    }
    group.finish();
}

// ── 10. Pipeline benchmarks ──────────────────────────────────────

fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline");
    let config = PipelineConfig {
        num_layers: 2,
        hidden_dim: 64,
        num_heads: 4,
        head_dim: 16,
        intermediate_dim: 176,
        vocab_size: 1024,
        max_seq_len: 256,
        use_gpu: false,
        fallback_to_cpu: true,
    };
    let mut pipeline = InferencePipeline::new(config).unwrap();

    group.bench_function("single_token_cpu", |b| {
        b.iter(|| {
            black_box(
                pipeline
                    .execute_single_token_cpu(black_box(&[42_u32]), black_box(0))
                    .unwrap(),
            );
        });
    });
    group.finish();
}

// ── Criterion groups ─────────────────────────────────────────────

criterion_group!(
    attention_benches,
    bench_scaled_dot_product,
    bench_multi_head_attention,
    bench_gqa,
    bench_attention_scores,
);

criterion_group!(ffn_benches, bench_ffn, bench_gated_ffn);

criterion_group!(norm_benches, bench_embedding_norm);

criterion_group!(
    embedding_benches,
    bench_embedding_lookup,
    bench_output_projection,
    bench_position_embedding,
);

criterion_group!(softmax_benches, bench_softmax);

criterion_group!(activation_benches, bench_activations);

criterion_group!(work_size_benches, bench_work_size);

criterion_group!(buffer_benches, bench_buffer_ops);

criterion_group!(
    transformer_benches,
    bench_residual,
    bench_transformer_layer,
    bench_pipeline,
);

criterion_main!(
    attention_benches,
    ffn_benches,
    norm_benches,
    embedding_benches,
    softmax_benches,
    activation_benches,
    work_size_benches,
    buffer_benches,
    transformer_benches,
);
