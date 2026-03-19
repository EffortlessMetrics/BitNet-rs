use criterion::{black_box, criterion_group, criterion_main, Criterion};
use bitnet_logits_filters::apply_typical;

fn bench_typical(c: &mut Criterion) {
    let mut group = c.benchmark_group("typical");

    // Generate some random probabilities
    let mut probs = vec![0.0; 32000];
    for i in 0..1000 {
        probs[i * 32] = 1.0 / 1000.0;
    }

    group.bench_function("apply_typical_sparse", |b| {
        b.iter(|| {
            let mut p = probs.clone();
            apply_typical(black_box(&mut p), 0.5);
            black_box(p);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_typical);
criterion_main!(benches);
