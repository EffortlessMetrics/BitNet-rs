//! Criterion benchmarks for tokenizer operations:
//! encode/decode at various input lengths, special token lookup, vocabulary
//! lookup, batch encoding, and tokenizer construction.

use bitnet_tokenizers::vocabulary::{VocabConfig, Vocabulary};
use bitnet_tokenizers::{BasicTokenizer, MockTokenizer, Tokenizer};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::collections::HashMap;
use std::hint::black_box;

// ── Helpers ──────────────────────────────────────────────────────────

/// Generate a string of approximately `n_words` words.
fn make_text(n_words: usize) -> String {
    const WORDS: &[&str] = &[
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "cat", "while",
        "running", "through", "forest", "under", "bright", "blue", "sky", "with", "warm",
    ];
    (0..n_words).map(|i| WORDS[i % WORDS.len()]).collect::<Vec<_>>().join(" ")
}

/// Generate a vector of `n` token IDs in the byte-level range.
fn make_token_ids(n: usize) -> Vec<u32> {
    (0..n).map(|i| (i % 128) as u32 + 32).collect() // printable ASCII range
}

/// Build a `Vocabulary` with `n` entries and special tokens configured.
fn make_vocabulary(n: usize) -> Vocabulary {
    let mut map: HashMap<String, u32> = (0..n as u32).map(|i| (format!("tok_{i}"), i)).collect();
    map.insert("<bos>".into(), n as u32);
    map.insert("<eos>".into(), n as u32 + 1);
    map.insert("<pad>".into(), n as u32 + 2);

    let config = VocabConfig {
        bos_token: Some("<bos>".into()),
        eos_token: Some("<eos>".into()),
        pad_token: Some("<pad>".into()),
        ..VocabConfig::default()
    };
    Vocabulary::new(map, config)
}

// ── Encode benchmarks ────────────────────────────────────────────────

fn bench_encode(c: &mut Criterion) {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    let mut group = c.benchmark_group("encode");

    for (label, n_words) in [("short_10w", 10), ("medium_100w", 100), ("long_1000w", 1000)] {
        let text = make_text(n_words);
        group.bench_with_input(BenchmarkId::new("basic", label), &text, |b, text| {
            b.iter(|| black_box(tok.encode(black_box(text), true, true).unwrap()));
        });
    }

    // Also benchmark MockTokenizer for comparison.
    let mock = MockTokenizer::new();
    for (label, n_words) in [("short_10w", 10), ("medium_100w", 100), ("long_1000w", 1000)] {
        let text = make_text(n_words);
        group.bench_with_input(BenchmarkId::new("mock", label), &text, |b, text| {
            b.iter(|| black_box(mock.encode(black_box(text), false, false).unwrap()));
        });
    }

    group.finish();
}

// ── Decode benchmarks ────────────────────────────────────────────────

fn bench_decode(c: &mut Criterion) {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    let mut group = c.benchmark_group("decode");

    for (label, n_tokens) in [("short_10", 10), ("medium_100", 100), ("long_1000", 1000)] {
        let ids = make_token_ids(n_tokens);
        group.bench_with_input(BenchmarkId::new("basic", label), &ids, |b, ids| {
            b.iter(|| black_box(tok.decode(black_box(ids)).unwrap()));
        });
    }

    let mock = MockTokenizer::new();
    for (label, n_tokens) in [("short_10", 10), ("medium_100", 100), ("long_1000", 1000)] {
        let ids = make_token_ids(n_tokens);
        group.bench_with_input(BenchmarkId::new("mock", label), &ids, |b, ids| {
            b.iter(|| black_box(mock.decode(black_box(ids)).unwrap()));
        });
    }

    group.finish();
}

// ── Special token lookup ─────────────────────────────────────────────

fn bench_special_tokens_lookup(c: &mut Criterion) {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    c.bench_function("special_tokens_lookup", |b| {
        b.iter(|| {
            black_box(tok.bos_token_id());
            black_box(tok.eos_token_id());
            black_box(tok.pad_token_id());
            black_box(tok.is_special_token(1));
            black_box(tok.is_special_token(2));
            black_box(tok.is_special_token(3));
            black_box(tok.is_special_token(42));
        });
    });
}

// ── Vocabulary lookup ────────────────────────────────────────────────

fn bench_vocab_lookup(c: &mut Criterion) {
    let vocab = make_vocabulary(10_000);
    let mut group = c.benchmark_group("vocab_lookup");

    group.bench_function("token_to_id", |b| {
        b.iter(|| {
            for i in (0..100).map(|i| i * 100) {
                black_box(vocab.token_to_id(black_box(&format!("tok_{i}"))));
            }
        });
    });

    group.bench_function("id_to_token", |b| {
        b.iter(|| {
            for i in (0..100).map(|i| i * 100) {
                black_box(vocab.id_to_token(black_box(i as u32)));
            }
        });
    });

    group.bench_function("is_special_token", |b| {
        b.iter(|| {
            for i in 0..100u32 {
                black_box(vocab.is_special_token(black_box(i)));
            }
        });
    });

    group.finish();
}

// ── Batch encode ─────────────────────────────────────────────────────

fn bench_batch_encode(c: &mut Criterion) {
    let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
    let texts: Vec<String> = (0..10).map(|i| make_text(20 + i * 10)).collect();

    c.bench_function("batch_encode_10_strings", |b| {
        b.iter(|| {
            let results: Vec<_> =
                texts.iter().map(|t| tok.encode(black_box(t), true, true).unwrap()).collect();
            black_box(results)
        });
    });
}

// ── Tokenizer creation ──────────────────────────────────────────────

fn bench_tokenizer_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer_creation");

    group.bench_function("basic_default", |b| {
        b.iter(|| black_box(BasicTokenizer::new()));
    });

    group.bench_function("basic_with_config", |b| {
        b.iter(|| {
            black_box(BasicTokenizer::with_config(
                black_box(50257),
                black_box(Some(1)),
                black_box(Some(2)),
                black_box(Some(3)),
            ))
        });
    });

    group.bench_function("mock_default", |b| {
        b.iter(|| black_box(MockTokenizer::new()));
    });

    group.bench_function("vocabulary_10k", |b| {
        b.iter_batched(
            || {
                let map: HashMap<String, u32> =
                    (0..10_000u32).map(|i| (format!("tok_{i}"), i)).collect();
                (map, VocabConfig::default())
            },
            |(map, config)| black_box(Vocabulary::new(map, config)),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ── Vocabulary construction from JSON ────────────────────────────────

fn bench_vocab_from_json(c: &mut Criterion) {
    // Build a JSON string with 1000-entry vocab to benchmark parsing.
    let entries: Vec<String> = (0..1000u32).map(|i| format!("\"tok_{i}\": {i}")).collect();
    let json = format!(r#"{{"model": {{"vocab": {{{}}}}}}}"#, entries.join(", "));

    c.bench_function("vocab_from_json_1k", |b| {
        b.iter(|| black_box(Vocabulary::from_json(black_box(&json)).unwrap()));
    });
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_special_tokens_lookup,
    bench_vocab_lookup,
    bench_batch_encode,
    bench_tokenizer_creation,
    bench_vocab_from_json,
);
criterion_main!(benches);
