//! Criterion benchmarks for prompt template operations.
//!
//! Measures throughput of template detection, application, and multi-turn
//! chat rendering across representative template families.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use bitnet_prompt_templates::{ChatRole, ChatTurn, TemplateType};

fn bench_detect(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_detect");

    group.bench_function("chatml_jinja", |b| {
        b.iter(|| {
            TemplateType::detect(
                black_box(Some("phi-4")),
                black_box(Some("<|im_start|>system\n{system}<|im_end|>")),
            )
        })
    });

    group.bench_function("llama3_jinja", |b| {
        b.iter(|| {
            TemplateType::detect(
                black_box(Some("llama-3")),
                black_box(Some("<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>")),
            )
        })
    });

    group.bench_function("name_only_fallback", |b| {
        b.iter(|| TemplateType::detect(black_box(Some("gemma")), black_box(None)))
    });

    group.bench_function("no_hints", |b| {
        b.iter(|| TemplateType::detect(black_box(None), black_box(None)))
    });

    group.finish();
}

fn bench_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_apply");
    let user_text = "Explain quantum computing in simple terms.";
    let system = Some("You are a helpful assistant.");

    for template in &[
        TemplateType::Llama3Chat,
        TemplateType::Phi4Chat,
        TemplateType::QwenChat,
        TemplateType::GemmaChat,
        TemplateType::MistralChat,
        TemplateType::AlpacaInstruct,
        TemplateType::VicunaChat,
    ] {
        group.bench_function(format!("{template}"), |b| {
            b.iter(|| template.apply(black_box(user_text), black_box(system)))
        });
    }

    group.finish();
}

fn bench_render_chat(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_render_chat");

    let history_3 = vec![
        ChatTurn::new(ChatRole::User, "Hello"),
        ChatTurn::new(ChatRole::Assistant, "Hi! How can I help?"),
        ChatTurn::new(ChatRole::User, "Tell me about Rust."),
    ];

    let history_10: Vec<ChatTurn> = (0..10)
        .flat_map(|i| {
            vec![
                ChatTurn::new(ChatRole::User, format!("Question {i}")),
                ChatTurn::new(ChatRole::Assistant, format!("Answer {i} with some detail.")),
            ]
        })
        .collect();

    let system = Some("You are a helpful assistant.");

    for template in &[TemplateType::Llama3Chat, TemplateType::Phi4Chat, TemplateType::MistralChat] {
        group.bench_function(format!("{template}/3_turns"), |b| {
            b.iter(|| template.render_chat(black_box(&history_3), black_box(system)))
        });

        group.bench_function(format!("{template}/20_turns"), |b| {
            b.iter(|| template.render_chat(black_box(&history_10), black_box(system)))
        });
    }

    group.finish();
}

fn bench_suggest_for_arch(c: &mut Criterion) {
    let mut group = c.benchmark_group("suggest_for_arch");

    group.bench_function("known_arch", |b| {
        b.iter(|| TemplateType::suggest_for_arch(black_box("llama-3.2")))
    });

    group.bench_function("unknown_arch", |b| {
        b.iter(|| TemplateType::suggest_for_arch(black_box("unknown-model")))
    });

    group.bench_function("case_insensitive", |b| {
        b.iter(|| TemplateType::suggest_for_arch(black_box("Phi-4")))
    });

    group.finish();
}

fn bench_all_variants(c: &mut Criterion) {
    c.bench_function("all_variants_count", |b| {
        b.iter(|| {
            let variants = TemplateType::all_variants();
            black_box(variants.len())
        })
    });
}

criterion_group!(
    benches,
    bench_detect,
    bench_apply,
    bench_render_chat,
    bench_suggest_for_arch,
    bench_all_variants,
);
criterion_main!(benches);
