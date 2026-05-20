use super::{ModelCoverageEntry, ModelStatusDashboard};

pub(super) fn print_model_status_text(dashboard: &ModelStatusDashboard) {
    println!("CUDA model status for {}", dashboard.device);
    println!("requested backend: {}", dashboard.requested_backend);
    if let Some(selected_backend) = &dashboard.selected_backend {
        println!("selected backend: {selected_backend}");
    } else {
        println!("selected backend: none");
    }
    println!("source: {}", dashboard.source.display());
    println!("{}", dashboard.note);
    println!();

    print_model_status_group(dashboard, "Supported", "supported");
    println!();
    print_model_status_group(dashboard, "Candidates", "candidate");
}

pub(super) fn print_model_status_group(
    dashboard: &ModelStatusDashboard,
    title: &str,
    category: &str,
) {
    println!("{title}:");
    let mut printed = false;
    for row in dashboard.models.iter().filter(|row| row.category == category) {
        printed = true;
        println!("  {}", row.display_name);
        println!("    id: {}", row.id);
        println!("    class: {}", model_status_class_label(&row.model_class));
        println!("    route: {}", row.route.as_deref().unwrap_or("not ready"));
        println!("    tier: {}", row.tier);
        println!("    cpu answer: {}", ready_label(row.cpu_answer_ready));
        println!("    cuda answer: {}", ready_label(row.accelerator_answer_ready));
        println!("    ask: {}", row.ask);
        if matches!(row.route.as_deref(), Some("dense_regular_llm_cuda" | "bitnet_qk256_cuda")) {
            println!("    one-token: {}", row.one_token);
            println!("    short-decode: {}", row.short_decode);
        }
        println!("    warm-session: {}", row.warm_session);
        println!("    benchmark: {}", row.benchmark);
        println!("    speedup: {}", if row.speedup_claim { "qualified" } else { "not qualified" });
        println!("    server: {}", row.server);
        println!(
            "    full residency: {}",
            if row.full_residency_claim { "claimed" } else { "not claimed" }
        );
        println!("    claim boundary: {}", row.claim_boundary);
        if row.category == "candidate" {
            println!("    next proof: {}", row.next_proof);
        }
        println!();
    }

    if !printed {
        println!("  none");
    }
}

pub(super) fn model_status_display_name(entry: &ModelCoverageEntry) -> String {
    if let Some(id) = &entry.capability_id {
        return id.clone();
    }
    if let Some(id) = entry.verifier_surface.split_whitespace().last()
        && !id.is_empty()
        && id != "only"
        && id != "matrix"
    {
        return id.to_string();
    }
    entry.contract_id.clone().unwrap_or_else(|| entry.id.clone())
}

pub(super) fn model_status_class_label(model_class: &str) -> &'static str {
    match model_class {
        "bitnet" => "BitNet",
        "dense_slm" => "dense SLM",
        "small_llm" => "small dense LLM",
        "modern_llm_docs_only" => "docs-only modern LLM",
        _ => "model",
    }
}

pub(super) fn ready_label(ready: bool) -> &'static str {
    if ready { "ready" } else { "not ready" }
}

pub(super) fn ask_status(entry: &ModelCoverageEntry) -> String {
    if entry.claims.product_cli_ready && entry.claims.accelerator_answer_ready {
        "ready".to_string()
    } else {
        "not ready".to_string()
    }
}

pub(super) fn dense_receipt_status(entry: &ModelCoverageEntry, receipt_fragment: &str) -> String {
    if entry.required_receipts.iter().any(|receipt| receipt.contains(receipt_fragment)) {
        "ready".to_string()
    } else {
        "not ready".to_string()
    }
}

pub(super) fn warm_session_status(entry: &ModelCoverageEntry) -> String {
    if entry.required_receipts.iter().any(|receipt| receipt.contains("warm_session"))
        && entry.claims.accelerator_answer_ready
    {
        "ready".to_string()
    } else {
        "not ready".to_string()
    }
}

pub(super) fn benchmark_status(entry: &ModelCoverageEntry) -> String {
    if entry.claims.benchmark_qualified && entry.claims.speedup_claim {
        return "qualified".to_string();
    }
    if (entry.claims.product_cli_ready || entry.claims.accelerator_answer_ready)
        && entry.required_receipts.iter().any(|receipt| receipt.contains("benchmark"))
    {
        return "reviewed, speedup not accepted".to_string();
    }
    "not ready".to_string()
}
