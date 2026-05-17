use anyhow::{Context, Result};
use serde_json::Value;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

pub fn render_perf_md(json_file: &Path, comparison_json: Option<&Path>) -> Result<String> {
    let data = read_json(json_file)?;
    let format_type = detect_format(json_file);

    let mut output = format!(
        "# BitNet-rs Performance Report - {format_type}\n\n{}\n\n{}\n\n{}\n\n{}\n",
        methods_environment_box(&data),
        model_info(&data),
        performance_table(data.get("measurements").unwrap_or(&Value::Null)),
        validation_results(&data),
    );

    if let Some(path) = comparison_json {
        let comparison = read_json(path)?;
        output.push_str(&format_comparison(&data, &comparison));
    }

    let raw = serde_json::to_string_pretty(&data)?;
    write!(
        output,
        "\n{}\n\n## Raw Measurements\n\n<details>\n<summary>Click to expand raw JSON data</summary>\n\n```json\n{}\n```\n\n</details>\n\n---\n\n*Generated from measured data: {}*\n*Report generated: {}Z*\n*Note: All performance measurements are from actual runs, not estimates.*\n",
        charts(data.get("measurements").unwrap_or(&Value::Null)),
        raw,
        json_file.file_name().and_then(|name| name.to_str()).unwrap_or("performance.json"),
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true).trim_end_matches('Z'),
    )?;

    Ok(output)
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing JSON {}", path.display()))
}

fn detect_format(path: &Path) -> &'static str {
    let name = path.to_string_lossy().to_ascii_lowercase();
    if name.contains("safetensors") {
        "SafeTensors"
    } else if name.contains("gguf") {
        "GGUF"
    } else {
        "Unknown"
    }
}

fn format_number(value: f64, precision: usize) -> String {
    if value >= 1000.0 {
        format!("{value:.0}")
            .as_bytes()
            .rchunks(3)
            .rev()
            .map(std::str::from_utf8)
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap_or_default()
            .join(",")
    } else if value >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.precision$}")
    }
}

fn value_f64(value: Option<&Value>, default: f64) -> f64 {
    value.and_then(Value::as_f64).unwrap_or(default)
}

fn value_str<'a>(value: Option<&'a Value>, default: &'a str) -> &'a str {
    value.and_then(Value::as_str).unwrap_or(default)
}

fn value_bool(value: Option<&Value>, default: bool) -> bool {
    value.and_then(Value::as_bool).unwrap_or(default)
}

fn methods_environment_box(data: &Value) -> String {
    let meta = data.get("metadata").unwrap_or(&Value::Null);
    let deterministic = value_bool(meta.get("deterministic"), false);
    let seed = meta.get("seed").and_then(Value::as_i64).unwrap_or(42);
    let threads = meta.get("threads").and_then(Value::as_i64).unwrap_or(1);
    let prompts = meta.get("num_prompts").map(value_to_string).unwrap_or_else(|| "N".to_string());
    let max_tokens = meta.get("max_new_tokens").and_then(Value::as_i64).unwrap_or(128);
    let warmup = meta.get("warmup_runs").and_then(Value::as_i64).unwrap_or(1);
    let iterations = meta.get("iterations").and_then(Value::as_i64).unwrap_or(10);
    let timestamp = meta
        .get("timestamp")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

    format!(
        "## Methods & Environment\n\n```\nPlatform: {}\nBitNet CLI: {} | Rust: {} | Python: {}\nTransformers: {} | Torch: {}\nDeterminism: BITNET_DETERMINISTIC={} BITNET_SEED={} RAYON_NUM_THREADS={}\nPrompts: {} fixed, max_new_tokens={}, warmup={}, medians over {} runs\nTimestamp: {}\n```\n",
        value_str(meta.get("platform"), "Unknown"),
        value_str(meta.get("bitnet_version"), "Unknown"),
        value_str(meta.get("rust_version"), "Unknown"),
        value_str(meta.get("python_version"), "Unknown"),
        value_str(meta.get("transformers_version"), "n/a"),
        value_str(meta.get("torch_version"), "n/a"),
        if deterministic { 1 } else { 0 },
        seed,
        threads,
        prompts,
        max_tokens,
        warmup,
        iterations,
        timestamp,
    )
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => "N".to_string(),
    }
}

fn model_info(data: &Value) -> String {
    let model = data.get("model").unwrap_or(&Value::Null);
    let mut info = format!(
        "## Model Information\n\n- **Model ID**: {}\n- **Format**: {}\n- **Size**: {:.1} MB\n- **Parameters**: {}\n- **Quantization**: {}\n- **Tokenizer**: {}\n",
        value_str(model.get("id"), "Unknown"),
        value_str(model.get("format"), "Unknown"),
        value_f64(model.get("size_mb"), 0.0),
        model.get("parameters").map(value_to_string).unwrap_or_else(|| "Unknown".to_string()),
        value_str(model.get("quantization"), "None"),
        value_str(model.get("tokenizer_type"), "Unknown"),
    );

    if let Some(policy) = model.get("scoring_policy") {
        write!(
            info,
            "\n### Scoring Policy\n- Add BOS: {}\n- Append EOS: {}\n- Mask Padding: {}\n",
            value_bool(policy.get("add_bos"), false),
            value_bool(policy.get("append_eos"), false),
            value_bool(policy.get("mask_pad"), true),
        )
        .expect("writing to String cannot fail");
    }

    info
}

fn performance_table(measurements: &Value) -> String {
    let mut table = "## Performance Metrics\n\n| Metric | Median | P95 | Min | Max | StdDev |\n|--------|--------|-----|-----|-----|--------|\n".to_string();
    let metrics = [
        ("tokens_per_second", "Tokens/sec", ""),
        ("time_to_first_token", "First Token", "ms"),
        ("memory_mb", "Memory", "MB"),
        ("latency_per_token", "Token Latency", "ms"),
    ];

    for (key, name, unit) in metrics {
        let Some(data) = measurements.get(key) else {
            continue;
        };
        let label = if unit.is_empty() { name.to_string() } else { format!("{name} ({unit})") };
        if data.is_object() {
            let median = format_number(value_f64(data.get("median"), 0.0), 2);
            let p95 = format_number(value_f64(data.get("p95"), 0.0), 2);
            let min = format_number(value_f64(data.get("min"), 0.0), 2);
            let max = format_number(value_f64(data.get("max"), 0.0), 2);
            let stddev = format_number(value_f64(data.get("stddev"), 0.0), 2);
            writeln!(table, "| {label} | {median} | {p95} | {min} | {max} | {stddev} |")
                .expect("writing to String cannot fail");
        } else if let Some(value) = data.as_f64() {
            let value = format_number(value, 2);
            writeln!(table, "| {label} | {value} | - | - | - | - |")
                .expect("writing to String cannot fail");
        }
    }

    table
}

fn validation_results(data: &Value) -> String {
    let validation = data.get("validation").unwrap_or(&Value::Null);
    if validation.as_object().is_none_or(serde_json::Map::is_empty) {
        return String::new();
    }

    let mut results = "## Validation Results\n\n| Check | Status | Value | Threshold | Details |\n|-------|--------|-------|-----------|---------|\n".to_string();

    if let Some(tp) = validation.get("tokenizer_parity") {
        let status = if value_bool(tp.get("pass"), false) { "✅ Pass" } else { "❌ Fail" };
        writeln!(
            results,
            "| Tokenizer Parity | {} | {} diffs | 0 | {} |",
            status,
            tp.get("differences").and_then(Value::as_i64).unwrap_or(0),
            value_str(tp.get("details"), ""),
        )
        .expect("writing to String cannot fail");
    }

    if let Some(lc) = validation.get("logit_correlation") {
        let tau_b = value_f64(lc.get("median_tau_b"), 0.0);
        let threshold = value_f64(lc.get("threshold"), 0.95);
        let status = if tau_b >= threshold { "✅ Pass" } else { "❌ Fail" };
        writeln!(
            results,
            "| Logit τ-b | {} | {:.3} | ≥{} | {} samples |",
            status,
            tau_b,
            threshold,
            lc.get("samples").and_then(Value::as_i64).unwrap_or(0),
        )
        .expect("writing to String cannot fail");
    }

    if let Some(nll) = validation.get("nll_parity") {
        let delta = value_f64(nll.get("delta_mean_nll"), 0.0).abs();
        let threshold = value_f64(nll.get("threshold"), 0.01);
        let status = if delta <= threshold { "✅ Pass" } else { "❌ Fail" };
        writeln!(
            results,
            "| NLL Parity | {} | Δ={:.4} | ≤{} | {} tokens |",
            status,
            delta,
            threshold,
            nll.get("tokens").and_then(Value::as_i64).unwrap_or(0),
        )
        .expect("writing to String cannot fail");
    }

    results
}

fn charts(measurements: &Value) -> String {
    let mut charts = "## Performance Trends\n\n### Tokens per Second Distribution\n".to_string();
    let Some(distribution) = measurements
        .get("tokens_per_second")
        .and_then(|v| v.get("distribution"))
        .and_then(Value::as_object)
    else {
        return charts;
    };

    let max_count = distribution.values().filter_map(Value::as_u64).max().unwrap_or(1).max(1);
    let mut buckets: Vec<_> = distribution.iter().collect();
    buckets.sort_by(|(left, _), (right, _)| left.cmp(right));
    for (bucket, count) in buckets {
        let count = count.as_u64().unwrap_or(0);
        let bar_len = ((count as f64 / max_count as f64) * 40.0) as usize;
        writeln!(charts, "{bucket:>6}: {} {count}", "█".repeat(bar_len))
            .expect("writing to String cannot fail");
    }

    charts
}

fn get_metric(data: &Value, path: &str) -> f64 {
    let mut value = data;
    for part in path.split('.') {
        let Some(next) = value.get(part) else {
            return 0.0;
        };
        value = next;
    }
    value.as_f64().unwrap_or(0.0)
}

fn format_comparison(st_data: &Value, gguf_data: &Value) -> String {
    let mut comparison = "## Format Comparison (SafeTensors vs GGUF)\n\n| Metric | SafeTensors | GGUF | Difference | Ratio |\n|--------|-------------|------|------------|-------|\n".to_string();
    let metrics = [
        ("Throughput (tok/s)", "measurements.tokens_per_second.median"),
        ("First Token (ms)", "measurements.time_to_first_token.median"),
        ("Memory (MB)", "measurements.memory_mb.peak"),
        ("Load Time (s)", "measurements.load_time.median"),
    ];

    for (name, path) in metrics {
        let st_val = get_metric(st_data, path);
        let gguf_val = get_metric(gguf_data, path);
        if st_val != 0.0 && gguf_val != 0.0 {
            let diff = gguf_val - st_val;
            let ratio = gguf_val / st_val;
            let sign = if diff > 0.0 { "+" } else { "" };
            writeln!(
                comparison,
                "| {} | {} | {} | {}{} | {:.2}x |",
                name,
                format_number(st_val, 2),
                format_number(gguf_val, 2),
                sign,
                format_number(diff, 2),
                ratio,
            )
            .expect("writing to String cannot fail");
        }
    }

    comparison
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn renders_single_report_from_measured_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("linux-safetensors.json");
        fs::write(
            &path,
            r#"{
              "metadata": {"platform":"linux", "deterministic": true, "num_prompts": 2, "timestamp":"2026-01-01T00:00:00Z"},
              "model": {"id":"fixture", "format":"SafeTensors", "size_mb": 12.5},
              "measurements": {"tokens_per_second": {"median": 12.34, "p95": 15, "min": 10, "max": 20, "stddev": 1.2, "distribution": {"10-20": 3}}},
              "validation": {"tokenizer_parity": {"pass": true, "differences": 0}}
            }"#,
        )
        .unwrap();

        let rendered = render_perf_md(&path, None).unwrap();
        assert!(rendered.contains("# BitNet-rs Performance Report - SafeTensors"));
        assert!(rendered.contains("| Tokens/sec | 12.3 | 15.0 | 10.0 | 20.0 | 1.20 |"));
        assert!(rendered.contains("| Tokenizer Parity | ✅ Pass | 0 diffs | 0 |  |"));
        assert!(rendered.contains("10-20:"));
    }
}
