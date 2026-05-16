//! Answer-quality receipt helpers.
//!
//! This module contains text sanitation and lightweight garbage filters used by
//! strict ask receipts.  It is intentionally independent from generation so the
//! receipt quality policy can evolve without increasing `main.rs` complexity.

pub(crate) fn answer_quality_receipt(
    answer: &str,
    run_receipt: &serde_json::Value,
    max_new_tokens: usize,
) -> serde_json::Value {
    let trimmed = strip_answer_special_markers(answer).trim().to_string();
    let non_empty_answer = !trimmed.is_empty();
    let printable_utf8 = trimmed.chars().all(|ch| ch == '\n' || ch == '\t' || !ch.is_control());
    let no_replacement_chars = !trimmed.contains('\u{FFFD}');
    let no_raw_special_tokens = !trimmed.contains("<|") && !trimmed.contains("|>");
    let mostly_text = answer_mostly_text(&trimmed);
    let language_signal = answer_has_language_signal(&trimmed);
    let suspicious_fragment_count = suspicious_answer_fragment_count(&trimmed);
    let fragment_filter_passed = suspicious_fragment_count <= 1;
    let garbage_filter_passed = non_empty_answer
        && printable_utf8
        && no_replacement_chars
        && no_raw_special_tokens
        && mostly_text
        && language_signal
        && fragment_filter_passed;
    let generated = run_receipt["tokens"]["generated"].as_u64().unwrap_or_default() as usize;
    serde_json::json!({
        "printable_utf8": printable_utf8,
        "non_empty_answer": non_empty_answer,
        "stop_reason": if generated >= max_new_tokens { "max_tokens" } else { "eos_or_stop_sequence" },
        "garbage_filter_passed": garbage_filter_passed,
        "no_replacement_chars": no_replacement_chars,
        "no_raw_special_tokens": no_raw_special_tokens,
        "mostly_text": mostly_text,
        "language_signal": language_signal,
        "suspicious_fragment_count": suspicious_fragment_count,
        "fragment_filter_passed": fragment_filter_passed,
    })
}

pub(crate) fn strip_answer_special_markers(answer: &str) -> String {
    answer.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "").replace("<|eot_id|>", "")
}

pub(crate) fn answer_mostly_text(answer: &str) -> bool {
    let mut meaningful = 0usize;
    let mut punctuation_or_control = 0usize;
    for ch in answer.chars() {
        if ch.is_alphanumeric() || ch.is_whitespace() {
            meaningful += 1;
        } else if ch.is_ascii_punctuation() || ch.is_control() {
            punctuation_or_control += 1;
        }
    }
    meaningful > 0 && punctuation_or_control <= meaningful.saturating_mul(2)
}

fn answer_has_language_signal(answer: &str) -> bool {
    let compact: String = answer.chars().filter(|ch| !ch.is_whitespace()).collect();
    let numeric_short_answer = compact.len() <= 8
        && compact.chars().any(|ch| ch.is_ascii_digit())
        && compact.chars().all(|ch| ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+'));
    if numeric_short_answer {
        return true;
    }

    answer_word_tokens(answer).any(|word| ANSWER_QUALITY_LANGUAGE_WORDS.contains(&word.as_str()))
}

fn suspicious_answer_fragment_count(answer: &str) -> usize {
    answer
        .split_whitespace()
        .filter(|token| {
            let alphabetic = token.chars().filter(|ch| ch.is_alphabetic()).count();
            if alphabetic == 0 {
                return false;
            }
            let apostrophes = token.matches('\'').count();
            let ascii_punctuation = token.chars().filter(|ch| ch.is_ascii_punctuation()).count();
            let internal_period = token.contains('.')
                && !token.ends_with('.')
                && token.chars().any(|ch| ch.is_alphabetic());
            (apostrophes > 1) || internal_period || (alphabetic >= 3 && ascii_punctuation >= 3)
        })
        .count()
}

fn answer_word_tokens(answer: &str) -> impl Iterator<Item = String> + '_ {
    answer
        .split(|ch: char| !ch.is_alphabetic())
        .filter(|word| word.len() >= 2)
        .map(str::to_ascii_lowercase)
}

const ANSWER_QUALITY_LANGUAGE_WORDS: &[&str] = &[
    "a",
    "about",
    "add",
    "adds",
    "an",
    "and",
    "answer",
    "are",
    "architecture",
    "blue",
    "bit",
    "bitnet",
    "black",
    "capital",
    "color",
    "colors",
    "common",
    "compute",
    "data",
    "efficient",
    "explain",
    "for",
    "four",
    "france",
    "function",
    "green",
    "is",
    "language",
    "low",
    "memory",
    "model",
    "number",
    "numbers",
    "of",
    "one",
    "paris",
    "python",
    "red",
    "reduce",
    "sentence",
    "shape",
    "shapes",
    "the",
    "that",
    "three",
    "to",
    "uses",
    "weight",
    "weights",
    "white",
    "with",
    "wet",
    "water",
    "yellow",
    "yes",
    "no",
];
