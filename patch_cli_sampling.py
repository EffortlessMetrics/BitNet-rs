with open("crates/bitnet-cli-sampling-core/src/lib.rs", "r") as f:
    content = f.read()

content = content.replace(
    "let rng = if let Some(seed) = seed {\n            ChaCha20Rng::seed_from_u64(seed)\n        } else {\n            ChaCha20Rng::from_rng(&mut rand::rng())\n        };",
    "let rng = seed.map_or_else(|| ChaCha20Rng::from_rng(&mut rand::rng()), ChaCha20Rng::seed_from_u64);"
)
content = content.replace(
    "|| (self.temperature == 1.0 && self.top_k == 0 && self.top_p == 1.0)",
    "|| ((self.temperature - 1.0).abs() < f32::EPSILON && self.top_k == 0 && (self.top_p - 1.0).abs() < f32::EPSILON)"
)
content = content.replace(
    "if self.temperature != 1.0 {",
    "if (self.temperature - 1.0).abs() > f32::EPSILON {"
)
content = content.replace(
    "if self.repetition_penalty == 1.0 {",
    "if (self.repetition_penalty - 1.0).abs() < f32::EPSILON {"
)
content = content.replace(
    "let penalty = self.repetition_penalty.powi(count as i32);",
    "#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]\n                let penalty = self.repetition_penalty.powi(count as i32);"
)
content = content.replace(
    "return i as u32;",
    "#[allow(clippy::cast_possible_truncation)] return i as u32;"
)
content = content.replace(
    "(probs.len() - 1) as u32",
    "#[allow(clippy::cast_possible_truncation)] { (probs.len() - 1) as u32 }"
)
content = content.replace(
    "if val > best_val || (val == best_val && i < best_idx) {",
    "if val > best_val || ((val - best_val).abs() < f32::EPSILON && i < best_idx) {"
)
content = content.replace(
    "best_idx as u32",
    "#[allow(clippy::cast_possible_truncation)] { best_idx as u32 }"
)
content = content.replace(
    "let id = i as u32;",
    "#[allow(clippy::cast_possible_truncation)] let id = i as u32;"
)
content = content.replace(
    "if x > best.0 || (x == best.0 && id < best.1) {",
    "if x > best.0 || ((x - best.0).abs() < f32::EPSILON && id < best.1) {"
)

with open("crates/bitnet-cli-sampling-core/src/lib.rs", "w") as f:
    f.write(content)
