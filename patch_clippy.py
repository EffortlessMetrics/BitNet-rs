with open("crates/bitnet-logits/tests/logits_integration.rs", "r") as f:
    content = f.read()

content = content.replace("let mut logits = input.to_vec();", "let mut logits = input.clone();")
content = content.replace("let input = vec![1.0f32, 2.0, 4.0];", "let input = [1.0f32, 2.0, 4.0].to_vec();")
content = content.replace("let logits: Vec<f32> = (0..n).map(|i| i as f32).collect();", "let logits: Vec<f32> = (0..n).map(|i| #[allow(clippy::cast_precision_loss)] (i as f32)).collect();")

with open("crates/bitnet-logits/tests/logits_integration.rs", "w") as f:
    f.write(content)

with open("crates/bitnet-logits/tests/logits_proptests.rs", "r") as f:
    content = f.read()

content = content.replace("let expected = 1.0 / n as f32;", "let expected = 1.0 / #[allow(clippy::cast_precision_loss)] (n as f32);")

with open("crates/bitnet-logits/tests/logits_proptests.rs", "w") as f:
    f.write(content)
