with open("crates/bitnet-logits/tests/logits_integration.rs", "r") as f:
    content = f.read()

content = content.replace("#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] (i as f32)", "i as f32")

with open("crates/bitnet-logits/tests/logits_integration.rs", "w") as f:
    f.write(content)

with open("crates/bitnet-logits/tests/logits_proptests.rs", "r") as f:
    content = f.read()

content = content.replace("1.0 / n as f64 as f32", "1.0 / n as f32")

with open("crates/bitnet-logits/tests/logits_proptests.rs", "w") as f:
    f.write(content)
