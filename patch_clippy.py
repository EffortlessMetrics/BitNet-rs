with open("crates/bitnet-logits/tests/logits_integration.rs", "r") as f:
    content = f.read()

content = content.replace("let mut logits = input.clone();\n\n    // Test apply_top_k", "let mut logits = input;\n\n    // Test apply_top_k")

with open("crates/bitnet-logits/tests/logits_integration.rs", "w") as f:
    f.write(content)
