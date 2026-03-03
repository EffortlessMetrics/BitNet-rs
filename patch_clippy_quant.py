with open("crates/bitnet-quantization/tests/quantization_extended_tests.rs", "r") as f:
    content = f.read()

content = content.replace(
"""        if let Ok(qd) = q.quantize_tensor(&tensor) {
            if let Ok(deq) = q.dequantize_tensor(&qd) {
                let vals = deq.to_vec().unwrap();
                for &v in &vals {
                    assert!(v.is_finite());
                }
            }
        }""",
"""        #[allow(clippy::collapsible_if)]
        if let Ok(qd) = q.quantize_tensor(&tensor) {
            if let Ok(deq) = q.dequantize_tensor(&qd) {
                let vals = deq.to_vec().unwrap();
                for &v in &vals {
                    assert!(v.is_finite());
                }
            }
        }""")

with open("crates/bitnet-quantization/tests/quantization_extended_tests.rs", "w") as f:
    f.write(content)
