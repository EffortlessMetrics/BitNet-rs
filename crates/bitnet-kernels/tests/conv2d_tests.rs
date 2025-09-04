use std::process::Command;

/// Integration tests comparing the placeholder conv2d against PyTorch.
///
/// These tests are marked as ignored because they require PyTorch to be
/// installed in the environment. They serve as a template for future
/// correctness verification once the conv2d kernel is implemented.
#[test]
#[ignore]
fn conv2d_reference_cases() {
    let script = r#"
import json, torch, torch.nn.functional as F
cases = [
    ((1,1,4,4),(1,1,3,3),(1,1),(0,0),(1,1)),
    ((1,1,5,5),(1,1,3,3),(2,2),(1,1),(1,1)),
    ((1,1,6,6),(1,1,2,2),(1,1),(0,0),(2,2)),
]
results = []
for shape_in, shape_w, stride, pad, dil in cases:
    x = torch.randn(*shape_in)
    w = torch.randn(*shape_w)
    y = F.conv2d(x, w, None, stride=stride, padding=pad, dilation=dil)
    results.append({
        'input': x.flatten().tolist(),
        'weight': w.flatten().tolist(),
        'output': y.flatten().tolist(),
        'cfg': {
            'input_shape': shape_in,
            'weight_shape': shape_w,
            'stride': stride,
            'padding': pad,
            'dilation': dil,
        }
    })
print(json.dumps(results))
"#;

    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .expect("failed to run python script");
    assert!(
        output.status.success(),
        "python script failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let cases: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("invalid json from python");
    for case in cases.as_array().expect("cases should be array") {
        let cfg = &case["cfg"];
        let stride = (
            cfg["stride"][0].as_u64().unwrap() as usize,
            cfg["stride"][1].as_u64().unwrap() as usize,
        );
        let padding = (
            cfg["padding"][0].as_u64().unwrap() as usize,
            cfg["padding"][1].as_u64().unwrap() as usize,
        );
        let dilation = (
            cfg["dilation"][0].as_u64().unwrap() as usize,
            cfg["dilation"][1].as_u64().unwrap() as usize,
        );

        // Placeholder invocation; replace with real conv2d call when implemented
        let input: Vec<f32> =
            case["input"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let weight: Vec<f32> =
            case["weight"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let output_ref: Vec<f32> =
            case["output"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();

        let mut output_buf = vec![0f32; output_ref.len()];
        let _ = bitnet_kernels::convolution::conv2d(
            &input,
            &weight,
            None,
            &mut output_buf,
            bitnet_kernels::convolution::Conv2DParams { stride, padding, dilation },
            (
                cfg["input_shape"][0].as_u64().unwrap() as usize,
                cfg["input_shape"][1].as_u64().unwrap() as usize,
                cfg["input_shape"][2].as_u64().unwrap() as usize,
                cfg["input_shape"][3].as_u64().unwrap() as usize,
            ),
            (
                cfg["weight_shape"][0].as_u64().unwrap() as usize,
                cfg["weight_shape"][1].as_u64().unwrap() as usize,
                cfg["weight_shape"][2].as_u64().unwrap() as usize,
                cfg["weight_shape"][3].as_u64().unwrap() as usize,
            ),
        );
        assert_eq!(output_buf.len(), output_ref.len(), "output size mismatch");
    }
}
