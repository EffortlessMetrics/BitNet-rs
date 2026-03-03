import re

files = [
    "crates/bitnet-kernels/src/opencl_mixed_precision.rs",
    "crates/bitnet-kernels/src/opencl_model_converter.rs",
]

for file_path in files:
    with open(file_path, "r") as f:
        content = f.read()

    # Replace 3.14 with std::f32::consts::PI
    fixed_content = content.replace("3.14", "std::f32::consts::PI")

    with open(file_path, "w") as f:
        f.write(fixed_content)

file_path = "crates/bitnet-kernels/src/opencl_pipeline.rs"
with open(file_path, "r") as f:
    content = f.read()

# Replace absurd extreme comparison
fixed_content = content.replace("assert!(exec.total_time_ns <= u64::MAX);", "assert!(true);")

with open(file_path, "w") as f:
    f.write(fixed_content)
