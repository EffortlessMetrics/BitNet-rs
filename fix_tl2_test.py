#!/usr/bin/env python3
"""Fix TL2 test to use Vec<f32> for weight data instead of ConcreteTensor"""

import re

# Read the file
with open('crates/bitnet-inference/tests/ac1_quantized_linear_layers.rs', 'r') as f:
    content = f.read()

# Find and fix the weight data creation in TL2 test
# Replace the create_mock_weight_matrix call with direct Vec<f32> creation like TL1
old_pattern = r'''    // Create input and weight tensors
    let input = create_mock_tensor\(
        test_config\.batch_size,
        test_config\.sequence_length,
        test_config\.hidden_size,
    \)\?;
    let weight_data = create_mock_weight_matrix\(test_config\.hidden_size, test_config\.intermediate_size\)\?;'''

new_code = '''    // Create real input tensor with non-zero values for meaningful computation
    let input_data: Vec<f32> = (0..test_config.batch_size * test_config.sequence_length * test_config.hidden_size)
        .map(|i| ((i % 100) as f32 / 100.0) - 0.5) // Range: [-0.5, 0.49]
        .collect();
    let input = BitNetTensor::from_slice(
        &input_data,
        &[test_config.batch_size, test_config.sequence_length, test_config.hidden_size],
        &Device::Cpu,
    )
    .context("Failed to create input tensor")?;

    // Create real weight matrix with non-zero values
    let weight_data_vec: Vec<f32> = (0..test_config.hidden_size * test_config.intermediate_size)
        .map(|i| ((i % 50) as f32 / 50.0) - 0.5) // Range: [-0.5, 0.48]
        .collect();
    let weight_data = BitNetTensor::from_slice(
        &weight_data_vec,
        &[test_config.hidden_size, test_config.intermediate_size],
        &Device::Cpu,
    )
    .context("Failed to create weight tensor")?;'''

content = content.replace(old_pattern, new_code)

# Also fix the weight statistics calculation
content = content.replace(
    "    let weight_vec = weight_data.to_vec().context(\"Failed to convert weight data to vector\")?;\n    let weight_stats = calculate_tensor_statistics(&weight_vec)?;",
    "    let weight_stats = calculate_tensor_statistics(&weight_data_vec)?"
)

# Write the updated content
with open('crates/bitnet-inference/tests/ac1_quantized_linear_layers.rs', 'w') as f:
    f.write(content)

print("TL2 test fixed to use Vec<f32> for weight data!")
