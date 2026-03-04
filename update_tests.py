import re

with open('crates/bitnet-common/tests/proptest_wave30_common.rs', 'r') as f:
    content = f.read()

content = content.replace(
    'use bitnet_common::shape_validator::{ShapeError, validate_matmul, validate_batched_matmul, validate_elementwise, validate_layer_norm, validate_embedding, validate_reshape, validate_attention\n};',
    'use bitnet_common::shape_validator::{assert_broadcastable, assert_dim, assert_element_count, assert_head_divisible, assert_matmul_compat, assert_rank, assert_shape_eq};'
)

with open('crates/bitnet-common/tests/proptest_wave30_common.rs', 'w') as f:
    f.write(content)
