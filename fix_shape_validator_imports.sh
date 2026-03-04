sed -i 's/use bitnet_common::shape_validator::{/use bitnet_common::shape_validator::{ShapeError, validate_matmul, validate_batched_matmul, validate_elementwise, validate_layer_norm, validate_embedding, validate_reshape, validate_attention/g' crates/bitnet-common/tests/proptest_wave30_common.rs
sed -i '/assert_broadcastable, assert_dim, assert_element_count, assert_head_divisible,/d' crates/bitnet-common/tests/proptest_wave30_common.rs
sed -i '/assert_matmul_compat, assert_rank, assert_shape_eq,/d' crates/bitnet-common/tests/proptest_wave30_common.rs
