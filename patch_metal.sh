cat << 'INNER_EOF' > crates/bitnet-kernels/tests/metal_buffer_alignment.rs
#[cfg(feature = "metal")]
mod metal_tests {
    use bitnet_kernels::metal_compute::{
        ThreadgroupConfig, compute_padded_length, compute_threadgroup_config,
    };
    #[test]
    fn buffer_alignment() {
        assert_eq!(compute_padded_length(1), 16);
    }
}
INNER_EOF
