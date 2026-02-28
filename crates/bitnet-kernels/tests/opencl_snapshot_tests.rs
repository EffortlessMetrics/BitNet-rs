//! Insta snapshot tests for OpenCL kernel sources, device display, and ternary encoding.
//!
//! These tests pin the exact text of kernel sources and type representations so
//! that unintentional changes are caught during code review.

use bitnet_common::Device;
use bitnet_kernels::kernels::{ELEMENTWISE_SRC, MATMUL_I2S_SRC, QUANTIZE_I2S_SRC};

// ── Kernel source snapshots ──────────────────────────────────────────────────

#[test]
fn matmul_i2s_kernel_source() {
    insta::assert_snapshot!(MATMUL_I2S_SRC);
}

#[test]
fn quantize_i2s_kernel_source() {
    insta::assert_snapshot!(QUANTIZE_I2S_SRC);
}

#[test]
fn elementwise_kernel_source() {
    insta::assert_snapshot!(ELEMENTWISE_SRC);
}

// ── Device display snapshots ─────────────────────────────────────────────────

#[test]
fn device_opencl_debug() {
    insta::assert_snapshot!(format!("{:?}", Device::OpenCL(0)));
}

#[test]
fn device_opencl_debug_index_1() {
    insta::assert_snapshot!(format!("{:?}", Device::OpenCL(1)));
}

// ── Ternary encoding snapshots ───────────────────────────────────────────────

/// Pack ternary values (-1, 0, +1) into bytes using the I2_S encoding.
///
/// Encoding: +1 → 0b01, −1 → 0b11, 0 → 0b00. Four values per byte.
fn pack_ternary(values: &[i8]) -> Vec<u8> {
    values
        .chunks(4)
        .map(|chunk| {
            let mut packed = 0u8;
            for (i, &v) in chunk.iter().enumerate() {
                let bits = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                packed |= bits << (i * 2);
            }
            packed
        })
        .collect()
}

#[test]
fn ternary_pack_known_values() {
    // [0, 1, -1, 0] → pos0:0b00, pos1:0b01, pos2:0b11, pos3:0b00 → 0x34
    let packed = pack_ternary(&[0, 1, -1, 0]);
    insta::assert_debug_snapshot!(packed);
}

#[test]
fn ternary_pack_all_positive() {
    // [1, 1, 1, 1] → 0x55
    let packed = pack_ternary(&[1, 1, 1, 1]);
    insta::assert_debug_snapshot!(packed);
}

#[test]
fn ternary_pack_all_negative() {
    // [-1, -1, -1, -1] → 0xFF
    let packed = pack_ternary(&[-1, -1, -1, -1]);
    insta::assert_debug_snapshot!(packed);
}

#[test]
fn ternary_pack_alternating() {
    // [1, -1, 1, -1] → 0b11_01_11_01 = 0xD5
    let packed = pack_ternary(&[1, -1, 1, -1]);
    insta::assert_debug_snapshot!(packed);
}
