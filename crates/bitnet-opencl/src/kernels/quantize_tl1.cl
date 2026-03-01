/// TL1 (Ternary Lookup 1) dequantization kernel.
///
/// 2 bits per value, byte-packed (4 values per byte).
/// Uses a 3-entry float lookup table for decoding.

__kernel void dequant_tl1(
    __global const uchar* packed,     // 2 bits per value, byte-packed
    __global const float* lut,         // 3-entry lookup table
    __global float* output,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;
    uint byte_idx = gid / 4;
    uint shift = (gid % 4) * 2;
    uint bits = (packed[byte_idx] >> shift) & 0x3;
    output[gid] = lut[bits];
}
