/// TL2 (Ternary Lookup 2) dequantization kernel.
///
/// Paired vectorized lookup: each work-item decodes two adjacent values
/// from a byte using a float2 LUT for coalesced memory writes.

__kernel void dequant_tl2(
    __global const uchar* packed,
    __global const float2* lut_pairs,  // paired LUT for vectorized lookup
    __global float* output,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid * 2 >= count) return;
    uint byte_idx = gid / 2;
    uint shift = (gid % 2) * 4;
    uint bits0 = (packed[byte_idx] >> shift) & 0x3;
    uint bits1 = (packed[byte_idx] >> (shift + 2)) & 0x3;
    output[gid * 2] = lut_pairs[bits0].x;
    output[gid * 2 + 1] = lut_pairs[bits1].y;
}
