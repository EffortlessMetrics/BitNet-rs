/// I2_S (2-bit integer symmetric) quantization kernels.
///
/// Packing: 16 ternary values {-1, 0, 1} into one 32-bit word (2 bits each).
/// Encoding: -1 → 0b10, 0 → 0b00, 1 → 0b01.

// Pack 16 ternary values {-1, 0, 1} into 32 bits (2 bits each)
__kernel void pack_i2s(
    __global const int* values,    // input ternary values
    __global uint* packed,         // output packed data
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count / 16) return;
    uint result = 0;
    for (int i = 0; i < 16; i++) {
        int v = values[gid * 16 + i];
        // Map: -1 -> 0b10, 0 -> 0b00, 1 -> 0b01
        uint bits = (v == -1) ? 2u : (uint)v;
        result |= (bits & 0x3) << (i * 2);
    }
    packed[gid] = result;
}

// Unpack I2_S packed data to float
__kernel void unpack_i2s(
    __global const uint* packed,
    __global float* output,
    const float scale,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;
    uint word_idx = gid / 16;
    uint bit_idx = (gid % 16) * 2;
    uint bits = (packed[word_idx] >> bit_idx) & 0x3;
    // Decode: 0b00 -> 0.0, 0b01 -> 1.0, 0b10 -> -1.0
    float val = (bits == 2u) ? -1.0f : (float)bits;
    output[gid] = val * scale;
}
