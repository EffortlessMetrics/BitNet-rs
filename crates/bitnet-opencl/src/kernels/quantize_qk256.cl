/// QK256 quantization kernels.
///
/// 256-element blocks with one f16 per-block scale.
/// Each block: 16 × 32-bit words (256 ternary values at 2 bits each) + 1 f16 scale.
/// Work-group size: 16 (one work-item per 32-bit word).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Dequantize QK256 block: 256 ternary values with one f16 scale
__kernel void dequant_qk256(
    __global const uint* packed_data,  // packed ternary (2 bits each)
    __global const half* scales,       // per-block f16 scale
    __global float* output,            // dequantized output
    const uint num_blocks
) {
    uint block_id = get_group_id(0);
    uint local_id = get_local_id(0);
    if (block_id >= num_blocks) return;

    float scale = vload_half(block_id, (__global const half*)scales);

    // Each work item handles one 32-bit word = 16 values
    uint word_offset = block_id * 16 + local_id; // 256/16 = 16 words per block
    uint val_offset = block_id * 256 + local_id * 16;

    uint word = packed_data[word_offset];
    for (int i = 0; i < 16; i++) {
        uint bits = (word >> (i * 2)) & 0x3;
        float val = (bits == 2u) ? -1.0f : (float)bits;
        output[val_offset + i] = val * scale;
    }
}

// Quantize float data to QK256 format
__kernel void quant_qk256(
    __global const float* input,
    __global uint* packed_output,
    __global half* scales_output,
    const uint num_blocks
) {
    uint block_id = get_group_id(0);
    uint local_id = get_local_id(0);
    if (block_id >= num_blocks) return;

    // Find max abs value in block (reduction)
    __local float shared_max[16];
    float local_max = 0.0f;
    uint base = block_id * 256 + local_id * 16;
    for (int i = 0; i < 16; i++) {
        local_max = fmax(local_max, fabs(input[base + i]));
    }
    shared_max[local_id] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction for max
    for (int s = 8; s > 0; s >>= 1) {
        if (local_id < (uint)s) {
            shared_max[local_id] =
                fmax(shared_max[local_id], shared_max[local_id + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float scale = shared_max[0];
    if (local_id == 0) {
        vstore_half(scale, block_id, (__global half*)scales_output);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Quantize to ternary
    float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    uint packed = 0;
    for (int i = 0; i < 16; i++) {
        float v = input[base + i] * inv_scale;
        int q = (v > 0.5f) ? 1 : (v < -0.5f) ? -1 : 0;
        uint bits = (q == -1) ? 2u : (uint)q;
        packed |= (bits & 0x3) << (i * 2);
    }
    packed_output[block_id * 16 + local_id] = packed;
}
