__kernel void silu(
    __global const float* input,
    __global float* output,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        output[i] = x / (1.0f + exp(-x));
    }
}
