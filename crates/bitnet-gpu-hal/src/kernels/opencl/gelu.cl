__kernel void gelu(
    __global const float* input,
    __global float* output,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float cdf = 0.5f * (1.0f + erf(x * 0.7071067811865476f));
        output[i] = x * cdf;
    }
}
