__kernel void gelu_approx(
    __global const float* input,
    __global float* output,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + tanh(c));
    }
}
