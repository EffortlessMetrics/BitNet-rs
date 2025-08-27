#include "ggml-quants.h"
#include <math.h>
#include <string.h>
#include <assert.h>

// simple fp16 helpers
static inline ggml_fp16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    ggml_fp16_t h;
    if (exp <= 0) {
        if (exp < -10) {
            h = sign;
        } else {
            mant |= 0x800000u;
            uint32_t t = mant >> (1 - exp);
            h = sign | (t >> 13);
        }
    } else if (exp >= 31) {
        h = sign | 0x7c00u;
    } else {
        h = sign | (exp << 10) | (mant >> 13);
    }
    return h;
}

static inline float fp16_to_fp32(ggml_fp16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            exp += 127 - 15;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        exp = exp + 127 - 15;
        f = sign | (exp << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

void dequantize_row_iq2_s(const void * restrict vx, float * restrict y, int64_t k) {
    assert(k % QK_IQ2_S == 0);
    const block_iq2_s * x = (const block_iq2_s *)vx;
    const float qmap[4] = { -2.f, -1.f, 1.f, 2.f };
    int64_t nb = k / QK_IQ2_S;
    for (int64_t ib = 0; ib < nb; ++ib) {
        float d = fp16_to_fp32(x[ib].d);
        const uint8_t * qs = x[ib].qs;
        for (int j = 0; j < QK_IQ2_S; ++j) {
            uint8_t q = (qs[j/4] >> (2*(j%4))) & 0x3;
            y[ib*QK_IQ2_S + j] = d * qmap[q];
        }
    }
}

size_t quantize_iq2_s(const float * restrict src, void * restrict dst,
                      int64_t nrows, int64_t n_per_row,
                      const float * quant_weights) {
    (void)quant_weights; // unused in this simplified implementation
    assert(n_per_row % QK_IQ2_S == 0);
    int64_t nblock = n_per_row / QK_IQ2_S;
    block_iq2_s * blocks = (block_iq2_s *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        const float * xrow = src + row*n_per_row;
        for (int64_t ib = 0; ib < nblock; ++ib) {
            const float * x = xrow + ib*QK_IQ2_S;
            block_iq2_s * b = &blocks[row*nblock + ib];
            float max = 0.f;
            for (int i = 0; i < QK_IQ2_S; ++i) {
                float v = fabsf(x[i]);
                if (v > max) max = v;
            }
            if (max == 0.f) max = 1e-8f;
            float d = max / 2.f; // map to levels {-2,-1,1,2}
            b->d = fp32_to_fp16(d);
            memset(b->qs, 0, sizeof(b->qs));
            memset(b->qh, 0, sizeof(b->qh));
            memset(b->scales, 0, sizeof(b->scales));
            float id = 1.0f / d;
            for (int i = 0; i < QK_IQ2_S; ++i) {
                float v = x[i] * id;
                int q;
                if (v < -1.5f) q = 0;
                else if (v < 0.f) q = 1;
                else if (v < 1.5f) q = 2;
                else q = 3;
                b->qs[i/4] |= (q & 0x3) << (2*(i%4));
            }
        }
    }
    return (size_t)(nrows * nblock * sizeof(block_iq2_s));
}

size_t quantize_row_iq2_s_ref(const float * restrict x, block_iq2_s * restrict y, int64_t k) {
    return quantize_iq2_s(x, y, 1, k, NULL);
}
