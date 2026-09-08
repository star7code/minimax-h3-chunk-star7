// SPDX-License-Identifier: MIT
// V2 CPU transfer conversion. This translation unit alone is built for AVX2,
// while the main bridge stays baseline-compatible and calls it only on CPUs
// whose OS-enabled feature set includes AVX and F16C.

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <intrin.h>

extern "C" int dlss5nr_cpu_has_f16c() {
    static const int available = []() {
        int regs[4]{};
        __cpuid(regs, 1);
        constexpr int OSXSAVE = 1 << 27;
        constexpr int AVX = 1 << 28;
        constexpr int F16C = 1 << 29;
        if ((regs[2] & (OSXSAVE | AVX | F16C)) != (OSXSAVE | AVX | F16C)) return 0;
        return (_xgetbv(0) & 0x6) == 0x6 ? 1 : 0;
    }();
    return available;
}

extern "C" void dlss5nr_rgb32_to_rgba16(const float* src, uint16_t* dst, int pixels) {
    const __m128 zero = _mm_setzero_ps();
    const __m128 one = _mm_set1_ps(1.0f);
    for (int x = 0; x < pixels; ++x) {
        __m128 rgba = _mm_set_ps(1.0f, src[2], src[1], src[0]);
        rgba = _mm_min_ps(one, _mm_max_ps(zero, rgba));
        const __m128i half = _mm_cvtps_ph(rgba, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), half);
        src += 3;
        dst += 4;
    }
}

extern "C" void dlss5nr_rgba16_to_rgb32(const uint16_t* src, float* dst, int pixels) {
    const __m128 zero = _mm_setzero_ps();
    const __m128 one = _mm_set1_ps(1.0f);
    alignas(16) float rgba[4];
    for (int x = 0; x < pixels; ++x) {
        const __m128i half = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
        __m128 values = _mm_cvtph_ps(half);
        values = _mm_min_ps(one, _mm_max_ps(zero, values));
        _mm_store_ps(rgba, values);
        dst[0] = rgba[0];
        dst[1] = rgba[1];
        dst[2] = rgba[2];
        src += 4;
        dst += 3;
    }
}
