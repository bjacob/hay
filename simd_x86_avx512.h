// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_X86_AVX512_H_
#define HAY_SIMD_X86_AVX512_H_

#include <cassert>
#include <immintrin.h>

struct Int64xN {
  static constexpr int elem_bits = 64;
  static constexpr int elem_count = 8;
  __m512i val;
  friend Int64xN add(Int64xN x, Int64xN y) {
    return {_mm512_add_epi64(x.val, y.val)};
  }
  friend Int64xN sub(Int64xN x, Int64xN y) {
    return {_mm512_sub_epi64(x.val, y.val)};
  }
  friend Int64xN min(Int64xN x, Int64xN y) {
    return {_mm512_min_epi64(x.val, y.val)};
  }
  friend Int64xN max(Int64xN x, Int64xN y) {
    return {_mm512_max_epi64(x.val, y.val)};
  }
  friend int64_t reduce_add(Int64xN x) {
    return _mm512_reduce_add_epi64(x.val);
  }
  static Int64xN load(const void *from) { return {_mm512_loadu_si512(from)}; }
  friend void store(void *to, Int64xN x) { _mm512_storeu_si512(to, x.val); }
  friend bool operator==(Int64xN x, Int64xN y) {
    return _mm512_cmp_epi64_mask(x.val, y.val, _MM_CMPINT_EQ) == 0xFF;
  }
  static Int64xN cst(int64_t c) { return {_mm512_set1_epi64(c)}; }
  friend int64_t extract(Int64xN x, int i) {
    assert(i < elem_count);
    return _mm256_extract_epi64(_mm512_castsi512_si256(_mm512_permutexvar_epi64(
                                    _mm512_set1_epi64(i), x.val)),
                                0);
  }
};

struct Uint1xN {
  static constexpr int elem_bits = 1;
  static constexpr int elem_count = 512;
  __m512i val;
  friend Uint1xN add(Uint1xN x, Uint1xN y) {
    return {_mm512_xor_si512(x.val, y.val)};
  }
  friend Uint1xN mul(Uint1xN x, Uint1xN y) {
    return {_mm512_and_si512(x.val, y.val)};
  }
  friend Uint1xN madd(Uint1xN x, Uint1xN y, Uint1xN z) {
    return {_mm512_ternarylogic_epi64(x.val, y.val, z.val, 0x78)};
  }
  static Uint1xN load(const void *from) { return {_mm512_loadu_si512(from)}; }
  friend void store(void *to, Uint1xN x) { _mm512_storeu_si512(to, x.val); }
  friend bool operator==(Uint1xN x, Uint1xN y) {
    return _mm512_cmp_epi64_mask(x.val, y.val, _MM_CMPINT_EQ) == 0xFF;
  }
  friend Int64xN popcount(Uint1xN x) { return {_mm512_popcnt_epi64(x.val)}; }
  static Uint1xN cst(uint8_t i) {
    return {_mm512_set1_epi8(i == 0 ? 0 : 0xFF)};
  }
  static Uint1xN seq(int i) {
    switch (i) {
    case 0:
      return {_mm512_set1_epi8(0xAA)};
    case 1:
      return {_mm512_set1_epi8(0xCC)};
    case 2:
      return {_mm512_set1_epi8(0xF0)};
    case 3:
      return {_mm512_set1_epi16(0xFF00)};
    case 4:
      return {_mm512_set1_epi32(0xFFFF0000u)};
    case 5:
      return {_mm512_set1_epi64(0xFFFFFFFF00000000u)};
    case 6:
      return {_mm512_unpacklo_epi64(_mm512_setzero_si512(),
                                    _mm512_set1_epi8(0xFF))};
    case 7: {
      __m256i a = _mm256_setr_m128i(_mm_setzero_si128(), _mm_set1_epi8(0xFF));
      return {_mm512_inserti64x4(_mm512_castsi256_si512(a), a, 1)};
    }
    case 8:
      return {_mm512_inserti64x4(_mm512_setzero_si512(), _mm256_set1_epi8(0xFF),
                                 1)};
    default:
      return {_mm512_setzero_si512()};
    }
  }
  friend uint8_t extract(Uint1xN x, int i) {
    assert(i < elem_count);
    __mmask32 mask = _cvtu32_mask32(1u << (i / 16));
    __m512i compress = _mm512_maskz_compress_epi16(mask, x.val);
    uint16_t low16 = _mm256_extract_epi16(_mm512_castsi512_si256(compress), 0);
    uint8_t bit = (low16 >> (i % 16)) & 1;
    return bit;
  }
};

#endif // HAY_SIMD_X86_AVX512_H_
