// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_X86_AVX512_H_
#define HAY_SIMD_X86_AVX512_H_

#include <cassert>
#include <immintrin.h>

namespace hay {

struct int64_vec_t {
  static constexpr int elem_bits = 64;
  static constexpr int elem_count = 8;
  __m512i val;
  friend int64_vec_t add(int64_vec_t x, int64_vec_t y) {
    return {_mm512_add_epi64(x.val, y.val)};
  }
  friend int64_vec_t sub(int64_vec_t x, int64_vec_t y) {
    return {_mm512_sub_epi64(x.val, y.val)};
  }
  friend int64_vec_t min(int64_vec_t x, int64_vec_t y) {
    return {_mm512_min_epi64(x.val, y.val)};
  }
  friend int64_vec_t max(int64_vec_t x, int64_vec_t y) {
    return {_mm512_max_epi64(x.val, y.val)};
  }
  friend int64_t reduce_add(int64_vec_t x) {
    return _mm512_reduce_add_epi64(x.val);
  }
  static int64_vec_t load(const void *from) {
    return {_mm512_loadu_si512(from)};
  }
  friend void store(void *to, int64_vec_t x) { _mm512_storeu_si512(to, x.val); }
  friend bool operator==(int64_vec_t x, int64_vec_t y) {
    return _mm512_cmp_epi64_mask(x.val, y.val, _MM_CMPINT_EQ) == 0xFF;
  }
  static int64_vec_t cst(int64_t c) { return {_mm512_set1_epi64(c)}; }
  friend int64_t extract(int64_vec_t x, int i) {
    assert(i < elem_count);
    return _mm256_extract_epi64(_mm512_castsi512_si256(_mm512_permutexvar_epi64(
                                    _mm512_set1_epi64(i), x.val)),
                                0);
  }
};

struct uint1_vec_t {
  static constexpr int elem_bits = 1;
  static constexpr int elem_count = 512;
  __m512i val;
  friend uint1_vec_t add(uint1_vec_t x, uint1_vec_t y) {
    return {_mm512_xor_si512(x.val, y.val)};
  }
  friend uint1_vec_t mul(uint1_vec_t x, uint1_vec_t y) {
    return {_mm512_and_si512(x.val, y.val)};
  }
  friend uint1_vec_t madd(uint1_vec_t x, uint1_vec_t y, uint1_vec_t z) {
    return {_mm512_ternarylogic_epi64(x.val, y.val, z.val, 0x78)};
  }
  static uint1_vec_t load(const void *from) {
    return {_mm512_loadu_si512(from)};
  }
  friend void store(void *to, uint1_vec_t x) { _mm512_storeu_si512(to, x.val); }
  friend bool operator==(uint1_vec_t x, uint1_vec_t y) {
    return _mm512_cmp_epi64_mask(x.val, y.val, _MM_CMPINT_EQ) == 0xFF;
  }
  friend int64_vec_t popcount(uint1_vec_t x) {
    return {_mm512_popcnt_epi64(x.val)};
  }
  static uint1_vec_t cst(uint8_t i) {
    return {_mm512_set1_epi8(i == 0 ? 0 : 0xFF)};
  }
  static uint1_vec_t seq(int i) {
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
  friend uint8_t extract(uint1_vec_t x, int i) {
    assert(i < elem_count);
    __mmask32 mask = _cvtu32_mask32(1u << (i / 16));
    __m512i compress = _mm512_maskz_compress_epi16(mask, x.val);
    uint16_t low16 = _mm256_extract_epi16(_mm512_castsi512_si256(compress), 0);
    uint8_t bit = (low16 >> (i % 16)) & 1;
    return bit;
  }
};

} // namespace hay

#endif // HAY_SIMD_X86_AVX512_H_
