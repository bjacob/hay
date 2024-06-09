// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_X86_H_
#define HAY_SIMD_X86_H_

#include <emmintrin.h>
#include <immintrin.h>

#include "simd_base.h"

#if defined(__AVX512VPOPCNTDQ__)

template <> struct SimdDefinition<Simd::Avx512> {
  using Reg = __m512i;
  static const char *name() { return "AVX-512"; }
  static bool detectCpu() { return true; }
  static Reg add(Reg x, Reg y) { return _mm512_xor_si512(x, y); }
  static Reg mul(Reg x, Reg y) { return _mm512_and_si512(x, y); }
  static Reg madd(Reg x, Reg y, Reg z) {
    return _mm512_ternarylogic_epi64(x, y, z, 0x78);
  }
  static Reg load(const void *from) { return _mm512_loadu_si512(from); }
  static void store(void *to, Reg x) { _mm512_storeu_si512(to, x); }
  static bool equal(Reg x, Reg y) {
    return _mm512_cmp_epi64_mask(x, y, _MM_CMPINT_EQ) == 0xFF;
  }
  static int popcount(Reg x) {
    return _mm512_reduce_add_epi64(_mm512_popcnt_epi64(x));
  }
  static Reg zero() { return _mm512_setzero_si512(); }
  static Reg ones() { return _mm512_set1_epi8(0xFF); }
  static Reg wave(int i) {
    switch (i) {
    case 0:
      return _mm512_set1_epi8(0xAA);
    case 1:
      return _mm512_set1_epi8(0xCC);
    case 2:
      return _mm512_set1_epi8(0xF0);
    case 3:
      return _mm512_set1_epi16(0xFF00);
    case 4:
      return _mm512_set1_epi32(0xFFFF0000u);
    case 5:
      return _mm512_set1_epi64(0xFFFFFFFF00000000u);
    case 6:
      return _mm512_unpacklo_epi64(zero(), ones());
    case 7: {
      __m256i a = _mm256_setr_m128i(_mm_setzero_si128(), _mm_set1_epi8(0xFF));
      return _mm512_inserti64x4(_mm512_castsi256_si512(a), a, 1);
    }
    case 8:
      return _mm512_inserti64x4(zero(), _mm256_set1_epi8(0xFF), 1);
    default:
      return zero();
    }
  }
};

#endif // defined(__AVX512VPOPCNTDQ__)

#endif // HAY_SIMD_X86_H_
