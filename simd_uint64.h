// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_UINT64_H_
#define HAY_SIMD_UINT64_H_

#include "simd_base.h"

#include <algorithm>
#include <bit>
#include <cstdint>

template <> inline const char *name<Simd::Uint64>() { return "Uint64"; }

template <> inline bool detect<Simd::Uint64>() { return true; }

template <> struct Int64xN<Simd::Uint64> {
  static constexpr int elem_bits = 64;
  static constexpr int elem_count = 1;
  int64_t val;
  friend Int64xN add(Int64xN x, Int64xN y) { return {x.val + y.val}; }
  friend Int64xN sub(Int64xN x, Int64xN y) { return {x.val - y.val}; }
  friend Int64xN min(Int64xN x, Int64xN y) { return {std::min(x.val, y.val)}; }
  friend Int64xN max(Int64xN x, Int64xN y) { return {std::max(x.val, y.val)}; }
  friend Int64 reduce_add(Int64xN x) { return {x.val}; }
  static Int64xN load(const void *from) {
    return {*static_cast<const int64_t *>(from)};
  }
  friend void store(void *to, Int64xN x) {
    *static_cast<int64_t *>(to) = x.val;
  }
  friend bool operator==(Int64xN x, Int64xN y) { return x.val == y.val; }
  static Int64xN zero() { return {0}; }
  static Int64xN cst(int64_t c) { return {c}; }
  static Int64xN wave() { return {0}; }
};

template <> struct Uint1xN<Simd::Uint64> {
  static constexpr int elem_bits = 1;
  static constexpr int elem_count = 64;
  uint64_t val;
  friend Uint1xN add(Uint1xN x, Uint1xN y) { return {x.val ^ y.val}; }
  friend Uint1xN mul(Uint1xN x, Uint1xN y) { return {x.val & y.val}; }
  friend Uint1xN madd(Uint1xN x, Uint1xN y, Uint1xN z) {
    return {x.val ^ (y.val & z.val)};
  }
  static Uint1xN load(const void *from) {
    return {*static_cast<const uint64_t *>(from)};
  }
  friend void store(void *to, Uint1xN x) {
    *static_cast<uint64_t *>(to) = x.val;
  }
  friend bool operator==(Uint1xN x, Uint1xN y) { return x.val == y.val; }
  friend Int64xN<Simd::Uint64> popcount64(Uint1xN x) {
    return {std::popcount(x.val)};
  }
  friend Int64xN<Simd::Uint64> lzcount64(Uint1xN x) {
    return {std::countl_zero(x.val)};
  }
  static Uint1xN zero() { return {0}; }
  static Uint1xN ones() { return {0xFFFFFFFFFFFFFFFFu}; }
  static Uint1xN wave(int i) {
    switch (i) {
    case 0:
      return {0xAAAAAAAAAAAAAAAAu};
    case 1:
      return {0xCCCCCCCCCCCCCCCCu};
    case 2:
      return {0xF0F0F0F0F0F0F0F0u};
    case 3:
      return {0xFF00FF00FF00FF00u};
    case 4:
      return {0xFFFF0000FFFF0000u};
    case 5:
      return {0xFFFFFFFF00000000u};
    default:
      return zero();
    }
  }
};

#endif // HAY_SIMD_UINT64_H_
