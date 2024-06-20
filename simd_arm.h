// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_ARM_H_
#define HAY_SIMD_ARM_H_

#include "simd_base.h"

#include <arm_neon.h>
#include <bit>
#include <cassert>

template <> inline const char *name<Simd::Neon>() { return "NEON"; }

template <> inline bool detect<Simd::Neon>() { return true; }

template <> struct Int64xN<Simd::Neon> {
  static constexpr int elem_bits = 64;
  static constexpr int elem_count = 2;
  int64x2_t val;
  friend Int64xN add(Int64xN x, Int64xN y) { return {vaddq_s64(x.val, y.val)}; }
  friend Int64xN sub(Int64xN x, Int64xN y) { return {vsubq_s64(x.val, y.val)}; }
  friend Int64xN min(Int64xN x, Int64xN y) {
    return {vbslq_s64(vcleq_s64(x.val, y.val), x.val, y.val)};
  }
  friend Int64xN max(Int64xN x, Int64xN y) {
    return {vbslq_s64(vcgeq_s64(x.val, y.val), x.val, y.val)};
  }
  friend int64_t reduce_add(Int64xN x) {
    return {vgetq_lane_s64(x.val, 0) + vgetq_lane_s64(x.val, 1)};
  }
  static Int64xN load(const void *from) {
    return {vld1q_s64(static_cast<const int64_t *>(from))};
  }
  friend void store(void *to, Int64xN x) {
    vst1q_s64(static_cast<int64_t *>(to), x.val);
  }
  friend bool operator==(Int64xN x, Int64xN y) {
    uint64x2_t c = vceqq_s64(x.val, y.val);
    return vgetq_lane_u64(c, 0) && vgetq_lane_u64(c, 1);
  }
  static Int64xN zero() { return {vdupq_n_s64(0)}; }
  static Int64xN cst(int64_t c) { return {vdupq_n_s64(c)}; }
  static Int64xN wave() { return {vsetq_lane_s64(1, vdupq_n_s64(0), 1)}; }
  friend int64_t extract(Int64xN x, int i) {
    assert(i < elem_count);
    return {i == 0 ? vgetq_lane_s64(x.val, 0) : vgetq_lane_s64(x.val, 1)};
  }
};

template <> struct Uint1xN<Simd::Neon> {
  static constexpr int elem_bits = 1;
  static constexpr int elem_count = 128;
  uint64x2_t val;
  friend Uint1xN add(Uint1xN x, Uint1xN y) { return {veorq_u64(x.val, y.val)}; }
  friend Uint1xN mul(Uint1xN x, Uint1xN y) { return {vandq_u64(x.val, y.val)}; }
  friend Uint1xN madd(Uint1xN x, Uint1xN y, Uint1xN z) {
    return add(x, mul(y, z));
  }
  static Uint1xN load(const void *from) {
    return {vld1q_u64(static_cast<const uint64_t *>(from))};
  }
  friend void store(void *to, Uint1xN x) {
    vst1q_u64(static_cast<uint64_t *>(to), x.val);
  }
  friend bool operator==(Uint1xN x, Uint1xN y) {
    uint64x2_t c = vceqq_u64(x.val, y.val);
    return vgetq_lane_u64(c, 0) && vgetq_lane_u64(c, 1);
  }
  friend Int64xN<Simd::Neon> popcount64(Uint1xN x) {
    int64x2_t p = vdupq_n_s64(0);
    p = vsetq_lane_s64(std::popcount(vgetq_lane_u64(x.val, 0)), p, 0);
    p = vsetq_lane_s64(std::popcount(vgetq_lane_u64(x.val, 1)), p, 1);
    return {p};
  }
  friend Int64xN<Simd::Neon> lzcount64(Uint1xN x) {
    int64x2_t p = vdupq_n_s64(0);
    p = vsetq_lane_s64(std::countl_zero(vgetq_lane_u64(x.val, 0)), p, 0);
    p = vsetq_lane_s64(std::countl_zero(vgetq_lane_u64(x.val, 1)), p, 1);
    return {p};
  }
  static Uint1xN zero() { return {vdupq_n_u64(0)}; }
  static Uint1xN ones() { return {vdupq_n_u64(-1)}; }
  static Uint1xN wave(int i) {
    switch (i) {
    case 0:
      return {vreinterpretq_u64_u8(vdupq_n_u8(0xAA))};
    case 1:
      return {vreinterpretq_u64_u8(vdupq_n_u8(0xCC))};
    case 2:
      return {vreinterpretq_u64_u8(vdupq_n_u8(0xF0))};
    case 3:
      return {vreinterpretq_u64_u16(vdupq_n_u16(0xFF00))};
    case 4:
      return {vreinterpretq_u64_u32(vdupq_n_u32(0xFFFF0000u))};
    case 5:
      return {vdupq_n_u64(0xFFFFFFFF00000000u)};
    case 6:
      return {vcombine_u64(vdup_n_u64(0), vdup_n_u64(-1))};
    default:
      return zero();
    }
  }
  friend uint8_t extract(Uint1xN x, int i) {
    assert(i < elem_count);
    uint8x16_t a = vreinterpretq_u8_u64(x.val);
    uint8x8_t b = vtbl2_u8({vget_low_u8(a), vget_high_u8(a)}, vdup_n_u8(i / 8));
    uint8_t c = vget_lane_u8(b, 0);
    return {static_cast<uint8_t>((c >> (i % 8)) & 1)};
  }
};

#endif // HAY_SIMD_ARM_H_
