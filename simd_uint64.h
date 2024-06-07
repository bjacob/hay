// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_UINT64_H_
#define HAY_SIMD_UINT64_H_

#include "simd_base.h"

#include <bit>
#include <cstdint>

template <> struct SimdDefinition<Simd::Uint64> {
  using Reg = uint64_t;
  static const char *name() { return "Uint64"; }
  static bool detectCpu() { return true; }
  static Reg add(Reg x, Reg y) { return x ^ y; }
  static Reg mul(Reg x, Reg y) { return x & y; }
  static Reg madd(Reg x, Reg y, Reg z) { return add(mul(x, y), z); }
  static Reg load(const void *from) { return *static_cast<const Reg *>(from); }
  static void store(void *to, Reg x) { *static_cast<Reg *>(to) = x; }
  static bool equal(Reg x, Reg y) { return x == y; }
  static int popcount(Reg x) { return std::popcount(x); }
  static Reg zero() { return 0; }
  static Reg ones() { return -1; }
  static Reg wave(int i) {
    switch (i) {
    case 0:
      return 0xAAAAAAAAAAAAAAAAu;
    case 1:
      return 0xCCCCCCCCCCCCCCCCu;
    case 2:
      return 0xF0F0F0F0F0F0F0F0u;
    case 3:
      return 0xFF00FF00FF00FF00u;
    case 4:
      return 0xFFFF0000FFFF0000u;
    case 5:
      return 0xFFFFFFFF00000000u;
    default:
      return 0;
    }
  }
};

#endif // HAY_SIMD_UINT64_H_
