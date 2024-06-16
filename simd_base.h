// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_BASE_H_
#define HAY_SIMD_BASE_H_

#include <cstdint>
#include <format>

enum class Simd {
  Uint64,
  Neon,
  Avx512,
};

struct Uint1 {
  static constexpr int elem_bits = 1;
  static constexpr int elem_count = 1;
  uint8_t val : 1;
};

struct Int64 {
  static constexpr int elem_bits = 64;
  static constexpr int elem_count = 1;
  int64_t val;
};

template <> struct std::formatter<Uint1> {
  auto format(const Uint1 &x, std::format_context &ctx) const {
    return std::format_to(ctx.out(), "{}", x.val);
  }
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
};

template <> struct std::formatter<Int64> {
  auto format(const Uint1 &x, std::format_context &ctx) const {
    return std::format_to(ctx.out(), "{}", x.val);
  }
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
};

template <Simd s> struct Uint1xN {};
template <Simd s> struct Int64xN {};

template <Simd s> const char *name();
template <Simd s> bool detect();

#endif // HAY_SIMD_BASE_H_
