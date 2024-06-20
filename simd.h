// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_H_
#define HAY_SIMD_H_

#include "simd_base.h"
#include "simd_u64.h"

#if defined __aarch64__
#include "simd_arm.h"
#elif defined __x86_64__
#include "simd_x86.h"
#endif

#include <format>

template <Simd s> struct std::formatter<Uint1xN<s>> {
  template <typename FormatContext>
  auto format(const Uint1xN<s> &x, FormatContext &ctx) const {
    uint64_t buf[sizeof(Uint1xN<s>) / sizeof(uint64_t)];
    store(buf, x);
    auto it = ctx.out();
    it = std::format_to(it, "{{ ");
    for (uint64_t b : buf) {
      it = std::format_to(it, "0x{:016x} ", b);
    }
    it = std::format_to(it, "}}");
    return it;
  }
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
};

template <Simd s> struct std::formatter<Int64xN<s>> {
  template <typename FormatContext>
  auto format(const Int64xN<s> &x, FormatContext &ctx) const {
    auto it = ctx.out();
    it = std::format_to(it, "{{ ");
    for (int i = 0; i < Int64xN<s>::elem_count; ++i) {
      it = std::format_to(it, "{} ", extract(x, i));
    }
    it = std::format_to(it, "}}");
    return it;
  }
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
};

#endif // HAY_SIMD_H_
