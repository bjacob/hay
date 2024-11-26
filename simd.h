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

#include <fmt/format.h>

template <Simd s> struct fmt::formatter<Uint1xN<s>> {
  template <typename FormatContext>
  auto format(const Uint1xN<s> &x, FormatContext &ctx) const {
    static constexpr int buf_elems = sizeof(Uint1xN<s>) / sizeof(uint64_t);
    uint64_t buf[buf_elems];
    store(buf, x);
    auto it = ctx.out();
    it = fmt::format_to(it, "{{");
    for (int i = 0; i < buf_elems; ++i) {
      if (i > 0) {
        it = fmt::format_to(it, ", ");
      }
      it = fmt::format_to(it, "0x{:016x}", buf[i]);
    }
    it = fmt::format_to(it, "}}");
    return it;
  }
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }
};

template <Simd s> struct fmt::formatter<Int64xN<s>> {
  template <typename FormatContext>
  auto format(const Int64xN<s> &x, FormatContext &ctx) const {
    auto it = ctx.out();
    it = fmt::format_to(it, "{{");
    for (int i = 0; i < Int64xN<s>::elem_count; ++i) {
      if (i > 0) {
        it = fmt::format_to(it, ", ");
      }
      it = fmt::format_to(it, "{}", extract(x, i));
    }
    it = fmt::format_to(it, "}}");
    return it;
  }
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }
};

#endif // HAY_SIMD_H_
