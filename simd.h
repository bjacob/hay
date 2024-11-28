// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_H_
#define HAY_SIMD_H_

#include <fmt/format.h>

#if defined __HIP_DEVICE_COMPILE__
#include "simd_u32_u64.h"
#elif defined(__AVX512VPOPCNTDQ__) && defined(__AVX512VBMI2__)
#include "simd_x86_avx512.h"
#elif defined __aarch64__
#include "simd_arm_neon.h"
#else
#include "simd_u32_u64.h"
#endif

template <typename T> struct ScalarTypeImpl {
  using Type = T;
};
template <> struct ScalarTypeImpl<Uint1xN> {
  using Type = uint8_t;
};
template <> struct ScalarTypeImpl<Int64xN> {
  using Type = int64_t;
};
template <typename T> using ScalarType = ScalarTypeImpl<T>::Type;

template <> struct fmt::formatter<Uint1xN> {
  template <typename FormatContext>
  auto format(const Uint1xN &x, FormatContext &ctx) const {
    static constexpr int buf_elems = sizeof(Uint1xN) / sizeof(uint32_t);
    uint32_t buf[buf_elems];
    store(buf, x);
    auto it = ctx.out();
    it = fmt::format_to(it, "{{");
    for (int i = 0; i < buf_elems; ++i) {
      if (i > 0) {
        it = fmt::format_to(it, ", ");
      }
      it = fmt::format_to(it, "0x{:08x}", buf[i]);
    }
    it = fmt::format_to(it, "}}");
    return it;
  }
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }
};

template <> struct fmt::formatter<Int64xN> {
  template <typename FormatContext>
  auto format(const Int64xN &x, FormatContext &ctx) const {
    auto it = ctx.out();
    it = fmt::format_to(it, "{{");
    for (int i = 0; i < Int64xN::elem_count; ++i) {
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
