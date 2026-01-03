// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_H_
#define HAY_SIMD_H_

#include <fmt/format.h>

#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512VBMI2__)
#include "vec_x86_avx512.h"
#elif defined __aarch64__
#include "vec_arm_neon.h"
#else
#error missing vec path
#endif

namespace hay {

template <typename T> struct ScalarTypeImpl {
  using Type = T;
};
template <> struct ScalarTypeImpl<uint1_vec_t> {
  using Type = uint8_t;
};
template <> struct ScalarTypeImpl<int64_vec_t> {
  using Type = int64_t;
};
template <typename T> using ScalarType = ScalarTypeImpl<T>::Type;

} // namespace hay

template <> struct fmt::formatter<hay::uint1_vec_t> {
  template <typename FormatContext>
  auto format(const hay::uint1_vec_t &x, FormatContext &ctx) const {
    static constexpr int buf_elems =
        sizeof(hay::uint1_vec_t) / sizeof(uint32_t);
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

template <> struct fmt::formatter<hay::int64_vec_t> {
  template <typename FormatContext>
  auto format(const hay::int64_vec_t &x, FormatContext &ctx) const {
    auto it = ctx.out();
    it = fmt::format_to(it, "{{");
    for (int i = 0; i < hay::int64_vec_t::elem_count; ++i) {
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
