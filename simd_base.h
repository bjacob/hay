// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_BASE_H_
#define HAY_SIMD_BASE_H_

enum class Simd {
  Uint64,
  Neon,
  Avx512,
};

template <Simd s> struct SimdDefinition {};

template <Simd s> struct SimdTraits : SimdDefinition<s> {
  using Base = SimdDefinition<s>;
  using Reg = Base::Reg;
  static constexpr int RegBytes = sizeof(Reg);
  static constexpr int RegBits = 8 * RegBytes;
};

#endif // HAY_SIMD_BASE_H_
