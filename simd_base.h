// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_SIMD_BASE_H_
#define HAY_SIMD_BASE_H_

#include <cstdint>

enum class Simd {
  U64,
  Neon,
  Avx512,
};

template <Simd s> struct Uint1xN {};
template <Simd s> struct Int64xN {};

template <Simd s> const char *name();
template <Simd s> bool detect();

#endif // HAY_SIMD_BASE_H_
