// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_Vector_H_
#define HAY_Vector_H_

#include "simd_base.h"
#include <array>

template <Simd simd, int... sizesPack> class Vector {
public:
  using ST = SimdTraits<simd>;
  using Reg = ST::Reg;
  static constexpr int order = sizeof...(sizesPack);
  static constexpr int flatSize = (sizesPack * ...);
  using Indices = std::array<int, order>;
  static Indices getSizes() { return Indices(sizesPack...); }

  static Indices getStrides() {
    auto sizes = getSizes();
    Indices strides;
    int product = 1;
    for (int i = order - 1; i >= 0; --i) {
      strides[i] = product;
      product *= sizes[i];
    }
    return strides;
  }

  static int offset(Indices indices) {
    int result = 0;
    auto strides = getStrides();
    for (int i = 0; i < order; ++i) {
      result += indices[i] * strides[i];
    }
    return result;
  }

private:
  Reg regs[flatSize];
};

template <Simd simd, int... sizes>
Vector<simd, sizes...> add(Vector<simd, sizes...> x, Vector<simd, sizes...> y) {
  using V = Vector<simd, sizes...>;
  using ST = SimdTraits<simd>;
  V result;
  for (int i = 0; i < V::flatSize; ++i) {
    result.regs[i] = ST::add(x.regs[i], y.regs[i]);
  }
  return result;
}

template <Simd simd, int... sizes>
Vector<simd, sizes...> elem_mul(Vector<simd, sizes...> x,
                                Vector<simd, sizes...> y) {
  using V = Vector<simd, sizes...>;
  using ST = SimdTraits<simd>;
  V result;
  for (int i = 0; i < V::flatSize; ++i) {
    result.regs[i] = ST::mul(x.regs[i], y.regs[i]);
  }
  return result;
}

#endif // HAY_Vector_H_
