// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "simd.h"
#include "simd_base.h"
#include "testlib.h"
#include "vector.h"

template <Simd s> struct TestVectorInt64xNLoadStore {
  using E = Int64xN<s>;
  template <int... sizes> static void Run() {
    using V = Vector<E, sizes...>;
    E buf[2 * V::flatSize];
    std::minstd_rand0 engine;
    for (int i = 0; i < 2 * V::flatSize; ++i) {
      buf[i] = getRandomInt64xN<s>(engine);
    }
    E buf_add[V::flatSize];
    for (int i = 0; i < V::flatSize; ++i) {
      buf_add[i] = add(buf[i], buf[i + V::flatSize]);
    }
    V x = V::load(buf);
    V y = V::load(buf + V::flatSize);
    CHECK_EQ(add(x, y), V::load(buf_add));
    store(buf, y);
    CHECK_EQ(y, V::load(buf));
  }
  static void Run() {
    Run<>();
    Run<1>();
    Run<5>();
    Run<2, 3>();
    Run<4, 3, 2>();
  }
};

template <Simd s> struct TestVectorUint1xNSeq {
  template <int... sizes> static void Run() {
    using E = Uint1xN<s>;
    using V = Vector<E, sizes...>;
    for (int k = 0; k < 4; ++k) {
      V x = V::seq(k);
      for (int i = 0; i < E::elem_count; ++i) {
        auto e = extract(x, i);
        for (int j = 0; j < e.flatSize; ++j) {
          CHECK_EQ(e.elems[j], ((i + k * E::elem_count) >> j) & 1);
        }
      }
    }
  }
  static void Run() {
    Run<>();
    Run<3>();
    Run<3, 3>();
    Run<2, 2, 2>();
    Run<1, 3, 2, 1>();
  }
};

int main() {
  TEST(TestVectorInt64xNLoadStore);
  TEST(TestVectorUint1xNSeq);
}