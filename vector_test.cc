// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "simd.h"
#include "simd_base.h"
#include "testlib.h"
#include "vector.h"

template <Simd s> struct TestVectorInt64xN {

  template <typename EType, int... sizesPack> static void Run() {
    using E = Int64xN<s>;
    using V = Vector<E, sizesPack...>;
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
  }
  static void Run() {
    using E = Int64xN<s>;
    Run<E, 4, 3, 2>();
    Run<E>();
    Run<E, 1>();
    Run<E, 5>();
    Run<E, 2, 3>();
    Run<E, 4, 3, 2>();
  }
};

int main() { TEST(TestVectorInt64xN); }