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
      buf[i] = getRandom<Int64xN<s>>(engine);
    }
    V y = V::load(buf + V::flatSize);
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

template <Simd s> struct TestVectorUint1xNLoadStore {
  using E = Uint1xN<s>;
  template <int... sizes> static void Run() {
    using V = Vector<E, sizes...>;
    E buf[2 * V::flatSize];
    std::minstd_rand0 engine;
    for (int i = 0; i < 2 * V::flatSize; ++i) {
      buf[i] = getRandom<Uint1xN<s>>(engine);
    }
    V y = V::load(buf + V::flatSize);
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

template <Simd s> struct TestVectorInt64xNArithmetic {
  using E = Int64xN<s>;
  template <int... sizes> static void Run() {
    using V = Vector<E, sizes...>;
    std::minstd_rand0 engine;
    V x = getRandom<V>(engine);
    V y = getRandom<V>(engine);
    V c = V::cst(123);
    for (int i = 0; i < V::flatSize; ++i) {
      CHECK_EQ(max(x, y).elems[i], max(x.elems[i], y.elems[i]));
      CHECK_EQ(min(x, y).elems[i], min(x.elems[i], y.elems[i]));
      CHECK_EQ(add(x, y).elems[i], add(x.elems[i], y.elems[i]));
      CHECK_EQ(sub(x, y).elems[i], sub(x.elems[i], y.elems[i]));
      for (int j = 0; j < E::elem_count; ++j) {
        CHECK_EQ(extract(c.elems[i], j), 123);
      }
    }
    using S = V::ScalarType;
    using VS = Vector<S, sizes...>;
    VS rx = reduce_add(x);
    for (int i = 0; i < V::flatSize; ++i) {
      S sum = 0;
      for (int j = 0; j < E::elem_count; ++j) {
        sum += extract(x.elems[i], j);
      }
      CHECK_EQ(rx.elems[i], sum);
    }
  }
  static void Run() {
    Run<>();
    Run<1>();
    Run<5>();
    Run<2, 3>();
    Run<4, 3, 2>();
  }
};

template <Simd s> struct TestVectorUint1xNArithmetic {
  using E = Uint1xN<s>;
  template <int... sizes> static void Run() {
    using V = Vector<E, sizes...>;
    std::minstd_rand0 engine;
    V x = getRandom<V>(engine);
    V y = getRandom<V>(engine);
    V z = getRandom<V>(engine);
    V c = V::cst(1);
    for (int i = 0; i < V::flatSize; ++i) {
      CHECK_EQ(add(x, y).elems[i], add(x.elems[i], y.elems[i]));
      CHECK_EQ(mul(x, y).elems[i], mul(x.elems[i], y.elems[i]));
      CHECK_EQ(madd(x, y, z).elems[i],
               madd(x.elems[i], y.elems[i], z.elems[i]));
      for (int j = 0; j < E::elem_count; ++j) {
        CHECK_EQ(extract(c.elems[i], j), 1);
      }
    }
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

template <Simd s> struct TestVectorUint1Format {
  static void Run() {
    using E = Uint1xN<s>;
    using V = Vector<E, 3, 3>;
    std::string actual = std::format("{}", extract(V::seq(0), 0b000101111));
    CHECK_EQ(actual, "[[1, 1, 1], [1, 0, 1], [0, 0, 0]]");
  }
};

template <Simd s> struct TestVectorUint1xNBits {
  static void Run() {
    using E = Uint1xN<s>;
    using V = Vector<E, 4, 4>;
    V x = V::seq(0);
    for (int i = 0; i < V::flatSize; ++i) {
      CHECK_EQ(extract(popcount64(x), 0).elems[i], i < 6 ? 32 : 0);
      CHECK_EQ(extract(lzcount64(x), 0).elems[i], i < 6 ? 0 : 64);
    }
  }
};

int main() {
  TEST(TestVectorInt64xNLoadStore);
  TEST(TestVectorUint1xNLoadStore);
  TEST(TestVectorInt64xNArithmetic);
  TEST(TestVectorUint1xNArithmetic);
  TEST(TestVectorUint1xNSeq);
  TEST(TestVectorUint1Format);
  TEST(TestVectorUint1xNBits);
}