// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "simd.h"
#include "simd_base.h"
#include "testlib.h"
#include "vector.h"

template <Simd s> struct TestVectorUint1xNLayout {
  using E = Uint1xN<s>;
  static void Run1() {
    using V = Vector<E, {3, 5, 4, 2}>;
    auto strides = V::get_strides();
    CHECK_EQ(static_cast<int>(strides.size()), 4);
    CHECK_EQ(strides[3], 1);
    CHECK_EQ(strides[2], 2);
    CHECK_EQ(strides[1], 8);
    CHECK_EQ(strides[0], 40);
    CHECK_EQ(V::flatten_indices({0, 0, 0, 0}), 0);
    CHECK_EQ(V::flatten_indices({0, 0, 0, 1}), 1);
    CHECK_EQ(V::flatten_indices({0, 0, 1, 0}), 2);
    CHECK_EQ(V::flatten_indices({0, 1, 0, 0}), 8);
    CHECK_EQ(V::flatten_indices({1, 0, 0, 0}), 40);
    CHECK_EQ(V::flatten_indices({2, 1, 3, 1}), 95);
    CHECK_EQ(V::unflatten_index(0), (Indices<4>{0, 0, 0, 0}));
    CHECK_EQ(V::unflatten_index(1), (Indices<4>{0, 0, 0, 1}));
    CHECK_EQ(V::unflatten_index(2), (Indices<4>{0, 0, 1, 0}));
    CHECK_EQ(V::unflatten_index(8), (Indices<4>{0, 1, 0, 0}));
    CHECK_EQ(V::unflatten_index(40), (Indices<4>{1, 0, 0, 0}));
    CHECK_EQ(V::unflatten_index(95), (Indices<4>{2, 1, 3, 1}));
  }
  static void Run() { Run1(); }
};

template <Simd s> struct TestVectorInt64xNLoadStore {
  using E = Int64xN<s>;
  template <Indices sizes> static void Run() {
    using V = Vector<E, sizes>;
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
    Run<{}>();
    Run<{1}>();
    Run<{5}>();
    Run<{2, 3}>();
    Run<{4, 3, 2}>();
  }
};

template <Simd s> struct TestVectorUint1xNLoadStore {
  using E = Uint1xN<s>;
  template <Indices sizes> static void Run() {
    using V = Vector<E, sizes>;
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
    Run<{}>();
    Run<{1}>();
    Run<{5}>();
    Run<{2, 3}>();
    Run<{4, 3, 2}>();
  }
};

template <Simd s> struct TestVectorInt64xNArithmetic {
  using E = Int64xN<s>;
  template <Indices sizes> static void Run() {
    using V = Vector<E, sizes>;
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
    using VS = Vector<S, sizes>;
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
    Run<{}>();
    Run<{1}>();
    Run<{5}>();
    Run<{2, 3}>();
    Run<{4, 3, 2}>();
  }
};

template <Simd s> struct TestVectorUint1xNArithmetic {
  using E = Uint1xN<s>;
  template <Indices sizes> static void Run() {
    using V = Vector<E, sizes>;
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
    Run<{}>();
    Run<{1}>();
    Run<{5}>();
    Run<{2, 3}>();
    Run<{4, 3, 2}>();
  }
};

template <Simd s> struct TestVectorUint1xNRow {
  using E = Uint1xN<s>;
  template <Indices sizes> static void Run() {
    using V = Vector<E, sizes>;
    std::minstd_rand0 engine;
    V x = getRandom<V>(engine);
    V z;
    insert_row(z, row(x, 0), 0);
    insert_row(z, row(x, 1), 1);
    insert_row(z, row(x, 2), 2);
    CHECK_EQ(x, z);
  }
  static void Run() {
    Run<{3}>();
    Run<{3, 2}>();
    Run<{3, 2, 4}>();
  }
};

template <Simd s> struct TestVectorUint1xNReshape {
  using E = Uint1xN<s>;
  static void Run1() {
    using V = Vector<E, Indices<1>{2}>;
    std::minstd_rand0 engine;
    V x = getRandom<V>(engine);
    V y = reshape<Indices<1>{2}>(x);
    for (int i = 0; i < 1; i++) {
      CHECK_EQ(y.elems[i], x.elems[i]);
    }
  }
  static void Run2() {
    using V23 = Vector<E, Indices<2>{2, 3}>;
    using V32 = Vector<E, Indices<2>{3, 2}>;
    std::minstd_rand0 engine;
    V23 x = getRandom<V23>(engine);
    V32 y = reshape<Indices<2>{3, 2}>(x);
    for (int i = 0; i < 6; i++) {
      CHECK_EQ(y.elems[i], x.elems[i]);
    }
  }
  static void Run3() {
    using V234 = Vector<E, Indices<3>{2, 3, 4}>;
    using V432 = Vector<E, Indices<3>{4, 3, 2}>;
    std::minstd_rand0 engine;
    V234 x = getRandom<V234>(engine);
    V432 y = reshape<Indices<3>{4, 3, 2}>(x);
    for (int i = 0; i < 24; i++) {
      CHECK_EQ(y.elems[i], x.elems[i]);
    }
  }
  static void Run4() {
    using V234 = Vector<E, Indices<1>{15}>;
    using V432 = Vector<E, Indices<2>{3, 5}>;
    std::minstd_rand0 engine;
    V234 x = getRandom<V234>(engine);
    V432 y = reshape<Indices<2>{3, 5}>(x);
    for (int i = 0; i < 15; i++) {
      CHECK_EQ(y.elems[i], x.elems[i]);
    }
  }
  static void Run5() {
    using V234 = Vector<E, Indices<3>{1, 2, 3}>;
    using V432 = Vector<E, Indices<1>{6}>;
    std::minstd_rand0 engine;
    V234 x = getRandom<V234>(engine);
    V432 y = reshape<Indices<1>{6}>(x);
    for (int i = 0; i < 6; i++) {
      CHECK_EQ(y.elems[i], x.elems[i]);
    }
  }
  static void Run() {
    Run1();
    Run2();
    Run3();
    Run4();
    Run5();
  }
};

template <Simd s> struct TestVectorUint1xNTranspose {
  using E = Uint1xN<s>;
  static void Run1() {
    using V2345 = Vector<E, Indices<4>{2, 3, 4, 5}>;
    using V4352 = Vector<E, Indices<4>{4, 3, 5, 2}>;
    std::minstd_rand0 engine;
    V2345 x = getRandom<V2345>(engine);
    // Check that this test won't be vacuous.
    CHECK_NE(x.elems[0], E::cst(0));
    CHECK_NE(x.elems[1], E::cst(0));
    V4352 y = transpose<Indices<4>{2, 1, 3, 0}>(x);
    CHECK_EQ(y.elems[0], x.elems[0]);
    CHECK_EQ(y.elems[1], x.elems[60]);
    CHECK_EQ(y.elems[2], x.elems[1]);
    CHECK_EQ(y.elems[3], x.elems[61]);
    CHECK_EQ(y.elems[8], x.elems[4]);
    CHECK_EQ(y.elems[9], x.elems[64]);
    CHECK_EQ(y.elems[10], x.elems[20]);
    CHECK_EQ(y.elems[19], x.elems[84]);
    CHECK_EQ(y.elems[29], x.elems[104]);
    CHECK_EQ(y.elems[30], x.elems[5]);
    CHECK_EQ(y.elems[31], x.elems[65]);
    CHECK_EQ(y.elems[39], x.elems[69]);
    CHECK_EQ(y.elems[119], x.elems[119]);
  }
  static void Run() { Run1(); }
};

template <Simd s> struct TestVectorUint1xNSeq {
  template <Indices sizes> static void Run() {
    using E = Uint1xN<s>;
    using V = Vector<E, sizes>;
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
    Run<{}>();
    Run<{3}>();
    Run<{3, 3}>();
    Run<{2, 2, 2}>();
    Run<{1, 3, 2, 1}>();
  }
};

template <Simd s> struct TestVectorUint1Format {
  static void Run() {
    using E = Uint1xN<s>;
    using V = Vector<E, Indices<2>{3, 3}>;
    std::string actual = std::format("{}", extract(V::seq(0), 0b000101111));
    CHECK_EQ(actual, "[[1, 1, 1], [1, 0, 1], [0, 0, 0]]");
  }
};

template <Simd s> struct TestVectorUint1xNBits {
  static void Run() {
    using E = Uint1xN<s>;
    using V = Vector<E, Indices<2>{4, 4}>;
    V x = V::seq(0);
    for (int i = 0; i < V::flatSize; ++i) {
      CHECK_EQ(extract(popcount64(x), 0).elems[i], i < 6 ? 32 : 0);
      CHECK_EQ(extract(lzcount64(x), 0).elems[i], i < 6 ? 0 : 64);
    }
  }
};

int main() {
  TEST(TestVectorUint1xNLayout);
  TEST(TestVectorInt64xNLoadStore);
  TEST(TestVectorUint1xNLoadStore);
  TEST(TestVectorInt64xNArithmetic);
  TEST(TestVectorUint1xNArithmetic);
  TEST(TestVectorUint1xNRow);
  TEST(TestVectorUint1xNReshape);
  TEST(TestVectorUint1xNTranspose);
  TEST(TestVectorUint1xNSeq);
  TEST(TestVectorUint1Format);
  TEST(TestVectorUint1xNBits);
}