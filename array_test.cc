// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "array.h"
#include "testlib.h"

using namespace hay;

struct TestUint1ArrayLayout {
  using E = uint1_vec_t;
  static void Run1() {
    using V = Array<E, {3, 5, 4, 2}>;
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
    CHECK_EQ(V::unflatten_index(0), (Indices{0, 0, 0, 0}));
    CHECK_EQ(V::unflatten_index(1), (Indices{0, 0, 0, 1}));
    CHECK_EQ(V::unflatten_index(2), (Indices{0, 0, 1, 0}));
    CHECK_EQ(V::unflatten_index(8), (Indices{0, 1, 0, 0}));
    CHECK_EQ(V::unflatten_index(40), (Indices{1, 0, 0, 0}));
    CHECK_EQ(V::unflatten_index(95), (Indices{2, 1, 3, 1}));
  }
  static void Run() { Run1(); }
};

struct TestInt64ArrayLoadStore {
  using E = int64_vec_t;
  template <Indices sizes> static void Run() {
    using V = Array<E, sizes>;
    E buf[2 * V::flatSize];
    std::minstd_rand0 engine;
    for (int i = 0; i < 2 * V::flatSize; ++i) {
      buf[i] = get_random<int64_vec_t>(engine);
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

struct TestUint1ArrayLoadStore {
  using E = uint1_vec_t;
  template <Indices sizes> static void Run() {
    using V = Array<E, sizes>;
    E buf[2 * V::flatSize];
    std::minstd_rand0 engine;
    for (int i = 0; i < 2 * V::flatSize; ++i) {
      buf[i] = get_random<uint1_vec_t>(engine);
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

struct TestInt64ArrayArithmetic {
  using E = int64_vec_t;
  template <Indices sizes> static void Run() {
    using V = Array<E, sizes>;
    std::minstd_rand0 engine;
    V x = get_random<V>(engine);
    V y = get_random<V>(engine);
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
    using VS = Array<S, sizes>;
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

struct TestUint1ArrayArithmetic {
  using E = uint1_vec_t;
  template <Indices sizes> static void Run() {
    using V = Array<E, sizes>;
    std::minstd_rand0 engine;
    V x = get_random<V>(engine);
    V y = get_random<V>(engine);
    V z = get_random<V>(engine);
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

struct TestUint1ArrayRow {
  using E = uint1_vec_t;
  template <Indices sizes> static void Run() {
    using V = Array<E, sizes>;
    std::minstd_rand0 engine;
    V x = get_random<V>(engine);
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

struct TestUint1ArrayReshape {
  using E = uint1_vec_t;
  static void Run1() {
    using V = Array<E, {2}>;
    std::minstd_rand0 engine;
    V x = get_random<V>(engine);
    V y = reshape<{2}>(x);
    for (int i = 0; i < 1; i++) {
      CHECK_EQ(y.elems[i], x.elems[i]);
    }
  }
  static void Run2() {
    using V23 = Array<E, {2, 3}>;
    using V32 = Array<E, {3, 2}>;
    std::minstd_rand0 engine;
    V23 x = get_random<V23>(engine);
    V32 y = reshape<{3, 2}>(x);
    for (int i = 0; i < 6; i++) {
      CHECK_EQ(y.elems[i], x.elems[i]);
    }
  }
  static void Run3() {
    using V234 = Array<E, {2, 3, 4}>;
    using V432 = Array<E, {4, 3, 2}>;
    std::minstd_rand0 engine;
    V234 x = get_random<V234>(engine);
    V432 y = reshape<{4, 3, 2}>(x);
    for (int i = 0; i < 24; i++) {
      CHECK_EQ(y.elems[i], x.elems[i]);
    }
  }
  static void Run4() {
    using V234 = Array<E, {15}>;
    using V432 = Array<E, {3, 5}>;
    std::minstd_rand0 engine;
    V234 x = get_random<V234>(engine);
    V432 y = reshape<{3, 5}>(x);
    for (int i = 0; i < 15; i++) {
      CHECK_EQ(y.elems[i], x.elems[i]);
    }
  }
  static void Run5() {
    using V234 = Array<E, {1, 2, 3}>;
    using V432 = Array<E, {6}>;
    std::minstd_rand0 engine;
    V234 x = get_random<V234>(engine);
    V432 y = reshape<{6}>(x);
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

struct TestUint1ArrayTranspose {
  using E = uint1_vec_t;
  static void Run1() {
    using V2345 = Array<E, {2, 3, 4, 5}>;
    using V4352 = Array<E, {4, 3, 5, 2}>;
    std::minstd_rand0 engine;
    V2345 x = get_random<V2345>(engine);
    // Check that this test won't be vacuous.
    CHECK_NE(x.elems[0], E::cst(0));
    CHECK_NE(x.elems[1], E::cst(0));
    V4352 y = transpose<{2, 1, 3, 0}>(x);
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

struct TestUint1ArraySeq {
  template <Indices sizes> static void Run() {
    using E = uint1_vec_t;
    using V = Array<E, sizes>;
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

struct TestArrayUint1Format {
  static void Run() {
    using E = uint1_vec_t;
    using V = Array<E, {3, 3}>;
    std::string actual = fmt::format("{}", extract(V::seq(0), 0b000011111));
    CHECK_EQ(actual, "[[1, 1, 1], [1, 1, 0], [0, 0, 0]]");
  }
};

struct TestUint1ArrayBits {
  static void Run() {
    using E = uint1_vec_t;
    using V = Array<E, {4, 4}>;
    V x = V::seq(0);
    for (int i = 0; i < V::flatSize; ++i) {
      int expected = std::min(uint1_vec_t::elem_count, 64) / 2;
      CHECK_EQ(extract(popcount(x), 0).elems[i],
               ((1 << i) <= expected) ? expected : 0);
    }
  }
};

struct TestUint1ArrayContractUnary {
  static void Run() {
    using E = uint1_vec_t;
    using V33 = Array<E, {3, 3}>;
    std::minstd_rand0 engine;
    V33 x = get_random<V33>(engine);
    Array<E, {}> x_contracted = contract<0, 1>(x);
    CHECK_EQ(x_contracted.elems[0],
             (add(x.elems[0], add(x.elems[4], x.elems[8]))));
    CHECK_EQ(x_contracted.elems[0], trace(x));
    using V32323 = Array<E, {3, 2, 3, 2, 3}>;
    using V333 = Array<E, {3, 3, 3}>;
    V32323 y = get_random<V32323>(engine);
    V333 y_contracted = contract<1, 3>(y);
    CHECK_EQ(y_contracted.elems[0], (add(y.elems[0], y.elems[21])));
    CHECK_EQ(y_contracted.elems[1], (add(y.elems[1], y.elems[22])));
    CHECK_EQ(y_contracted.elems[2], (add(y.elems[2], y.elems[23])));
    CHECK_EQ(y_contracted.elems[3], (add(y.elems[6], y.elems[27])));
  }
};

struct TestUint1ArrayContractBinary {
  static void Run() {
    using E = uint1_vec_t;
    using V32 = Array<E, {3, 2}>;
    using V24 = Array<E, {2, 4}>;
    std::minstd_rand0 engine;
    V32 x = get_random<V32>(engine);
    V24 y = get_random<V24>(engine);
    Array<E, {3, 4}> contract_result = contract<1, 0>(x, y);
    Array<E, {3, 4}> matmul_result = matmul(x, y);
    CHECK_EQ(contract_result, matmul_result);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        CHECK_EQ(matmul_result.elems[4 * i + j],
                 add(mul(x.elems[2 * i + 0], y.elems[j]),
                     mul(x.elems[2 * i + 1], y.elems[4 + j])));
      }
    }
  }
};

int main() {
  TEST(TestUint1ArrayLayout);
  TEST(TestInt64ArrayLoadStore);
  TEST(TestUint1ArrayLoadStore);
  TEST(TestInt64ArrayArithmetic);
  TEST(TestUint1ArrayArithmetic);
  TEST(TestUint1ArrayRow);
  TEST(TestUint1ArrayReshape);
  TEST(TestUint1ArrayTranspose);
  TEST(TestUint1ArraySeq);
  TEST(TestArrayUint1Format);
  TEST(TestUint1ArrayBits);
  TEST(TestUint1ArrayContractUnary);
  TEST(TestUint1ArrayContractBinary);
}