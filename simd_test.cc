// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "simd.h"
#include "testlib.h"

#include <cstring>

template <Simd s> struct TestInt64xNLoadStore {
  static void Run() {
    const int buf_elems = 2 * Int64xN<s>::elem_count;
    int64_t buf[buf_elems];
    for (int i = 0; i < buf_elems; ++i) {
      buf[i] = i;
    }
    Int64xN<s> x = Int64xN<s>::load(buf);
    Int64xN<s> y = Int64xN<s>::load(buf + Int64xN<s>::elem_count);
    CHECK_NE(x, y);
    for (int b = 0; b < Int64xN<s>::elem_count; ++b) {
      CHECK_EQ(extract(x, b), int64_t{b});
    }
    store(buf, y);
    CHECK_EQ(y, Int64xN<s>::load(buf));
    CHECK(!memcmp(buf, buf + Int64xN<s>::elem_count, sizeof(Int64xN<s>)));
  }
};

template <Simd s> struct TestUint1xNLoadStore {
  static void Run() {
    const int buf_bytes = 2 * sizeof(Uint1xN<s>);
    uint8_t buf[buf_bytes];
    for (int i = 0; i < buf_bytes; ++i) {
      buf[i] = i;
    }
    Uint1xN<s> x = Uint1xN<s>::load(buf);
    Uint1xN<s> y = Uint1xN<s>::load(buf + sizeof(Uint1xN<s>));
    CHECK(!(x == y));
    for (int b = 0; b < Uint1xN<s>::elem_count; ++b) {
      CHECK_EQ(extract(x, b),
               uint8_t{static_cast<uint8_t>((buf[b / 8] >> (b % 8)) & 1)});
    }
    store(buf, y);
    CHECK_EQ(y, Uint1xN<s>::load(buf));
    CHECK(!memcmp(buf, buf + sizeof(Uint1xN<s>), sizeof(Uint1xN<s>)));
  }
};

template <Simd s> struct TestInt64xNArithmetic {
  static void Run() {
    CHECK_EQ(Int64xN<s>::zero(), Int64xN<s>::zero());
    CHECK_EQ(Int64xN<s>::cst(1), Int64xN<s>::cst(1));
    CHECK_NE(Int64xN<s>::zero(), Int64xN<s>::cst(1));
    CHECK_EQ(add(Int64xN<s>::cst(-123), Int64xN<s>::cst(-456)),
             Int64xN<s>::cst(-579));
    CHECK_EQ(sub(Int64xN<s>::cst(-123), Int64xN<s>::cst(-456)),
             Int64xN<s>::cst(333));
    CHECK_EQ(max(Int64xN<s>::cst(-123), Int64xN<s>::cst(-456)),
             Int64xN<s>::cst(-123));
    CHECK_EQ(min(Int64xN<s>::cst(-123), Int64xN<s>::cst(-456)),
             Int64xN<s>::cst(-456));
    std::minstd_rand0 engine;
    Int64xN<s> x = getRandomInt64xN<s>(engine);
    Int64xN<s> y = getRandomInt64xN<s>(engine);
    Int64xN<s> z = getRandomInt64xN<s>(engine);
    CHECK_EQ(x, x);
    CHECK_EQ(add(x, Int64xN<s>::zero()), x);
    CHECK_NE(add(x, Int64xN<s>::cst(1)), x);
    CHECK_EQ(add(add(x, Int64xN<s>::cst(1)), Int64xN<s>::cst(-1)), x);
    CHECK_EQ(add(x, Int64xN<s>::cst(-1)), sub(x, Int64xN<s>::cst(1)));
    CHECK_EQ(add(x, max(y, z)), max(add(x, y), add(x, z)));
    CHECK_EQ(add(x, min(y, z)), min(add(x, y), add(x, z)));
    int64_t r = 0;
    for (int i = 0; i < Int64xN<s>::elem_count; ++i) {
      r += extract(x, i);
    }
    CHECK_EQ(r, reduce_add(x));
    CHECK_EQ(reduce_add(add(x, y)), reduce_add(x) + reduce_add(y));
    CHECK_EQ(reduce_add(Int64xN<s>::wave()),
             Int64xN<s>::elem_count * (Int64xN<s>::elem_count - 1) / 2);
  }
};

template <Simd s> struct TestUint1xNArithmetic {
  static void Run() {
    CHECK_EQ(Uint1xN<s>::zero(), Uint1xN<s>::zero());
    CHECK_EQ(Uint1xN<s>::ones(), Uint1xN<s>::ones());
    CHECK_NE(Uint1xN<s>::zero(), Uint1xN<s>::ones());
    std::minstd_rand0 engine;
    Uint1xN<s> x = getRandomUint1xN<s>(engine);
    Uint1xN<s> y = getRandomUint1xN<s>(engine);
    Uint1xN<s> z = getRandomUint1xN<s>(engine);
    CHECK_EQ(x, x);
    CHECK_NE(add(x, Uint1xN<s>::ones()), x);
    CHECK_EQ(add(x, x), Uint1xN<s>::zero());
    CHECK_EQ(add(x, Uint1xN<s>::zero()), x);
    CHECK_EQ(add(x, y), add(y, x));
    CHECK_EQ(add(add(x, y), z), add(x, add(y, z)));
    CHECK_EQ(mul(x, x), x);
    CHECK_EQ(mul(x, Uint1xN<s>::ones()), x);
    CHECK_EQ(mul(x, y), mul(y, x));
    CHECK_EQ(mul(mul(x, y), z), mul(x, mul(y, z)));
    CHECK_EQ(mul(x, add(y, z)), add(mul(x, y), mul(x, z)));
    CHECK_EQ(madd(x, y, z), add(x, mul(y, z)));
  }
};

template <Simd s> struct TestUint1xNBitcounts {
  static void Run() {
    const int bits = Uint1xN<s>::elem_count;
    CHECK_EQ(bits, static_cast<int>(8 * sizeof(Uint1xN<s>)));
    CHECK_EQ(reduce_add(lzcount64(Uint1xN<s>::zero())), bits);
    CHECK_EQ(reduce_add(popcount64(Uint1xN<s>::zero())), 0);
    CHECK_EQ(reduce_add(lzcount64(Uint1xN<s>::ones())), 0);
    CHECK_EQ(reduce_add(popcount64(Uint1xN<s>::ones())), bits);
    for (int i = 0; (1 << i) < bits; ++i) {
      uint8_t buf[sizeof(Uint1xN<s>)] = {0};
      buf[i / 8] = 1u << (i % 8);
      Uint1xN<s> x = Uint1xN<s>::load(buf);
      CHECK_EQ(extract(x, i), 1);
      CHECK_EQ(reduce_add(popcount64(x)), 1);
      CHECK_EQ(extract(lzcount64(x), i / 64), 63 - (i % 64));
    }
    for (int i = 0; (1 << i) < bits; ++i) {
      CHECK_EQ(reduce_add(popcount64(Uint1xN<s>::wave(i))), bits / 2);
      for (int j = 0; (1 << j) < bits; ++j) {
        if (i == j) {
          continue;
        }
        CHECK_EQ(reduce_add(
                     popcount64(add(Uint1xN<s>::wave(i), Uint1xN<s>::wave(j)))),
                 bits / 2);
        CHECK_EQ(reduce_add(
                     popcount64(mul(Uint1xN<s>::wave(i), Uint1xN<s>::wave(j)))),
                 bits / 4);
      }
    }
  }
};

template <Simd s> struct TestUint1xNWave {
  static void Run() {
    const int bytes = sizeof(Uint1xN<s>);
    const int bits = Uint1xN<s>::elem_count;
    CHECK_EQ(bits, 8 * bytes);
    constexpr int numslices = 1 + __builtin_ctz(bits);
    uint8_t slices[bytes * numslices];
    for (int i = 0; i < numslices; ++i) {
      store(slices + i * bytes, Uint1xN<s>::wave(i));
    }
    for (int b = 0; b < bytes; ++b) {
      uint64_t sequence_value = 0;
      for (int i = 0; i < numslices; ++i) {
        uint8_t slice_byte = slices[i * bytes + b / 8];
        uint8_t slice_bit = (slice_byte & (1 << (b % 8))) ? 1 : 0;
        sequence_value |= slice_bit << i;
      }
      CHECK_EQ(sequence_value, static_cast<uint64_t>(b));
    }
  }
};

int main() {
  TEST(TestInt64xNLoadStore);
  TEST(TestUint1xNLoadStore);
  TEST(TestInt64xNArithmetic);
  TEST(TestUint1xNArithmetic);
  TEST(TestUint1xNBitcounts);
  TEST(TestUint1xNWave);
}
