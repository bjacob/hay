// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "simd.h"
#include "simd_base.h"
#include "simd_x86.h"
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
      CHECK(extract(x, b).val == b);
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
      CHECK(extract(x, b).val == ((buf[b / 8] >> (b % 8)) & 1));
    }
    store(buf, y);
    CHECK(y == Uint1xN<s>::load(buf));
    CHECK(!memcmp(buf, buf + sizeof(Uint1xN<s>), sizeof(Uint1xN<s>)));
  }
};

template <Simd s> struct TestUint1xNArithmetic {
  static void Run() {
    CHECK(Uint1xN<s>::zero() == Uint1xN<s>::zero());
    CHECK(Uint1xN<s>::ones() == Uint1xN<s>::ones());
    CHECK(!(Uint1xN<s>::zero() == Uint1xN<s>::ones()));
    std::minstd_rand0 engine;
    Uint1xN<s> x = getRandomUint1xN<s>(engine);
    Uint1xN<s> y = getRandomUint1xN<s>(engine);
    Uint1xN<s> z = getRandomUint1xN<s>(engine);
    CHECK(x == x);
    CHECK(!(add(x, Uint1xN<s>::ones()) == x));
    CHECK((add(x, x) == Uint1xN<s>::zero()));
    CHECK((add(x, Uint1xN<s>::zero()) == x));
    CHECK((add(x, y) == add(y, x)));
    CHECK((add(add(x, y), z) == add(x, add(y, z))));
    CHECK((mul(x, x) == x));
    CHECK((mul(x, Uint1xN<s>::ones()) == x));
    CHECK((mul(x, y) == mul(y, x)));
    CHECK((mul(mul(x, y), z) == mul(x, mul(y, z))));
    CHECK((mul(x, add(y, z)) == add(mul(x, y), mul(x, z))));
    CHECK((madd(x, y, z) == add(x, mul(y, z))));
  }
};

template <Simd s> struct TestUint1xNBitcounts {
  static void Run() {
    const int bits = Uint1xN<s>::elem_count;
    CHECK(bits == 8 * sizeof(Uint1xN<s>));
    CHECK(reduce_add(lzcount64(Uint1xN<s>::zero())).val == bits);
    CHECK(reduce_add(popcount64(Uint1xN<s>::zero())).val == 0);
    CHECK(reduce_add(lzcount64(Uint1xN<s>::ones())).val == 0);
    CHECK(reduce_add(popcount64(Uint1xN<s>::ones())).val == bits);
    for (int i = 0; (1 << i) < bits; ++i) {
      uint8_t buf[sizeof(Uint1xN<s>)] = {0};
      buf[i / 8] = 1u << (i % 8);
      Uint1xN<s> x = Uint1xN<s>::load(buf);
      CHECK(extract(x, i).val == 1);
      CHECK(reduce_add(popcount64(x)).val == 1);
      CHECK(extract(lzcount64(x), i / 64).val == 63 - (i % 64));
    }
    for (int i = 0; (1 << i) < bits; ++i) {
      CHECK(reduce_add(popcount64(Uint1xN<s>::wave(i))).val == bits / 2);
      for (int j = 0; (1 << j) < bits; ++j) {
        if (i == j) {
          continue;
        }
        CHECK(reduce_add(
                  popcount64(add(Uint1xN<s>::wave(i), Uint1xN<s>::wave(j))))
                  .val == bits / 2);
        CHECK(reduce_add(
                  popcount64(mul(Uint1xN<s>::wave(i), Uint1xN<s>::wave(j))))
                  .val == bits / 4);
      }
    }
  }
};

template <Simd s> struct TestUint1xNWave {
  static void Run() {
    const int bytes = sizeof(Uint1xN<s>);
    const int bits = Uint1xN<s>::elem_count;
    CHECK(bits == 8 * bytes);
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
      CHECK(sequence_value == static_cast<uint64_t>(b));
    }
  }
};

int main() {
  TEST(TestInt64xNLoadStore);
  TEST(TestUint1xNLoadStore);
  TEST(TestUint1xNArithmetic);
  TEST(TestUint1xNBitcounts);
  TEST(TestUint1xNWave);
}
