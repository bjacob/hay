// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "simd.h"
#include "testlib.h"

#include <cstring>

struct TestInt64xNLoadStore {
  static void Run() {
    const int buf_elems = 2 * Int64xN::elem_count;
    int64_t buf[buf_elems];
    for (int i = 0; i < buf_elems; ++i) {
      buf[i] = i;
    }
    Int64xN x = Int64xN::load(buf);
    Int64xN y = Int64xN::load(buf + Int64xN::elem_count);
    CHECK_NE(x, y);
    for (int b = 0; b < Int64xN::elem_count; ++b) {
      CHECK_EQ(extract(x, b), int64_t{b});
    }
    store(buf, y);
    CHECK_EQ(y, Int64xN::load(buf));
    CHECK(!memcmp(buf, buf + Int64xN::elem_count, sizeof(Int64xN)));
  }
};

struct TestUint1xNLoadStore {
  static void Run() {
    const int buf_bytes = 2 * sizeof(Uint1xN);
    uint8_t buf[buf_bytes];
    for (int i = 0; i < buf_bytes; ++i) {
      buf[i] = i;
    }
    Uint1xN x = Uint1xN::load(buf);
    Uint1xN y = Uint1xN::load(buf + sizeof(Uint1xN));
    CHECK(!(x == y));
    for (int b = 0; b < Uint1xN::elem_count; ++b) {
      CHECK_EQ(extract(x, b),
               uint8_t{static_cast<uint8_t>((buf[b / 8] >> (b % 8)) & 1)});
    }
    store(buf, y);
    CHECK_EQ(y, Uint1xN::load(buf));
    CHECK(!memcmp(buf, buf + sizeof(Uint1xN), sizeof(Uint1xN)));
  }
};

struct TestInt64xNArithmetic {
  static void Run() {
    CHECK_EQ(Int64xN::cst(0), Int64xN::cst(0));
    CHECK_EQ(Int64xN::cst(1), Int64xN::cst(1));
    CHECK_NE(Int64xN::cst(0), Int64xN::cst(1));
    CHECK_EQ(add(Int64xN::cst(-123), Int64xN::cst(-456)), Int64xN::cst(-579));
    CHECK_EQ(sub(Int64xN::cst(-123), Int64xN::cst(-456)), Int64xN::cst(333));
    CHECK_EQ(max(Int64xN::cst(-123), Int64xN::cst(-456)), Int64xN::cst(-123));
    CHECK_EQ(min(Int64xN::cst(-123), Int64xN::cst(-456)), Int64xN::cst(-456));
    std::minstd_rand0 engine;
    Int64xN x = getRandom<Int64xN>(engine);
    Int64xN y = getRandom<Int64xN>(engine);
    Int64xN z = getRandom<Int64xN>(engine);
    CHECK_EQ(x, x);
    CHECK_EQ(add(x, Int64xN::cst(0)), x);
    CHECK_NE(add(x, Int64xN::cst(1)), x);
    CHECK_EQ(add(add(x, Int64xN::cst(1)), Int64xN::cst(-1)), x);
    CHECK_EQ(add(x, Int64xN::cst(-1)), sub(x, Int64xN::cst(1)));
    CHECK_EQ(add(x, max(y, z)), max(add(x, y), add(x, z)));
    CHECK_EQ(add(x, min(y, z)), min(add(x, y), add(x, z)));
    int64_t r = 0;
    for (int i = 0; i < Int64xN::elem_count; ++i) {
      r += extract(x, i);
    }
    CHECK_EQ(r, reduce_add(x));
    CHECK_EQ(reduce_add(add(x, y)), reduce_add(x) + reduce_add(y));
  }
};

struct TestUint1xNArithmetic {
  static void Run() {
    CHECK_EQ(Uint1xN::cst(0), Uint1xN::cst(0));
    CHECK_EQ(Uint1xN::cst(1), Uint1xN::cst(1));
    CHECK_NE(Uint1xN::cst(0), Uint1xN::cst(1));
    std::minstd_rand0 engine;
    Uint1xN x = getRandom<Uint1xN>(engine);
    Uint1xN y = getRandom<Uint1xN>(engine);
    Uint1xN z = getRandom<Uint1xN>(engine);
    CHECK_EQ(x, x);
    CHECK_NE(add(x, Uint1xN::cst(1)), x);
    CHECK_EQ(add(x, x), Uint1xN::cst(0));
    CHECK_EQ(add(x, Uint1xN::cst(0)), x);
    CHECK_EQ(add(x, y), add(y, x));
    CHECK_EQ(add(add(x, y), z), add(x, add(y, z)));
    CHECK_EQ(mul(x, x), x);
    CHECK_EQ(mul(x, Uint1xN::cst(1)), x);
    CHECK_EQ(mul(x, y), mul(y, x));
    CHECK_EQ(mul(mul(x, y), z), mul(x, mul(y, z)));
    CHECK_EQ(mul(x, add(y, z)), add(mul(x, y), mul(x, z)));
    CHECK_EQ(madd(x, y, z), add(x, mul(y, z)));
  }
};

struct TestUint1xNBitcounts {
  static void Run() {
    const int bits = Uint1xN::elem_count;
    CHECK_EQ(bits, static_cast<int>(8 * sizeof(Uint1xN)));
    CHECK_EQ(reduce_add(popcount(Uint1xN::cst(0))), 0);
    CHECK_EQ(reduce_add(popcount(Uint1xN::cst(1))), bits);
    for (int i = 0; (1 << i) < bits; ++i) {
      uint8_t buf[sizeof(Uint1xN)] = {0};
      buf[i / 8] = 1u << (i % 8);
      Uint1xN x = Uint1xN::load(buf);
      CHECK_EQ(extract(x, i), 1);
      CHECK_EQ(reduce_add(popcount(x)), 1);
    }
    for (int i = 0; (1 << i) < bits; ++i) {
      CHECK_EQ(reduce_add(popcount(Uint1xN::seq(i))), bits / 2);
      for (int j = 0; (1 << j) < bits; ++j) {
        if (i == j) {
          continue;
        }
        CHECK_EQ(reduce_add(popcount(add(Uint1xN::seq(i), Uint1xN::seq(j)))),
                 bits / 2);
        CHECK_EQ(reduce_add(popcount(mul(Uint1xN::seq(i), Uint1xN::seq(j)))),
                 bits / 4);
      }
    }
  }
};

struct TestUint1xNSeq {
  static void Run() {
    const int bytes = sizeof(Uint1xN);
    const int bits = Uint1xN::elem_count;
    CHECK_EQ(bits, 8 * bytes);
    constexpr int numslices = 1 + __builtin_ctz(bits);
    uint8_t slices[bytes * numslices];
    for (int i = 0; i < numslices; ++i) {
      store(slices + i * bytes, Uint1xN::seq(i));
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

struct TestInt64xNFormat {
  static void Run() {
    static constexpr int elems = Int64xN::elem_count;
    int64_t buf[elems];
    for (int i = 0; i < elems; ++i) {
      buf[i] = 100000000000ll + i;
    }
    std::string actual = fmt::format("{}", Int64xN::load(buf));
    if (elems == 1) {
      CHECK_EQ(actual, "{100000000000}");
    } else if (elems == 2) {
      CHECK_EQ(actual, "{100000000000, 100000000001}");
    } else {
      CHECK(actual.starts_with(
          "{100000000000, 100000000001, 100000000002, 100000000003"));
    }
  }
};

struct TestUint1xNFormat {
  static void Run() {
    static constexpr int bytes = sizeof(Uint1xN);
    uint8_t buf[bytes];
    for (int i = 0; i < bytes; ++i) {
      buf[i] = i;
    }
    std::string actual = fmt::format("{}", Uint1xN::load(buf));
    if (bytes == 4) {
      CHECK_EQ(actual, "{0x03020100}");
    } else if (bytes == 8) {
      CHECK_EQ(actual, "{0x03020100, 0x07060504}");
    } else if (bytes == 16) {
      CHECK_EQ(actual, "{0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c}");
    } else {
      CHECK(
          actual.starts_with("{0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, "
                             "0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c"));
    }
  }
};

int main() {
  TEST(TestInt64xNLoadStore);
  TEST(TestUint1xNLoadStore);
  TEST(TestInt64xNArithmetic);
  TEST(TestUint1xNArithmetic);
  TEST(TestUint1xNBitcounts);
  TEST(TestUint1xNSeq);
  TEST(TestInt64xNFormat);
  TEST(TestUint1xNFormat);
}
