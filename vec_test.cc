// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testlib.h"

#include <cstring>

using namespace hay;

struct TestInt64VecLoadStore {
  static void Run() {
    const int buf_elems = 2 * int64_vec_t::elem_count;
    int64_t buf[buf_elems];
    for (int i = 0; i < buf_elems; ++i) {
      buf[i] = i;
    }
    int64_vec_t x = int64_vec_t::load(buf);
    int64_vec_t y = int64_vec_t::load(buf + int64_vec_t::elem_count);
    CHECK_NE(x, y);
    for (int b = 0; b < int64_vec_t::elem_count; ++b) {
      CHECK_EQ(extract(x, b), int64_t{b});
    }
    store(buf, y);
    CHECK_EQ(y, int64_vec_t::load(buf));
    CHECK(!memcmp(buf, buf + int64_vec_t::elem_count, sizeof(int64_vec_t)));
  }
};

struct TestUint1VecLoadStore {
  static void Run() {
    const int buf_bytes = 2 * sizeof(uint1_vec_t);
    uint8_t buf[buf_bytes];
    for (int i = 0; i < buf_bytes; ++i) {
      buf[i] = i;
    }
    uint1_vec_t x = uint1_vec_t::load(buf);
    uint1_vec_t y = uint1_vec_t::load(buf + sizeof(uint1_vec_t));
    CHECK(!(x == y));
    for (int b = 0; b < uint1_vec_t::elem_count; ++b) {
      CHECK_EQ(extract(x, b),
               uint8_t{static_cast<uint8_t>((buf[b / 8] >> (b % 8)) & 1)});
    }
    store(buf, y);
    CHECK_EQ(y, uint1_vec_t::load(buf));
    CHECK(!memcmp(buf, buf + sizeof(uint1_vec_t), sizeof(uint1_vec_t)));
  }
};

struct TestInt64VecArithmetic {
  static void Run() {
    CHECK_EQ(int64_vec_t::cst(0), int64_vec_t::cst(0));
    CHECK_EQ(int64_vec_t::cst(1), int64_vec_t::cst(1));
    CHECK_NE(int64_vec_t::cst(0), int64_vec_t::cst(1));
    CHECK_EQ(add(int64_vec_t::cst(-123), int64_vec_t::cst(-456)),
             int64_vec_t::cst(-579));
    CHECK_EQ(sub(int64_vec_t::cst(-123), int64_vec_t::cst(-456)),
             int64_vec_t::cst(333));
    CHECK_EQ(max(int64_vec_t::cst(-123), int64_vec_t::cst(-456)),
             int64_vec_t::cst(-123));
    CHECK_EQ(min(int64_vec_t::cst(-123), int64_vec_t::cst(-456)),
             int64_vec_t::cst(-456));
    std::minstd_rand0 engine;
    int64_vec_t x = get_random<int64_vec_t>(engine);
    int64_vec_t y = get_random<int64_vec_t>(engine);
    int64_vec_t z = get_random<int64_vec_t>(engine);
    CHECK_EQ(x, x);
    CHECK_EQ(add(x, int64_vec_t::cst(0)), x);
    CHECK_NE(add(x, int64_vec_t::cst(1)), x);
    CHECK_EQ(add(add(x, int64_vec_t::cst(1)), int64_vec_t::cst(-1)), x);
    CHECK_EQ(add(x, int64_vec_t::cst(-1)), sub(x, int64_vec_t::cst(1)));
    CHECK_EQ(add(x, max(y, z)), max(add(x, y), add(x, z)));
    CHECK_EQ(add(x, min(y, z)), min(add(x, y), add(x, z)));
    int64_t r = 0;
    for (int i = 0; i < int64_vec_t::elem_count; ++i) {
      r += extract(x, i);
    }
    CHECK_EQ(r, reduce_add(x));
    CHECK_EQ(reduce_add(add(x, y)), reduce_add(x) + reduce_add(y));
  }
};

struct TestUint1VecArithmetic {
  static void Run() {
    CHECK_EQ(uint1_vec_t::cst(0), uint1_vec_t::cst(0));
    CHECK_EQ(uint1_vec_t::cst(1), uint1_vec_t::cst(1));
    CHECK_NE(uint1_vec_t::cst(0), uint1_vec_t::cst(1));
    std::minstd_rand0 engine;
    uint1_vec_t x = get_random<uint1_vec_t>(engine);
    uint1_vec_t y = get_random<uint1_vec_t>(engine);
    uint1_vec_t z = get_random<uint1_vec_t>(engine);
    CHECK_EQ(x, x);
    CHECK_NE(add(x, uint1_vec_t::cst(1)), x);
    CHECK_EQ(add(x, x), uint1_vec_t::cst(0));
    CHECK_EQ(add(x, uint1_vec_t::cst(0)), x);
    CHECK_EQ(add(x, y), add(y, x));
    CHECK_EQ(add(add(x, y), z), add(x, add(y, z)));
    CHECK_EQ(mul(x, x), x);
    CHECK_EQ(mul(x, uint1_vec_t::cst(1)), x);
    CHECK_EQ(mul(x, y), mul(y, x));
    CHECK_EQ(mul(mul(x, y), z), mul(x, mul(y, z)));
    CHECK_EQ(mul(x, add(y, z)), add(mul(x, y), mul(x, z)));
    CHECK_EQ(madd(x, y, z), add(x, mul(y, z)));
  }
};

struct TestUint1VecBitcounts {
  static void Run() {
    const int bits = uint1_vec_t::elem_count;
    CHECK_EQ(bits, static_cast<int>(8 * sizeof(uint1_vec_t)));
    CHECK_EQ(reduce_add(popcount(uint1_vec_t::cst(0))), 0);
    CHECK_EQ(reduce_add(popcount(uint1_vec_t::cst(1))), bits);
    for (int i = 0; (1 << i) < bits; ++i) {
      uint8_t buf[sizeof(uint1_vec_t)] = {0};
      buf[i / 8] = 1u << (i % 8);
      uint1_vec_t x = uint1_vec_t::load(buf);
      CHECK_EQ(extract(x, i), 1);
      CHECK_EQ(reduce_add(popcount(x)), 1);
    }
    for (int i = 0; (1 << i) < bits; ++i) {
      CHECK_EQ(reduce_add(popcount(uint1_vec_t::seq(i))), bits / 2);
      for (int j = 0; (1 << j) < bits; ++j) {
        if (i == j) {
          continue;
        }
        CHECK_EQ(
            reduce_add(popcount(add(uint1_vec_t::seq(i), uint1_vec_t::seq(j)))),
            bits / 2);
        CHECK_EQ(
            reduce_add(popcount(mul(uint1_vec_t::seq(i), uint1_vec_t::seq(j)))),
            bits / 4);
      }
    }
  }
};

struct TestUint1VecSeq {
  static void Run() {
    const int bytes = sizeof(uint1_vec_t);
    const int bits = uint1_vec_t::elem_count;
    CHECK_EQ(bits, 8 * bytes);
    constexpr int numslices = 1 + __builtin_ctz(bits);
    uint8_t slices[bytes * numslices];
    for (int i = 0; i < numslices; ++i) {
      store(slices + i * bytes, uint1_vec_t::seq(i));
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

struct TestInt64VecFormat {
  static void Run() {
    static constexpr int elems = int64_vec_t::elem_count;
    int64_t buf[elems];
    for (int i = 0; i < elems; ++i) {
      buf[i] = 100000000000ll + i;
    }
    std::string actual = fmt::format("{}", int64_vec_t::load(buf));
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

struct TestUint1VecFormat {
  static void Run() {
    static constexpr int bytes = sizeof(uint1_vec_t);
    uint8_t buf[bytes];
    for (int i = 0; i < bytes; ++i) {
      buf[i] = i;
    }
    std::string actual = fmt::format("{}", uint1_vec_t::load(buf));
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
  TEST(TestInt64VecLoadStore);
  TEST(TestUint1VecLoadStore);
  TEST(TestInt64VecArithmetic);
  TEST(TestUint1VecArithmetic);
  TEST(TestUint1VecBitcounts);
  TEST(TestUint1VecSeq);
  TEST(TestInt64VecFormat);
  TEST(TestUint1VecFormat);
}
