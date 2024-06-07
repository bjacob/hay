// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hay.h"
#include "testlib.h"

template <Simd s> struct TestRegLoadStore {
  static void Run() {
    using ST = SimdTraits<s>;
    uint8_t buf[2 * ST::RegBytes];
    for (int i = 0; i < 2 * ST::RegBytes; ++i) {
      buf[i] = i;
    }
    using Reg = ST::Reg;
    Reg x = ST::load(buf);
    Reg y = ST::load(buf + ST::RegBytes);
    CHECK(!ST::equal(x, y));
    ST::store(buf, y);
    CHECK(ST::equal(y, ST::load(buf)));
    CHECK(!memcmp(buf, buf + ST::RegBytes, ST::RegBytes));
  }
};

template <Simd s> struct TestRegArithmetic {
  static void Run() {
    using ST = SimdTraits<s>;
    CHECK(ST::equal(ST::zero(), ST::zero()));
    CHECK(ST::equal(ST::ones(), ST::ones()));
    CHECK(!ST::equal(ST::zero(), ST::ones()));
    using Reg = ST::Reg;
    Reg x = getRandomReg<s>();
    Reg y = getRandomReg<s>();
    Reg z = getRandomReg<s>();
    CHECK(ST::equal(x, x));
    CHECK(!ST::equal(ST::add(x, ST::ones()), x));
    CHECK(ST::equal(ST::add(x, x), ST::zero()));
    CHECK(ST::equal(ST::add(x, ST::zero()), x));
    CHECK(ST::equal(ST::add(x, y), ST::add(y, x)));
    CHECK(ST::equal(ST::add(ST::add(x, y), z), ST::add(x, ST::add(y, z))));
    CHECK(ST::equal(ST::mul(x, x), x));
    CHECK(ST::equal(ST::mul(x, ST::ones()), x));
    CHECK(ST::equal(ST::mul(x, y), ST::mul(y, x)));
    CHECK(ST::equal(ST::mul(ST::mul(x, y), z), ST::mul(x, ST::mul(y, z))));
    CHECK(ST::equal(ST::mul(x, ST::add(y, z)),
                    ST::add(ST::mul(x, z), ST::mul(y, z))));
    CHECK(ST::equal(ST::madd(x, y, z), ST::add(x, ST::mul(y, z))));
  }
};

template <Simd s> struct TestRegWaveAndPopcount {
  static void Run() {
    using ST = SimdTraits<s>;
    CHECK(ST::popcount(ST::zero()) == 0);
    CHECK(ST::popcount(ST::ones()) == ST::RegBits);
    for (int i = 0; (1 << i) < ST::RegBits; ++i) {
      CHECK(ST::popcount(ST::wave(i)) == ST::RegBits / 2);
      for (int j = 0; (1 << j) < ST::RegBits; ++j) {
        if (i == j) {
          continue;
        }
        CHECK(ST::popcount(ST::add(ST::wave(i), ST::wave(j))) ==
              ST::RegBits / 2);
        CHECK(ST::popcount(ST::mul(ST::wave(i), ST::wave(j))) ==
              ST::RegBits / 4);
      }
    }
  }
};

template <Simd s> struct TestRegWaveAsBitSlicedSequence {
  static void Run() {
    using ST = SimdTraits<s>;
    constexpr int numslices = 1 + __builtin_ctz(ST::RegBits);
    uint8_t slices[ST::RegBytes * numslices];
    for (int i = 0; i < numslices; ++i) {
      ST::store(slices + i * ST::RegBytes, ST::wave(i));
    }
    for (int b = 0; b < ST::RegBits; ++b) {
      uint64_t sequence_value = 0;
      for (int i = 0; i < numslices; ++i) {
        uint8_t slice_byte = slices[i * ST::RegBytes + b / 8];
        uint8_t slice_bit = (slice_byte & (1 << (b % 8))) ? 1 : 0;
        sequence_value |= slice_bit << i;
      }
      CHECK(sequence_value == static_cast<uint64_t>(b));
    }
  }
};

int main() {
  TEST(TestRegLoadStore);
  TEST(TestRegArithmetic);
  TEST(TestRegWaveAndPopcount);
  TEST(TestRegWaveAsBitSlicedSequence);
}
