// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_TESTLIB_H_
#define HAY_TESTLIB_H_

#include "simd.h"
#include "vector.h"

#include <cstdint>
#include <cstdio>
#include <format>
#include <random>
#include <string_view>

void check_fail_impl(std::string_view condstr, const char *file, int line);

void check_impl(bool cond, const char *condstr, const char *file, int line);

template <typename X, typename Y>
void check_eq_impl(const X &x, const Y &y, bool expected_equality,
                   const char *xstr, const char *ystr, const char *file,
                   int line) {
  if ((x == y) != expected_equality) {
    std::string str = std::format(
        "Expected {} between {}, which has the value:\n{}\n\nand {}, which "
        "has "
        "the value:\n{}\n\n",
        (expected_equality ? "equality" : "non-equality"), xstr, x, ystr, y);
    check_fail_impl(str, file, line);
  }
}

void printTestLogLine(const char *header, const char *testname,
                      const char *simdname);

#define CHECK(cond) check_impl(cond, #cond, __FILE__, __LINE__)
#define CHECK_EQ(x, y) check_eq_impl((x), (y), true, #x, #y, __FILE__, __LINE__)
#define CHECK_NE(x, y)                                                         \
  check_eq_impl((x), (y), false, #x, #y, __FILE__, __LINE__)

template <template <Simd> class TestClass, Simd s>
void TestOneSimd(const char *testname) {
  if (detect<s>()) {
    printTestLogLine("[ RUN     ]", testname, name<s>());
    TestClass<s>::Run();
    printTestLogLine("[      OK ]", testname, name<s>());
  } else {
    printTestLogLine("[ SKIPPED ]", testname, name<s>());
  }
}

template <template <Simd> class TestClass> void Test(const char *testname) {
  TestOneSimd<TestClass, Simd::U64>(testname);
#ifdef __aarch64__
  TestOneSimd<TestClass, Simd::Neon>(testname);
#endif
#ifdef __x86_64__
  TestOneSimd<TestClass, Simd::Avx512>(testname);
#endif
}

#define TEST(TestClass) Test<TestClass>(#TestClass)

template <typename T> struct GetRandomImpl {};

template <typename T> T getRandom(std::minstd_rand0 &engine) {
  return GetRandomImpl<T>::Run(engine);
}

template <Simd s> struct GetRandomImpl<Uint1xN<s>> {
  static Uint1xN<s> Run(std::minstd_rand0 &engine) {
    uint32_t buf[sizeof(Uint1xN<s>) / sizeof(uint32_t)];
    for (uint32_t &val : buf) {
      val = engine();
    }
    return Uint1xN<s>::load(buf);
  }
};

template <Simd s> struct GetRandomImpl<Int64xN<s>> {
  static Int64xN<s> Run(std::minstd_rand0 &engine) {
    int64_t buf[Int64xN<s>::elem_count];
    for (int64_t &val : buf) {
      val = engine() - engine.max() / 2;
    }
    return Int64xN<s>::load(buf);
  }
};

template <typename EType, int... sizes>
struct GetRandomImpl<Vector<EType, sizes...>> {
  using V = Vector<EType, sizes...>;
  static V Run(std::minstd_rand0 &engine) {
    V result;
    for (int i = 0; i < V::flatSize; ++i) {
      result.elems[i] = getRandom<EType>(engine);
    }
    return result;
  }
};

#endif // HAY_TESTLIB_H_
