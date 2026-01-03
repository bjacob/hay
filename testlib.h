// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_TESTLIB_H_
#define HAY_TESTLIB_H_

#include "array.h"

#include <cstdint>
#include <fmt/format.h>
#include <random>
#include <string_view>

namespace hay {

void check_fail_impl(std::string_view condstr, const char *file, int line);

void check_impl(bool cond, const char *condstr, const char *file, int line);

template <typename X, typename Y>
void check_eq_impl(const X &x, const Y &y, bool expected_equality,
                   const char *xstr, const char *ystr, const char *file,
                   int line) {
  if ((x == y) != expected_equality) {
    std::string str = fmt::format(
        "Expected {} between {}, which has the value:\n{}\n\nand {}, which "
        "has "
        "the value:\n{}\n\n",
        (expected_equality ? "equality" : "non-equality"), xstr, x, ystr, y);
    check_fail_impl(str, file, line);
  }
}

void printTestLogLine(const char *header, const char *testname);

#define CHECK(cond) check_impl(cond, #cond, __FILE__, __LINE__)
#define CHECK_EQ(x, y) check_eq_impl((x), (y), true, #x, #y, __FILE__, __LINE__)
#define CHECK_NE(x, y)                                                         \
  check_eq_impl((x), (y), false, #x, #y, __FILE__, __LINE__)

template <typename TestClass> void Test(const char *testname) {
  printTestLogLine("[ RUN     ]", testname);
  TestClass::Run();
  printTestLogLine("[      OK ]", testname);
}

#define TEST(TestClass) Test<TestClass>(#TestClass)

template <typename T> struct GetRandomImpl {};

template <typename T> T get_random(std::minstd_rand0 &engine) {
  return GetRandomImpl<T>::Run(engine);
}

template <> struct GetRandomImpl<hay::uint1_vec_t> {
  static hay::uint1_vec_t Run(std::minstd_rand0 &engine) {
    uint32_t buf[sizeof(hay::uint1_vec_t) / sizeof(uint32_t)];
    for (uint32_t &val : buf) {
      val = engine();
    }
    return hay::uint1_vec_t::load(buf);
  }
};

template <> struct GetRandomImpl<hay::int64_vec_t> {
  static hay::int64_vec_t Run(std::minstd_rand0 &engine) {
    int64_t buf[hay::int64_vec_t::elem_count];
    for (int64_t &val : buf) {
      val = engine() - engine.max() / 2;
    }
    return hay::int64_vec_t::load(buf);
  }
};

template <typename EType, Indices sizes>
struct GetRandomImpl<Array<EType, sizes>> {
  using V = Array<EType, sizes>;
  static V Run(std::minstd_rand0 &engine) {
    V result;
    for (int i = 0; i < V::flatSize; ++i) {
      result.elems[i] = get_random<EType>(engine);
    }
    return result;
  }
};

} // namespace hay

#endif // HAY_TESTLIB_H_
