// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_TESTLIB_H_
#define HAY_TESTLIB_H_

#include "simd.h"

#include <cstdint>
#include <cstdio>
#include <random>

void check_impl(bool cond, const char *condstr, const char *file, int line);
void printTestLogLine(const char *header, const char *testname,
                      const char *simdname, int regbits);

#define CHECK(cond) check_impl(cond, #cond, __FILE__, __LINE__)

template <Simd s>
static void printTestLogLine(const char *header, const char *testname) {
  using ST = SimdTraits<s>;
  printTestLogLine(header, testname, ST::name(), ST::RegBits);
}

template <template <Simd> class TestClass, Simd s>
void TestOneSimd(const char *testname) {
  printTestLogLine<s>("[ RUN     ]", testname);
  if (SimdTraits<s>::detectCpu()) {
    TestClass<s>::Run();
    printTestLogLine<s>("[      OK ]", testname);
  } else {
    printTestLogLine<s>("[ SKIPPED ]", testname);
  }
}

template <template <Simd> class TestClass> void Test(const char *testname) {
  TestOneSimd<TestClass, Simd::Uint64>(testname);
#ifdef __aarch64__
  TestOneSimd<TestClass, Simd::Neon>(testname);
#endif
#ifdef __x86_64__
  TestOneSimd<TestClass, Simd::Avx2>(testname);
  TestOneSimd<TestClass, Simd::Avx512>(testname);
  TestOneSimd<TestClass, Simd::Avx512Bitalg>(testname);
#endif
}

#define TEST(TestClass) Test<TestClass>(#TestClass)

template <Simd s> SimdTraits<s>::Reg getRandomReg() {
  using ST = SimdTraits<s>;
  uint32_t buf[ST::RegBytes / sizeof(uint32_t)];
  std::minstd_rand0 engine;
  for (uint32_t &val : buf) {
    val = engine();
  }
  return ST::load(buf);
}

#endif // HAY_TESTLIB_H_
