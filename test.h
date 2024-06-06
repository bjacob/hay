#ifndef HAY_TEST_H_
#define HAY_TEST_H_

#include "simd.h"

#include <cstdint>
#include <cstdio>
#include <random>

void check_impl(bool cond, const char *condstr, const char *file, int line);
void printTestLogLine(const char *header, const char *testname,
                      const char *simdname, int regbits);

#define CHECK(cond) check_impl(cond, #cond, __FILE__, __LINE__)

template <Simd p>
static void printTestLogLine(const char *header, const char *testname) {
  using PT = SimdTraits<p>;
  printTestLogLine(header, testname, PT::name(), PT::RegBits);
}

template <template <Simd> class TestClass, Simd p>
void TestOneSimd(const char *testname) {
  printTestLogLine<p>("[ RUN     ]", testname);
  if (SimdTraits<p>::detectCpu()) {
    TestClass<p>::Run();
    printTestLogLine<p>("[      OK ]", testname);
  } else {
    printTestLogLine<p>("[ SKIPPED ]", testname);
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

template <Simd p> SimdTraits<p>::Reg getRandomReg() {
  using PT = SimdTraits<p>;
  uint32_t buf[PT::RegBytes / sizeof(uint32_t)];
  std::minstd_rand0 engine;
  for (uint32_t &val : buf) {
    val = engine();
  }
  return PT::load(buf);
}

#endif // HAY_TEST_H_
