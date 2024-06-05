#ifndef HAY_TEST_H_
#define HAY_TEST_H_

#include "path.h"

#include <cstdint>
#include <cstdio>
#include <random>

void check_impl(bool cond, const char *condstr, const char *file, int line);
void printTestLogLine(const char *header, const char *testname,
                      const char *pathname, int regbits);

#define CHECK(cond) check_impl(cond, #cond, __FILE__, __LINE__)

template <Path p>
static void printTestLogLine(const char *header, const char *testname) {
  using PT = PathTraits<p>;
  printTestLogLine(header, testname, PT::name(), PT::RegBits);
}

template <template <Path> class TestClass, Path p>
void TestOnePath(const char *testname) {
  printTestLogLine<p>("[ RUN     ]", testname);
  if (PathTraits<p>::detectCpu()) {
    TestClass<p>::Run();
    printTestLogLine<p>("[      OK ]", testname);
  } else {
    printTestLogLine<p>("[ SKIPPED ]", testname);
  }
}

template <template <Path> class TestClass> void Test(const char *testname) {
  TestOnePath<TestClass, Path::Uint64>(testname);
#ifdef __aarch64__
  TestOnePath<TestClass, Path::ArmNeon>(testname);
#endif
#ifdef __x86_64__
  TestOnePath<TestClass, Path::X86Avx2>(testname);
  TestOnePath<TestClass, Path::X86Avx512>(testname);
  TestOnePath<TestClass, Path::X86Avx512Bitalg>(testname);
#endif
}

#define TEST(TestClass) Test<TestClass>(#TestClass)

template <Path p> PathTraits<p>::Reg getRandomReg() {
  using PT = PathTraits<p>;
  uint32_t buf[PT::RegBytes / sizeof(uint32_t)];
  std::minstd_rand0 engine;
  for (uint32_t &val : buf) {
    val = engine();
  }
  return PT::load(buf);
}

#endif // HAY_TEST_H_
