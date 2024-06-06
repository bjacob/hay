#include "test.h"

void check_impl(bool cond, const char *condstr, const char *file, int line) {
  if (!cond) {
    fprintf(stderr, "[  FAILED ]  CHECK(%s) at %s:%d\n", condstr, file, line);
    abort();
  }
}

void printTestLogLine(const char *header, const char *testname,
                      const char *simdname, int regbits) {
  fprintf(stderr, "%s  %s, %s (%d-bit)\n", header, testname, simdname, regbits);
}
