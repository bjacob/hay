#include "test.h"

void check_impl(bool cond, const char *condstr, const char *file, int line) {
  if (!cond) {
    fprintf(stderr, "[  FAILED ]  CHECK(%s) at %s:%d\n", condstr, file, line);
    abort();
  }
}

void printTestLogLine(const char *header, const char *testname,
                      const char *pathname, int regbits) {
  fprintf(stderr, "%s  %s, path: %s (%d-bit)\n", header, testname, pathname,
          regbits);
}
