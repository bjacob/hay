// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testlib.h"

void check_fail_impl(std::string_view condstr, const char *file, int line) {
  fprintf(stderr, "[  FAILED ]  At %s:%d:\n\n%.*s\n", file, line,
          static_cast<int>(condstr.size()), condstr.data());
  abort();
}

void check_impl(bool cond, const char *condstr, const char *file, int line) {
  if (!cond) {
    check_fail_impl(condstr, file, line);
  }
}

void printTestLogLine(const char *header, const char *testname,
                      const char *simdname) {
  fprintf(stderr, "%s  %s, %s\n", header, testname, simdname);
}
