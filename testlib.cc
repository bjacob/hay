// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testlib.h"

void check_impl(bool cond, const char *condstr, const char *file, int line) {
  if (!cond) {
    fprintf(stderr, "[  FAILED ]  CHECK(%s) at %s:%d\n", condstr, file, line);
    abort();
  }
}

void printTestLogLine(const char *header, const char *testname,
                      const char *simdname) {
  fprintf(stderr, "%s  %s, %s\n", header, testname, simdname);
}
