// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testlib.h"

void check_fail_impl(std::string_view condstr, const char *file, int line) {
  fmt::print(stderr, "[  FAILED ]  At {}:{}:\n\n{}\n", file, line, condstr);
  abort();
}

void check_impl(bool cond, const char *condstr, const char *file, int line) {
  if (!cond) {
    check_fail_impl(condstr, file, line);
  }
}

void printTestLogLine(const char *header, const char *testname) {
  fmt::print(stderr, "{}  {}\n", header, testname);
}
