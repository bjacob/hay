# Copyright 2023 Benoit Jacob (github.com/bjacob)
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.24)
project(hay CXX)
set(CMAKE_CXX_STANDARD 20)
enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/bazel_like_rules.cmake)

cc_library(
    NAME
        "hay"
    HDRS
        "hay.h"
)

cc_library(
    NAME
        "testlib"
    HDRS
        "test.h"
    SRCS
        "test.cc"
)

cc_test(
    NAME
        "simd_test"
    SRCS
        "simd_test.cc"
    DEPS
        "hay"
        "testlib"
)
