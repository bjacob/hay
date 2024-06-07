# Copyright 2024 The Hay Authors (see AUTHORS).
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
include(cmake/platform.cmake)

set(HAY_TARGET_CPU "native" CACHE STRING "Target CPU")
message(STATUS "Target CPU: ${HAY_TARGET_CPU}")

if (HAY_ARCH STREQUAL "x86_64")
  set(HAY_TARGET_CPU_FLAG "-march=${HAY_TARGET_CPU}")
else()
  set(HAY_TARGET_CPU_FLAG "-mcpu=${HAY_TARGET_CPU}")
endif()

add_compile_options(
    "${HAY_TARGET_CPU_FLAG}"
    "-Wall"
    "-Wextra"
    "-Werror"
    "-fno-lax-vector-conversions"
)

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
        "testlib.h"
    SRCS
        "testlib.cc"
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
