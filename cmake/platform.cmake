# Copyright 2024 The Hay Authors (see AUTHORS).
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Adapted from IREE's iree_macros.cmake
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# HAY_ARCH: identifies the target CPU architecture. May be empty when this is
# ill-defined, such as multi-architecture builds.
# This should be kept consistent with the C preprocessor token HAY_ARCH defined
# in target_platform.h.
#-------------------------------------------------------------------------------

# First, get the raw CMake architecture name, not yet normalized. Even that is
# non-trivial: it usually is CMAKE_SYSTEM_PROCESSOR, but on some platforms, we
# have to read other variables instead.
if(CMAKE_OSX_ARCHITECTURES)
  # Borrowing from:
  # https://boringssl.googlesource.com/boringssl/+/c5f0e58e653d2d9afa8facc090ce09f8aaa3fa0d/CMakeLists.txt#43
  # https://github.com/google/XNNPACK/blob/2eb43787bfad4a99bdb613111cea8bc5a82f390d/CMakeLists.txt#L40
  list(LENGTH CMAKE_OSX_ARCHITECTURES NUM_ARCHES)
  if(${NUM_ARCHES} EQUAL 1)
    # Only one arch in CMAKE_OSX_ARCHITECTURES, use that.
    set(_HAY_UNNORMALIZED_ARCH "${CMAKE_OSX_ARCHITECTURES}")
  endif()
  # Leaving _HAY_UNNORMALIZED_ARCH empty disables arch code paths. We will
  # issue a performance warning about that below.
elseif(CMAKE_GENERATOR MATCHES "^Visual Studio " AND CMAKE_GENERATOR_PLATFORM)
  # Borrowing from:
  # https://github.com/google/XNNPACK/blob/2eb43787bfad4a99bdb613111cea8bc5a82f390d/CMakeLists.txt#L50
  set(_HAY_UNNORMALIZED_ARCH "${CMAKE_GENERATOR_PLATFORM}")
else()
  set(_HAY_UNNORMALIZED_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

string(TOLOWER "${_HAY_UNNORMALIZED_ARCH}" _HAY_UNNORMALIZED_ARCH_LOWERCASE)

# Normalize _HAY_UNNORMALIZED_ARCH into HAY_ARCH.
if(EMSCRIPTEN)
  # TODO: figure what to do about the wasm target, which masquerades as x86.
  set(HAY_ARCH "")
elseif((_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "aarch64") OR
        (_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm64") OR
        (_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm64e") OR
        (_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm64ec"))
  set(HAY_ARCH "arm_64")
elseif((_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "arm") OR
        (_HAY_UNNORMALIZED_ARCH_LOWERCASE MATCHES "^armv[5-8]"))
  set(HAY_ARCH "arm_32")
elseif((_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "x86_64") OR
        (_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "amd64") OR
        (_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "x64"))
  set(HAY_ARCH "x86_64")
elseif((_HAY_UNNORMALIZED_ARCH_LOWERCASE MATCHES "^i[3-7]86$") OR
        (_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "x86") OR
        (_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "win32"))
  set(HAY_ARCH "x86_32")
elseif(_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "riscv64")
  set(HAY_ARCH "riscv_64")
elseif(_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "riscv32")
  set(HAY_ARCH "riscv_32")
elseif(_HAY_UNNORMALIZED_ARCH_LOWERCASE STREQUAL "")
  set(HAY_ARCH "")
  message(WARNING "Performance advisory: architecture-specific code paths "
    "disabled because no target architecture was specified or we didn't know "
    "which CMake variable to read. Some relevant CMake variables:\n"
    "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}\n"
    "CMAKE_GENERATOR=${CMAKE_GENERATOR}\n"
    "CMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}\n"
    "CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}\n"
    )
else()
  set(HAY_ARCH "")
  message(SEND_ERROR "Unrecognized target architecture ${_HAY_UNNORMALIZED_ARCH_LOWERCASE}")
endif()

if (HAY_ARCH)
  message(STATUS "Target architecture: ${HAY_ARCH}")
endif()
