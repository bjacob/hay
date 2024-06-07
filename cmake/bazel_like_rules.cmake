# Copyright 2024 The Hay Authors (see AUTHORS).
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

function(cc_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "HDRS;SRCS;COPTS;DEPS"
    ${ARGN}
  )

  set(_NAME "${_RULE_NAME}")

  file(RELATIVE_PATH _SUBDIR ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR})

  if("${_RULE_SRCS}" STREQUAL "")
    # Generating a header-only library.
    add_library(${_NAME} INTERFACE)
    set_target_properties(${_NAME} PROPERTIES PUBLIC_HEADER "${_RULE_HDRS}")
    target_include_directories(${_NAME}
      INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )
    target_link_libraries(${_NAME}
      INTERFACE
        ${_RULE_DEPS}
    )
    target_compile_definitions(${_NAME}
      INTERFACE
        ${_RULE_DEFINES}
    )
  else()
    # Generating a static binary library.
    add_library(${_NAME} STATIC ${_RULE_SRCS} ${_RULE_HDRS})
    set_target_properties(${_NAME} PROPERTIES PUBLIC_HEADER "${_RULE_HDRS}")
    target_compile_options(${_NAME}
      PRIVATE
        ${_RULE_COPTS}
    )
    target_link_libraries(${_NAME}
      PUBLIC
        ${_RULE_DEPS}
    )
    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )
  endif()

  add_library(${PROJECT_NAME}::${_NAME} ALIAS ${_NAME})
endfunction()

# cc_test()
# 
# CMake function to imitate Bazel's cc_test rule.
function(cc_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;COPTS;DEPS"
    ${ARGN}
  )

  set(_NAME "${_RULE_NAME}")

  add_executable(${_NAME} "")
  target_sources(${_NAME}
    PRIVATE
      ${_RULE_SRCS}
  )
  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  target_compile_options(${_NAME}
    PRIVATE
      ${_RULE_COPTS}
  )
  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
  )
  add_test(
    NAME
      ${_NAME}
    COMMAND
      "$<TARGET_FILE:${_NAME}>"
    )
endfunction()