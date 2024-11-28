// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "device.h"

#include <cstring>

#ifdef __HIP__
#include <fmt/format.h>
#include <hip/hip_runtime.h>

void hip_check_impl(hipError_t hip_error_code, const char *condstr,
                    const char *file, int line) {
  if (hip_error_code != hipSuccess) {
    fmt::print(stderr, "HIP Error \"{}\" produced by `{}` at {}:{}\n",
               hipGetErrorString(hip_error_code), condstr, file, line);
    exit(EXIT_FAILURE);
  }
}

#define HIP_CHECK(expr) hip_check_impl(expr, #expr, __FILE__, __LINE__)
#endif

DevicePtr<void> deviceAllocBytes(ssize_t size) {
  void *ptr = nullptr;
#ifdef __HIP__
  HIP_CHECK(hipMalloc(&ptr, size));
#else
  ptr = malloc(size);
#endif
  return {ptr};
}

void deviceDeallocBytes(DevicePtr<void> ptr) {
#ifdef __HIP__
  HIP_CHECK(hipFree(ptr.val));
#else
  free(ptr.val);
#endif
}

void copyBytes(DevicePtr<void> dst, const void *src, ssize_t size) {
#ifdef __HIP__
  HIP_CHECK(hipMemcpy(dst.val, src, size, hipMemcpyHostToDevice));
#else
  memcpy(dst.val, src, size);
#endif
}

void copyBytes(void *dst, const DevicePtr<const void> src, ssize_t size) {
#ifdef __HIP__
  HIP_CHECK(hipMemcpy(dst, src.val, size, hipMemcpyDeviceToHost));
#else
  memcpy(dst, src.val, size);
#endif
}
