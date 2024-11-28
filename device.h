// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_DEVICE_H_
#define HAY_DEVICE_H_

#include <cstdlib>

template <typename T> struct DevicePtr { T *val; };

template <typename U, typename T> DevicePtr<U> cast(DevicePtr<T> ptr) {
  return DevicePtr<U>{static_cast<U *>(ptr.val)};
}

DevicePtr<void> deviceAllocBytes(ssize_t size);
void deviceDeallocBytes(DevicePtr<void> ptr);
void copyBytes(DevicePtr<void> dst, const void *src, ssize_t size);
void copyBytes(void *dst, const DevicePtr<const void> src, ssize_t size);

template <typename T> DevicePtr<T> deviceAlloc(ssize_t size) {
  return cast<T>(deviceAllocBytes(size * sizeof(T)));
}

template <typename T> void deviceDealloc(DevicePtr<T> ptr) {
  deviceDeallocBytes(cast<void>(ptr));
}

template <typename T> void copy(DevicePtr<T> dst, const T *src, ssize_t size) {
  copyBytes(cast<void>(dst), src, size * sizeof(T));
}

template <typename T>
void copy(T *dst, const DevicePtr<const T> src, ssize_t size) {
  copyBytes(dst, cast<const void>(src), size * sizeof(T));
}

template <typename T> void copy(T *dst, const DevicePtr<T> src, ssize_t size) {
  copy(dst, cast<const T>(src), size);
}

#endif // HAY_DEVICE_H_