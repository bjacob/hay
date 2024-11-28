// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "device.h"
#include "testlib.h"

struct TestDeviceAlloc {
  static void Run() {
    DevicePtr<int> deviceBuf = deviceAlloc<int>(100);
    for (int i = 0; i < 100; ++i)
      deviceBuf.val[i] = i;
    deviceDealloc(deviceBuf);
  }
};

struct TestDeviceCopyHostToDeviceToHost {
  static void Run() {
    int *hostBuf = new int[100];
    for (int i = 0; i < 100; ++i)
      hostBuf[i] = i;
    DevicePtr<int> deviceBuf = deviceAlloc<int>(100);
    copy(deviceBuf, hostBuf, 100);
    int *hostBuf2 = new int[100];
    copy(hostBuf2, deviceBuf, 100);
    for (int i = 0; i < 100; ++i)
      CHECK_EQ(hostBuf2[i], i);
    deviceDealloc(deviceBuf);
    delete[] hostBuf;
    delete[] hostBuf2;
  }
};

int main() {
  TEST(TestDeviceAlloc);
  TEST(TestDeviceCopyHostToDeviceToHost);
}
