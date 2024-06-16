// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_CPUINFO_H_
#define HAY_CPUINFO_H_

#include <cstdint>

const uint64_t CPUINFO_AVX512VBMI2 = 0x1;
const uint64_t CPUINFO_AVX512VPOPCNTDQ = 0x2;

uint64_t getCpuInfo();

#endif // HAY_CPUINFO_H_