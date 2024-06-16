// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cpuinfo.h"

#ifdef __x86_64__

#include <cpuid.h>

namespace {

struct hay_cpuid_regs_t {
  uint32_t eax;
  uint32_t ebx;
  uint32_t ecx;
  uint32_t edx;
};

hay_cpuid_regs_t hay_cpuid_raw(uint32_t eax, uint32_t ecx) {
  hay_cpuid_regs_t regs;
  __cpuid_count(eax, ecx, regs.eax, regs.ebx, regs.ecx, regs.edx);
  return regs;
}

typedef struct hay_cpuid_bounds_t {
  uint32_t max_base_eax;
  uint32_t max_extended_eax;
} hay_cpuid_bounds_t;

static hay_cpuid_bounds_t hay_cpuid_query_bounds() {
  hay_cpuid_bounds_t bounds;
  bounds.max_base_eax = hay_cpuid_raw(0, 0).eax;
  bounds.max_extended_eax = hay_cpuid_raw(0x80000000u, 0).eax;
  if (bounds.max_extended_eax < 0x80000000u)
    bounds.max_extended_eax = 0;
  return bounds;
}

static bool hay_cpuid_is_in_range(uint32_t eax, uint32_t ecx,
                                  hay_cpuid_bounds_t bounds) {
  if (eax < 0x80000000u) {
    // EAX is a base function id.
    if (eax > bounds.max_base_eax)
      return false;
  } else {
    // EAX is an extended function id.
    if (eax > bounds.max_extended_eax)
      return false;
  }
  if (ecx) {
    // ECX is a nonzero sub-function id.
    uint32_t max_ecx = hay_cpuid_raw(eax, 0).eax;
    if (ecx > max_ecx)
      return false;
  }
  return true;
}

static hay_cpuid_regs_t hay_cpuid_or_zero(uint32_t eax, uint32_t ecx,
                                          hay_cpuid_bounds_t bounds) {
  if (!hay_cpuid_is_in_range(eax, ecx, bounds)) {
    return (hay_cpuid_regs_t){0, 0, 0, 0};
  }
  return hay_cpuid_raw(eax, ecx);
}

} // namespace

uint64_t getCpuInfo() {
  hay_cpuid_bounds_t bounds = hay_cpuid_query_bounds();
  hay_cpuid_regs_t leaf7_0 = hay_cpuid_or_zero(7, 0, bounds);
  uint64_t out = 0;
  if (leaf7_0.ecx & (1 << 6)) {
    out |= CPUINFO_AVX512VBMI2;
  }
  if (leaf7_0.ecx & (1 << 14)) {
    out |= CPUINFO_AVX512VPOPCNTDQ;
  }
  return out;
}

#else // __x86_64__

uint64_t getCpuInfo() { return 0; }

#endif // __x86_64__