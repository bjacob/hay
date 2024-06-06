#ifndef HAY_SIMD_H_
#define HAY_SIMD_H_

#include "simd_base.h"
#include "simd_uint64.h"

#if defined __aarch64__
#include "simd_arm.h"
#elif defined __x86_64__
#include "simd_x86.h"
#endif

#endif // HAY_SIMD_H_
