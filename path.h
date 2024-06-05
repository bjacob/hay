#ifndef HAY_PATH_H_
#define HAY_PATH_H_

#include "path_base.h"
#include "path_uint64.h"

#if defined __aarch64__
#include "path_arm_neon.h"
#elif defined __x86_64__
#if defined __AVX2__
#include "path_x86_avx2.h"
#elif defined __AVX512F__
#include "path_x86_avx512.h"
#endif
#endif

#endif // HAY_PATH_H_
