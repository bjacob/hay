#ifndef HAY_PATH_X86_AVX2_H_
#define HAY_PATH_X86_AVX2_H_

#include "path_base.h"
template <> struct PathDefinition<Path::X86Avx2> {
  using Reg = __m256i;
  static const char *name() { return "AVX2"; }
  static bool detectCpu() { return true; }
  static Reg add(Reg x, Reg y) { return veorq_u8(x, y); }
  static Reg mul(Reg x, Reg y) { return vandq_u8(x, y); }
  static Reg madd(Reg x, Reg y, Reg z) { return add(x, mul(y, z)); }
  static Reg load(const void *from) {
    return vld1q_u8(static_cast<const uint8_t *>(from));
  }
  static void store(void *to, Reg x) {
    vst1q_u8(static_cast<uint8_t *>(to), x);
  }
  static Reg zero() { return vdupq_n_u8(0); }
  static Reg ones() { return vdupq_n_u8(0xFF); }
  static Reg wave(int i) {
    return i == 0   ? vdupq_n_u8(0xAA)
           : i == 1 ? vdupq_n_u8(0xCC)
           : i == 2 ? vdupq_n_u8(0xF0)
           : i == 3 ? vreinterpretq_u8_u16(vdupq_n_u16(0xFF00))
           : i == 4 ? vreinterpretq_u8_u32(vdupq_n_u32(0xFFFF0000u))
           : i == 5 ? vreinterpretq_u8_u64(vdupq_n_u64(0xFFFFFFFF00000000u))
           : i == 6 ? vcombine_u8(vdup_n_u8(0), vdup_n_u8(0xFF))
                    : zero();
  }
};

#endif // HAY_PATH_X86_AVX2_H_
