#ifndef HAY_PATH_ARM_NEON_H_
#define HAY_PATH_ARM_NEON_H_

#include "path_base.h"

#include <arm_neon.h>
#include <bit>

template <> struct PathDefinition<Path::ArmNeon> {
  using Reg = uint64x2_t;
  static const char *name() { return "Arm NEON"; }
  static bool detectCpu() { return true; }
  static Reg add(Reg x, Reg y) { return veorq_u64(x, y); }
  static Reg mul(Reg x, Reg y) { return vandq_u64(x, y); }
  static Reg madd(Reg x, Reg y, Reg z) { return add(x, mul(y, z)); }
  static Reg load(const void *from) {
    return vld1q_u64(static_cast<const uint64_t *>(from));
  }
  static void store(void *to, Reg x) {
    vst1q_u64(static_cast<uint64_t *>(to), x);
  }
  static bool equal(Reg x, Reg y) {
    Reg c = vceqq_u64(x, y);
    return ((vgetq_lane_u64(c, 0) & vgetq_lane_u64(c, 1)) + 1) == 0;
  }
  static int popcount(Reg x) {
    return std::popcount(vgetq_lane_u64(x, 0)) +
           std::popcount(vgetq_lane_u64(x, 1));
  }
  static Reg zero() { return vdupq_n_u8(0); }
  static Reg ones() { return vdupq_n_u8(0xFF); }
  static Reg wave(int i) {
    switch (i) {
    case 0:
      return vdupq_n_u8(0xAA);
    case 1:
      return vdupq_n_u8(0xCC);
    case 2:
      return vdupq_n_u8(0xF0);
    case 3:
      return vreinterpretq_u8_u16(vdupq_n_u16(0xFF00));
    case 4:
      return vreinterpretq_u8_u32(vdupq_n_u32(0xFFFF0000u));
    case 5:
      return vreinterpretq_u8_u64(vdupq_n_u64(0xFFFFFFFF00000000u));
    case 6:
      return vcombine_u8(vdup_n_u8(0), vdup_n_u8(0xFF));
    default:
      return zero();
    }
  }
};

#endif // HAY_PATH_ARM_NEON_H_
