#ifndef HAY_SIMD_BASE_H_
#define HAY_SIMD_BASE_H_

enum class Simd {
  Uint64,
  Neon,
  Avx512,
};

template <Simd p> struct SimdDefinition {};

template <Simd p> struct SimdTraits : SimdDefinition<p> {
  using Base = SimdDefinition<p>;
  using Reg = Base::Reg;
  static constexpr int RegBytes = sizeof(Reg);
  static constexpr int RegBits = 8 * RegBytes;
};

#endif // HAY_SIMD_BASE_H_
