#ifndef HAY_SIMD_BASE_H_
#define HAY_SIMD_BASE_H_

enum class Path {
  Uint64,
  Neon,
  Avx512,
};

template <Path p> struct PathDefinition {};

template <Path p> struct PathTraits : PathDefinition<p> {
  using Base = PathDefinition<p>;
  using Reg = Base::Reg;
  static constexpr int RegBytes = sizeof(Reg);
  static constexpr int RegBits = 8 * RegBytes;
};

#endif // HAY_SIMD_BASE_H_
