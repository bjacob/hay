// Copyright 2024 The Hay Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HAY_VECTOR_H_
#define HAY_VECTOR_H_

#include "simd.h"

#include <array>
#include <cstdint>
#include <numeric>

#include <fmt/format.h>

using Index = int;

template <int order> struct Indices : std::array<Index, order> {};

// Deduction guide allowing passing an initializer list of integer sizes for the
// shape of a Vector type.
template <typename... IntTypes>
Indices(IntTypes...) -> Indices<sizeof...(IntTypes)>;

template <int order> struct fmt::formatter<Indices<order>> {
  using I = Indices<order>;
  template <typename FormatContext>
  auto format(const I &x, FormatContext &ctx) const {
    auto it = ctx.out();
    it = fmt::format_to(it, "[");
    for (int i = 0; i < order; ++i) {
      if (i > 0) {
        it = fmt::format_to(it, ", ");
      }
      it = fmt::format_to(it, "{}", x[i]);
    }
    it = fmt::format_to(it, "]");
    return it;
  }
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }
};

template <int order, int ndrops>
constexpr Indices<std::max(0, order - ndrops)> drop(Indices<order> src,
                                                    Indices<ndrops> drops) {
  Indices<std::max(0, order - ndrops)> result{};
  if (result.size() == 0) {
    return result;
  }
  int src_pos = 0;
  int result_pos = 0;
  for (int drop : drops) {
    // Copy src values until we hit `drop`.
    while (src_pos < drop) {
      result[result_pos++] = src[src_pos++];
    }
    src_pos++; // Hit `drop`, so drop that one.
  }
  // Append remaining src values past the last drop.
  while (src_pos < order) {
    result[result_pos++] = src[src_pos++];
  }
  return result;
}

template <int order1, int order2>
constexpr Indices<order1 + order2> concat(Indices<order1> indices1,
                                          Indices<order2> indices2) {
  Indices<order1 + order2> result{};
  for (int i = 0; i < order1; ++i) {
    result[i] = indices1[i];
  }
  for (int i = 0; i < order2; ++i) {
    result[order1 + i] = indices2[i];
  }
  return result;
}

template <typename EType, Indices sizes> class Vector;

template <typename EType, Indices sizes>
using RowType = Vector<EType, drop(sizes, Indices{0})>;

template <typename EType> struct Int64EType {
  using Type = EType;
};
template <> struct Int64EType<Uint1xN> {
  using Type = Int64xN;
};

template <int order> inline constexpr int product(Indices<order> sizes) {
  return std::reduce(std::begin(sizes), std::end(sizes), Index{1},
                     std::multiplies<Index>());
}

template <int order>
inline constexpr Indices<order> permute(Indices<order> src,
                                        Indices<order> permutation) {
  Indices<order> result;
  for (int i = 0; i < order; ++i) {
    result[i] = src[permutation[i]];
  }
  return result;
}

template <typename EType, Indices sizes> class Vector {
public:
  static constexpr int order = sizes.size();
  static constexpr int flatSize = product(sizes);

  using ScalarType = ScalarType<EType>;
  using RowType = RowType<EType, sizes>;
  using Int64EType = Int64EType<EType>::Type;
  using Int64Vector = Vector<Int64EType, sizes>;
  template <Indices permutation>
  using TransposedType = Vector<EType, permute(sizes, permutation)>;

  using IndicesType = Indices<order>;

  static IndicesType get_strides() {
    IndicesType s;
    Index p = 1;
    for (int i = order - 1; i >= 0; --i) {
      s[i] = p;
      p *= sizes[i];
    }
    return s;
  }

  static int flatten_indices(IndicesType indices) {
    IndicesType strides = get_strides();
    Index f = 0;
    for (int i = 0; i < order; ++i) {
      f += strides[i] * indices[i];
    }
    return f;
  }

  static IndicesType unflatten_index(int flat_index) {
    IndicesType strides = get_strides();
    IndicesType result_indices;
    for (int i = 0; i < order; ++i) {
      result_indices[i] = flat_index / strides[i];
      flat_index -= result_indices[i] * strides[i];
    }
    return result_indices;
  }

  static Vector cst(ScalarType c) {
    Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = EType::cst(c);
    }
    return result;
  }

  static Vector load(const void *from) {
    Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] =
          EType::load(static_cast<const uint8_t *>(from) + i * sizeof(EType));
    }
    return result;
  }

  friend void store(void *to, Vector x) {
    for (int i = 0; i < flatSize; ++i) {
      store(static_cast<uint8_t *>(to) + i * sizeof(EType), x.elems[i]);
    }
  }

  friend Vector add(Vector x, Vector y) {
    Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = add(x.elems[i], y.elems[i]);
    }
    return result;
  }

  friend Vector sub(Vector x, Vector y) {
    Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = sub(x.elems[i], y.elems[i]);
    }
    return result;
  }

  friend Vector min(Vector x, Vector y) {
    Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = min(x.elems[i], y.elems[i]);
    }
    return result;
  }

  friend Vector max(Vector x, Vector y) {
    Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = max(x.elems[i], y.elems[i]);
    }
    return result;
  }

  friend Vector mul(Vector x, Vector y) {
    Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = mul(x.elems[i], y.elems[i]);
    }
    return result;
  }

  friend Vector madd(Vector x, Vector y, Vector z) {
    Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = madd(x.elems[i], y.elems[i], z.elems[i]);
    }
    return result;
  }

  friend bool operator==(Vector x, Vector y) {
    for (int i = 0; i < flatSize; ++i) {
      if (!(x.elems[i] == y.elems[i])) {
        return false;
      }
    }
    return true;
  }

  friend RowType row(Vector x, int i) {
    RowType result;
    for (int j = 0; j < result.flatSize; ++j) {
      result.elems[j] = x.elems[j + i * result.flatSize];
    }
    return result;
  }

  friend void insert_row(Vector &dst, RowType x, int i) {
    for (int j = 0; j < x.flatSize; ++j) {
      dst.elems[j + i * x.flatSize] = x.elems[j];
    }
  }

  template <Indices newSizes> friend Vector<EType, newSizes> reshape(Vector x) {
    using ResultVector = Vector<EType, newSizes>;
    static_assert(ResultVector::flatSize == flatSize);
    ResultVector result;
    for (int j = 0; j < flatSize; ++j) {
      result.elems[j] = x.elems[j];
    }
    return result;
  }

  template <Indices permutation>
  friend TransposedType<permutation> transpose(Vector x) {
    using ResultVector = TransposedType<permutation>;
    static_assert(ResultVector::flatSize == flatSize);
    ResultVector result;
    for (int j = 0; j < flatSize; ++j) {
      Indices source_indices = unflatten_index(j);
      Indices permuted_indices = permute(source_indices, permutation);
      int result_index = ResultVector::flatten_indices(permuted_indices);
      result.elems[result_index] = x.elems[j];
    }
    return result;
  }

  template <Index c0, Index c1>
  friend Vector<EType, drop(sizes, Indices{c0, c1})> contract(Vector x) {
    static_assert(c0 < c1);
    static_assert(sizes[c0] == sizes[c1]);
    using ResultVector = Vector<EType, drop(sizes, Indices{c0, c1})>;
    using ResultIndices = ResultVector::IndicesType;
    ResultVector r = ResultVector::cst(0);
    for (int i = 0; i < flatSize; ++i) {
      Indices src_ind = unflatten_index(i);
      if (src_ind[c0] == src_ind[c1]) {
        ResultIndices dst_ind = drop(src_ind, Indices{c0, c1});
        int dst_flat_idx = ResultVector::flatten_indices(dst_ind);
        r.elems[dst_flat_idx] = add(r.elems[dst_flat_idx], x.elems[i]);
      }
    }
    return r;
  }

  friend EType trace(Vector x) {
    static_assert(order == 2);
    return contract<0, 1>(x).elems[0];
  }

  friend Vector<ScalarType, sizes> reduce_add(Vector x) {
    Vector<ScalarType, sizes> result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = reduce_add(x.elems[i]);
    }
    return result;
  }

  friend Vector<ScalarType, sizes> extract(Vector x, int i) {
    Vector<ScalarType, sizes> result;
    for (int j = 0; j < flatSize; ++j) {
      result.elems[j] = extract(x.elems[j], i);
    }
    return result;
  }

  static Vector seq(int i) {
    Vector result;
    int j = 0;
    for (; (1 << j) < EType::elem_count && j < flatSize; ++j) {
      result.elems[j] = EType::seq(j);
    }
    int k = 0;
    for (; j < flatSize; ++j, ++k) {
      result.elems[j] = EType::cst((i >> k) & 1);
    }
    return result;
  }

  friend Int64Vector popcount(Vector x) {
    Int64Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = popcount(x.elems[i]);
    }
    return result;
  }

  EType elems[flatSize];
};

template <Index c1, Index c2, typename EType, Indices sizes1, Indices sizes2>
Vector<EType, concat(drop(sizes1, Indices{c1}), drop(sizes2, Indices{c2}))>
contract(Vector<EType, sizes1> v1, Vector<EType, sizes2> v2) {
  static_assert(sizes1[c1] == sizes2[c2]);
  using Vector1 = Vector<EType, sizes1>;
  using Vector2 = Vector<EType, sizes2>;
  using ResultVector = Vector<EType, concat(drop(sizes1, Indices{c1}),
                                            drop(sizes2, Indices{c2}))>;
  using ResultIndices = ResultVector::IndicesType;
  ResultVector r = ResultVector::cst(0);
  for (int i1 = 0; i1 < Vector1::flatSize; ++i1) {
    auto ind1 = Vector1::unflatten_index(i1);
    for (int i2 = 0; i2 < Vector2::flatSize; ++i2) {
      auto ind2 = Vector2::unflatten_index(i2);
      if (ind1[c1] == ind2[c2]) {
        ResultIndices dst_ind =
            concat(drop(ind1, Indices{c1}), drop(ind2, Indices{c2}));
        int dst_flat_idx = ResultVector::flatten_indices(dst_ind);
        r.elems[dst_flat_idx] =
            madd(r.elems[dst_flat_idx], v1.elems[i1], v2.elems[i2]);
      }
    }
  }
  return r;
}

template <typename EType, Indices sizes1, Indices sizes2>
Vector<EType, {sizes1[0], sizes2[1]}> matmul(Vector<EType, sizes1> v1,
                                             Vector<EType, sizes2> v2) {
  static_assert(sizes1.size() == 2);
  static_assert(sizes2.size() == 2);
  static_assert(sizes1[1] == sizes2[0]);
  return contract<1, 0>(v1, v2);
}

template <typename EType, Indices sizes>
struct fmt::formatter<Vector<EType, sizes>> {
  using V = Vector<EType, sizes>;
  template <typename FormatContext>
  auto format(const V &x, FormatContext &ctx) const {
    auto it = ctx.out();
    if constexpr (V::order == 0) {
      it = fmt::format_to(it, "{}", x.elems[0]);
    } else {
      it = fmt::format_to(it, "[");
      for (int i = 0; i < sizes[0]; ++i) {
        if (i > 0) {
          it = fmt::format_to(it, ", ");
        }
        it = fmt::format_to(it, "{}", row(x, i));
      }
      it = fmt::format_to(it, "]");
    }
    return it;
  }
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }
};

#endif // HAY_VECTOR_H_
