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
#include <format>

template <typename EType, int... sizes> class Vector;

template <typename EType, int... sizes> struct RowType {};
template <typename EType, int size0, int... sizes>
struct RowType<EType, size0, sizes...> {
  using Type = Vector<EType, sizes...>;
};
template <typename EType> struct RowType<EType> {
  using Type = Vector<EType>;
};

template <typename EType> struct Int64EType {
  using Type = EType;
};
template <Simd s> struct Int64EType<Uint1xN<s>> {
  using Type = Int64xN<s>;
};

template <typename EType, int... sizes> class Vector {
public:
  static constexpr int order = sizeof...(sizes);
  static constexpr int flatSize = (sizes * ... * 1);
  static constexpr int sizes_array[] = {sizes...};

  using ScalarType = ScalarType<EType>;
  using RowType = RowType<EType, sizes...>::Type;
  using Int64EType = Int64EType<EType>::Type;
  using Int64Vector = Vector<Int64EType, sizes...>;
  template <int... permutation>
  using TransposedType = Vector<EType, sizes_array[permutation]...>;

  using Indices = std::array<int, order>;

  static Indices get_strides() {
    Indices s;
    int p = 1;
    for (int i = order - 1; i >= 0; --i) {
      s[i] = p;
      p *= sizes_array[i];
    }
    return s;
  }

  static int flatten_indices(Indices indices) {
    Indices strides = get_strides();
    int f = 0;
    for (int i = 0; i < order; ++i) {
      f += strides[i] * indices[i];
    }
    return f;
  }

  static Indices unflatten_index(int flat_index) {
    Indices strides = get_strides();
    Indices result_indices;
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

  template <int... newSizes>
  friend Vector<EType, newSizes...> reshape(Vector x) {
    using ResultVector = Vector<EType, newSizes...>;
    static_assert(ResultVector::flatSize == flatSize);
    ResultVector result;
    for (int j = 0; j < flatSize; ++j) {
      result.elems[j] = x.elems[j];
    }
    return result;
  }

  template <int... permutation>
  friend TransposedType<permutation...> transpose(Vector x) {
    using ResultVector = TransposedType<permutation...>;
    static_assert(ResultVector::flatSize == flatSize);
    ResultVector result;
    static constexpr int permutation_array[] = {permutation...};
    for (int j = 0; j < flatSize; ++j) {
      Indices source_indices = unflatten_index(j);
      Indices permuted_indices;
      for (int k = 0; k < order; ++k) {
        permuted_indices[k] = source_indices[permutation_array[k]];
      }
      int result_index = ResultVector::flatten_indices(permuted_indices);
      result.elems[result_index] = x.elems[j];
    }
    return result;
  }

  friend Vector<ScalarType, sizes...> reduce_add(Vector x) {
    Vector<ScalarType, sizes...> result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = reduce_add(x.elems[i]);
    }
    return result;
  }

  friend Vector<ScalarType, sizes...> extract(Vector x, int i) {
    Vector<ScalarType, sizes...> result;
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

  friend Int64Vector popcount64(Vector x) {
    Int64Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = popcount64(x.elems[i]);
    }
    return result;
  }

  friend Int64Vector lzcount64(Vector x) {
    Int64Vector result;
    for (int i = 0; i < flatSize; ++i) {
      result.elems[i] = lzcount64(x.elems[i]);
    }
    return result;
  }

  EType elems[flatSize];
};

template <typename EType, int... sizes>
struct std::formatter<Vector<EType, sizes...>> {};

template <typename EType, int size0, int... sizes>
struct std::formatter<Vector<EType, size0, sizes...>> {
  using V = Vector<EType, size0, sizes...>;
  template <typename FormatContext>
  auto format(const V &x, FormatContext &ctx) const {
    auto it = ctx.out();
    it = std::format_to(it, "[");
    for (int i = 0; i < size0; ++i) {
      if (i > 0) {
        it = std::format_to(it, ", ");
      }
      it = std::format_to(it, "{}", row(x, i));
    }
    it = std::format_to(it, "]");
    return it;
  }
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
};

template <typename EType, int size0>
struct std::formatter<Vector<EType, size0>> {
  using V = Vector<EType, size0>;
  template <typename FormatContext>
  auto format(const V &x, FormatContext &ctx) const {
    auto it = ctx.out();
    it = std::format_to(it, "[");
    for (int i = 0; i < size0; ++i) {
      if (i > 0) {
        it = std::format_to(it, ", ");
      }
      it = std::format_to(it, "{}", x.elems[i]);
    }
    it = std::format_to(it, "]");
    return it;
  }
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
};

template <typename EType> struct std::formatter<Vector<EType>> {
  using V = Vector<EType>;
  template <typename FormatContext>
  auto format(const V &x, FormatContext &ctx) const {
    auto it = ctx.out();
    it = std::format_to(it, "[{}]", x.elems[0]);
    return it;
  }
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }
};

#endif // HAY_VECTOR_H_
