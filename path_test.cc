#include "hay.h"
#include "test.h"

template <Path p> struct TestRegLoadStore {
  static void Run() {
    using PT = PathTraits<p>;
    uint8_t buf[2 * PT::RegBytes];
    for (int i = 0; i < 2 * PT::RegBytes; ++i) {
      buf[i] = i;
    }
    using Reg = PT::Reg;
    Reg x = PT::load(buf);
    Reg y = PT::load(buf + PT::RegBytes);
    CHECK(!PT::equal(x, y));
    PT::store(buf, y);
    CHECK(PT::equal(y, PT::load(buf)));
    CHECK(!memcmp(buf, buf + PT::RegBytes, PT::RegBytes));
  }
};

template <Path p> struct TestRegArithmetic {
  static void Run() {
    using PT = PathTraits<p>;
    CHECK(PT::equal(PT::zero(), PT::zero()));
    CHECK(PT::equal(PT::ones(), PT::ones()));
    CHECK(!PT::equal(PT::zero(), PT::ones()));
    using Reg = PT::Reg;
    Reg x = getRandomReg<p>();
    Reg y = getRandomReg<p>();
    Reg z = getRandomReg<p>();
    CHECK(PT::equal(x, x));
    CHECK(!PT::equal(PT::add(x, PT::ones()), x));
    CHECK(PT::equal(PT::add(x, x), PT::zero()));
    CHECK(PT::equal(PT::add(x, PT::zero()), x));
    CHECK(PT::equal(PT::add(x, y), PT::add(y, x)));
    CHECK(PT::equal(PT::add(PT::add(x, y), z), PT::add(x, PT::add(y, z))));
    CHECK(PT::equal(PT::mul(x, x), x));
    CHECK(PT::equal(PT::mul(x, PT::ones()), x));
    CHECK(PT::equal(PT::mul(x, y), PT::mul(y, x)));
    CHECK(PT::equal(PT::mul(PT::mul(x, y), z), PT::mul(x, PT::mul(y, z))));
    CHECK(PT::equal(PT::mul(x, PT::add(y, z)),
                    PT::add(PT::mul(x, z), PT::mul(y, z))));
    CHECK(PT::equal(PT::madd(x, y, z), PT::add(x, PT::mul(y, z))));
  }
};

template <Path p> struct TestRegWaveAndPopcount {
  static void Run() {
    using PT = PathTraits<p>;
    CHECK(PT::popcount(PT::zero()) == 0);
    CHECK(PT::popcount(PT::ones()) == PT::RegBits);
    for (int i = 0; (1 << i) < PT::RegBits; ++i) {
      CHECK(PT::popcount(PT::wave(i)) == PT::RegBits / 2);
      for (int j = 0; (1 << j) < PT::RegBits; ++j) {
        if (i == j) {
          continue;
        }
        CHECK(PT::popcount(PT::add(PT::wave(i), PT::wave(j))) ==
              PT::RegBits / 2);
        CHECK(PT::popcount(PT::mul(PT::wave(i), PT::wave(j))) ==
              PT::RegBits / 4);
      }
    }
  }
};

template <Path p> struct TestRegWaveAsBitSlicedSequence {
  static void Run() {
    using PT = PathTraits<p>;
    constexpr int numslices = 1 + __builtin_ctz(PT::RegBits);
    uint8_t slices[PT::RegBytes * numslices];
    for (int i = 0; i < numslices; ++i) {
      PT::store(slices + i * PT::RegBytes, PT::wave(i));
    }
    for (int b = 0; b < PT::RegBits; ++b) {
      uint64_t sequence_value = 0;
      for (int i = 0; i < numslices; ++i) {
        uint8_t slice_byte = slices[i * PT::RegBytes + b / 8];
        uint8_t slice_bit = (slice_byte & (1 << (b % 8))) ? 1 : 0;
        sequence_value |= slice_bit << i;
      }
      CHECK(sequence_value == b);
    }
  }
};

int main() {
  TEST(TestRegLoadStore);
  TEST(TestRegArithmetic);
  TEST(TestRegWaveAndPopcount);
  TEST(TestRegWaveAsBitSlicedSequence);
}
