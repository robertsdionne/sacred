#include <gtest/gtest.h>

#include "hashed_lookup.hpp"

namespace sacred {

TEST(HashedLookup, Lookup) {
  auto hashed_lookup = HashedLookup<>();
  auto data = vector<float>({0, 1, 2});

  EXPECT_EQ(0, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {0, 0, 0, 0}));
  EXPECT_EQ(-1, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {0, 0, 0, 1}));
  EXPECT_EQ(-2, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {0, 0, 1, 0}));
  EXPECT_EQ(1, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {0, 0, 1, 1}));
  EXPECT_EQ(0, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {0, 1, 0, 0}));
  EXPECT_EQ(-1, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {0, 1, 0, 1}));
  EXPECT_EQ(-2, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {0, 1, 1, 0}));
  EXPECT_EQ(1, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {0, 1, 1, 1}));
  EXPECT_EQ(-1, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {1, 0, 0, 0}));
  EXPECT_EQ(0, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {1, 0, 0, 1}));
  EXPECT_EQ(1, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {1, 0, 1, 0}));
  EXPECT_EQ(-2, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {1, 0, 1, 1}));
  EXPECT_EQ(-1, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {1, 1, 0, 0}));
  EXPECT_EQ(0, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {1, 1, 0, 1}));
  EXPECT_EQ(0, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {1, 1, 1, 0}));
  EXPECT_EQ(-1, hashed_lookup.Lookup(data, data.size(), {2, 2, 2, 2}, {8, 4, 2, 1}, {1, 1, 1, 1}));
}

}  // namespace sacred
