new_http_archive(
  name = "gtest",
  url = "https://googletest.googlecode.com/files/gtest-1.7.0.zip",
  sha256 = "247ca18dd83f53deb1328be17e4b1be31514cedfc1e3424f672bf11fd7e0d60d",
  build_file = "gtest.BUILD",
)

bind(
  name = "googletest",
  actual = "@gtest//:main",
)

new_http_archive(
  name = "gflags",
  url = "file:///Users/robertsdionne/Code/sacred/library/gflags.zip",
  sha256 = "4db00c986de59961f53e19499a1a05e330f1e299aff605b16b301d37112249c2",
  build_file = "gflags.BUILD",
)

bind(
  name = "googleflags",
  actual = "@gflags//:gflags",
)
