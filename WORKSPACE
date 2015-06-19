new_http_archive(
  name = "gtest",
  url = "file:///Users/robertsdionne/Code/sacred/library/gtest-1.7.0.zip",
  # url = "https://googletest.googlecode.com/files/gtest-1.7.0.zip",
  sha256 = "2fa27ff3820916bd9a13ee1a29a2dbcbfe76beb34ef5278599e0c4bda46324af",
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
