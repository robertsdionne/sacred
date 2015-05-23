
cc_binary(
  name = "convcl",
  srcs = [
    "convcl.cpp",
  ],
  linkopts = [
    "-framework OpenCL",
  ],
)

cc_binary(
  name = "convcpp",
  srcs = [
    "convcpp.cpp",
    "math.hpp",
  ],
)

cc_binary(
  name = "gemmcl",
  srcs = [
    "gemmcl.cpp",
  ],
  linkopts = [
    "-framework OpenCL",
  ],
)

cc_binary(
  name = "gemmcpp",
  srcs = [
    "gemmcpp.cpp",
    "layer.hpp",
    "math.hpp",
  ],
)

cc_test(
  name = "array_test",
  srcs = [
    "array_test.cpp",
  ],
  size = "small",
  deps = [
    "//external:googletest",
  ],
)
