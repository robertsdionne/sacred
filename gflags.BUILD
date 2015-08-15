cc_library(
    name = "gflags",
    srcs = glob([
      "include/gflags/*.h",
      "src/*.cc",
      "src/*.h",
    ], exclude = [
      "src/windows_*",
    ]),
    hdrs = glob([
      "include/gflags/*.h",
      "src/*.h",
    ]),
    visibility = [
      "//visibility:public",
    ],
    includes = [
      "include",
      "include/gflags",
    ],
    linkopts = [
      "-lpthread",
    ],
)
