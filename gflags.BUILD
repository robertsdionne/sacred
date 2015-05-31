cc_library(
    name = "gflags",
    srcs = glob([
      "gflags/include/gflags/*.h",
      "gflags/src/*.cc",
      "gflags/src/*.h",
    ], exclude = [
      "gflags/src/windows_*",
    ]),
    hdrs = glob([
      "gflags/include/gflags/*.h",
      "gflags/src/*.h",
    ]),
    visibility = [
      "//visibility:public",
    ],
    includes = [
      "gflags/include",
      "gflags/include/gflags",
    ],
    linkopts = [
      "-lpthread",
    ],
)
