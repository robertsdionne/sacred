package(
  default_visibility = [
    '//visibility:public',
  ],
)

py_library(
  name = 'conversation',
  srcs = [
    'conversation_pb2.py',
  ],
)

cc_library(
  name = 'opencl',
  srcs = [
    'opencl.pb.cc',
    'opencl.pb.h',
  ],
  deps = [
    '//google/protobuf',
  ],
)
