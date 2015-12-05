package(
  default_visibility = [
    '//visibility:public',
  ],
)

cc_library(
  name = 'conversation_cc',
  srcs = [
    'conversation.pb.cc',
    'conversation.pb.h',
  ],
  deps = [
    '//google/protobuf',
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
