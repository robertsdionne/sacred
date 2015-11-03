py_binary(
  name = 'group_consecutive_utterances',
  srcs = [
    'group_consecutive_utterances.py',
  ],
)

sh_binary(
  name = 'group_consecutive_utterances_sh',
  srcs = [
    'group_consecutive_utterances.sh',
  ],
)

cc_binary(
  name = 'print_floats',
  srcs = [
    'print_floats.cpp',
  ],
  deps = [
    '//external:googleflags',
  ],
)

py_binary(
  name = 'prepare_data',
  srcs = [
    'prepare_data.py',
  ],
)

py_binary(
  name = 'proto3',
  srcs = [
    'proto3.py',
  ],
  deps = [
    '//sacred/proto:conversation',
  ],
)
