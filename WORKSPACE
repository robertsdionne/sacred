
new_local_repository(
  name = 'gflags',
  path = '/Users/robertsdionne/Code/gflags',
  build_file = 'BUILD.gflags',
)

new_local_repository(
  name = 'glog',
  path = '/Users/robertsdionne/Code/glog',
  build_file = 'BUILD.glog',
)

bind(
  name = 'googleflags',
  actual = '@gflags//:gflags',
)

bind(
  name = 'googlelog',
  actual = '@glog//:glog',
)
