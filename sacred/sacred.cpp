#include <gflags/gflags.h>
#include <fstream>
#include <iostream>
#include <limits>

#include "array.hpp"

using namespace sacred;
using std::ifstream;
using std::numeric_limits;

DEFINE_string(input_filename, "", "Test");

int main(int argument_count, char *arguments[]) {
  gflags::ParseCommandLineFlags(&argument_count, &arguments, true);

  auto input_filename = std::string(FLAGS_input_filename);

  ifstream in(input_filename);

  float floats[4 * 1024];
  while (in) {
    float minimum = numeric_limits<float>::max(), maximum = numeric_limits<float>::min();
    in.read(reinterpret_cast<char *>(floats), 4 * 1024 * sizeof(float));
    for (auto i = 0; i < 4 * 1024; ++i) {
      minimum = std::min(floats[i], minimum);
      maximum = std::max(floats[i], maximum);
    }
    int value = 256 * maximum;
    std::cout << value << std::endl;
  }
  in.close();

  return 0;
}
