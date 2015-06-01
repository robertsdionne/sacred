#include <gflags/gflags.h>
#include <fstream>
#include <iostream>
#include <limits>

#include "array.hpp"
#include "blob.hpp"
#include "nonrecursive_filter_layer.hpp"
#include "recursive_filter_layer.hpp"

using namespace sacred;
using std::ifstream;
using std::numeric_limits;

DEFINE_string(input_filename, "", "Test");

int main(int argument_count, char *arguments[]) {
  gflags::ParseCommandLineFlags(&argument_count, &arguments, true);

  auto input_filename = std::string(FLAGS_input_filename);

  ifstream in(input_filename);

  auto input = Blob<float>({1024, 1024});
  auto bias_0 = Blob<float>({1024 - 32 + 1, 1024 - 32 + 1});
  auto filter_0 = Blob<float>({32, 32});
  auto output_0 = Blob<float>({1024 - 32 + 1, 1024 - 32 + 1});
  auto bias_1 = Blob<float>({1024 - 32 + 1});
  auto filter_1 = Blob<float>({32, 32});
  auto output_1 = Blob<float>({1024 - 32 + 1, 1024 - 32 + 1});

  filter_0.value().Randomize();
  filter_1.value().Randomize();

  auto nonrecursive = NonrecursiveFilterLayer<float>(bias_0, filter_0);
  auto recursive = RecursiveFilterLayer<float>(bias_1, filter_1);

  std::cout << "reading data" << std::endl;
  in.read(reinterpret_cast<char *>(input.value().data()), 1024 * 1024 * sizeof(float));

  std::cout << "nonrecursive.Forward()" << std::endl;
  nonrecursive.Forward(input, &output_0);
  std::cout << "recursive.Forward()" << std::endl;
  recursive.Forward(output_0, &output_1);

  std::cout << "done" << std::endl;
  std::cout << output_1.value() << std::endl;

  in.close();

  return 0;
}
