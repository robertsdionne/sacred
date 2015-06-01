#include <cstdio>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <limits>
#include <string>

DEFINE_string(type, u8"float", "The type format of the input file.");
DEFINE_int32(block_size, 4096, "The number of items to max over.");

template <typename F>
void ReadAndPrint(const std::string &filename) {
  std::ifstream in(filename);
  auto out = popen("spark", "w");
  if (nullptr == out) {
    std::cerr << "ERROR: Unable to pipe to spark." << std::endl;
    out = stdout;
  }
  F buffer[FLAGS_block_size];
  while (in) {
    auto maximum = F(0.0);
    in.read(reinterpret_cast<char *>(buffer), FLAGS_block_size * sizeof(F));
    for (auto i = 0; i < in.gcount() / sizeof(F); ++i) {
      maximum = std::max(maximum, buffer[i]);
    }
    int value = 256 * maximum;
    fprintf(out, "%d\n", value);
  }
  pclose(out);
}

template <char>
void ReadAndPrint(const std::string &filename) {
  std::ifstream in(filename);
  auto out = popen("spark", "w");
  if (nullptr == out) {
    std::cout << "ERROR: Unable to pipe to spark." << std::endl;
    out = stdout;
  }
  char buffer[FLAGS_block_size];
  while (in) {
    auto maximum = 0;
    in.read(buffer, FLAGS_block_size);
    for (auto i = 0; i < in.gcount(); ++i) {
      maximum = std::max(static_cast<int>(maximum), static_cast<int>(buffer[i]));
    }
    int value = maximum;
    fprintf(out, "%d", value);
  }
  pclose(out);
}

int main(int argument_count, char *arguments[]) {
  gflags::ParseCommandLineFlags(&argument_count, &arguments, true);

  if (argument_count < 2) {
    std::cerr << "ERROR: Please supply a filename to read." << std::endl;
    return 1;
  }

  std::string filename = arguments[1];

  if (u8"char" == FLAGS_type) {
    ReadAndPrint<char>(filename);
  } else if (u8"float" == FLAGS_type) {
    ReadAndPrint<float>(filename);
  } else if (u8"double" == FLAGS_type) {
    ReadAndPrint<double>(filename);
  } else if (u8"long double" == FLAGS_type) {
    ReadAndPrint<long double>(filename);
  } else {
    std::cerr << "ERROR: " << FLAGS_type << " is an unknown type format." << std::endl;
  }
  return 0;
}
