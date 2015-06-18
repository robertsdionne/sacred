#ifndef SACRED_MEMORY_MAPPED_LOOKUP_TABLE_HPP_
#define SACRED_MEMORY_MAPPED_LOOKUP_TABLE_HPP_

#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#include "blob.hpp"
#include "lookup_table.hpp"

namespace sacred {

  using std::string;
  using std::unordered_map;
  using std::vector;

  template <typename F>
  class MemoryMappedLookupTable : public LookupTable<F> {
  public:
    MemoryMappedLookupTable(int dimension, const string &index_file, const string &vector_file) : dimension(dimension) {
      std::ifstream index_in(index_file);
      for (auto i = 0; index_in.good(); ++i) {
        string word;
        index_in >> word;
        index.insert({word, i});
        reverse_index.insert({i, word});
      }
      auto file_descriptor = open(vector_file.c_str(), O_RDONLY);
      fstat(file_descriptor, &stat_info);
      auto map = mmap(nullptr, stat_info.st_size, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, file_descriptor, 0);
      close(file_descriptor);
      memory_map = reinterpret_cast<F *>(map);
    }

    virtual ~MemoryMappedLookupTable() {
      if (memory_map) {
        munmap(memory_map, stat_info.st_size);
        memory_map = nullptr;
      }
    }

    string word_at(int i) {
      if (i < 0 || i >= reverse_index.size()) {
        return string();
      }
      return reverse_index[i];
    }

    int index_of(const string &word) {
      auto result = index.find(word);
      if (index.cend() != result) {
        return result->second;
      } else {
        return -1;
      }
    }

    F at(int index) const {
      if (index < 0 || index >= stat_info.st_size / sizeof(F)) {
        return F(0.0);
      }
      return memory_map[index];
    }

    void Forward(const vector<string> &bottom, Blob<F> *top) override {
      auto j = 0;
      for (auto &word : bottom) {
        auto word_index = index.at(word);
        for (auto i = 0; i < 300; ++i) {
          top->value().set({i, j}, at(word_index * dimension + i));
        }
        ++j;
      }
    }

    void Backward(const Blob<F> &top, const vector<string> &bottom) override {
      // TODO: Make this optional instead of unsupported.
    }

  private:
    int dimension;
    unordered_map<string, int> index;
    unordered_map<int, string> reverse_index;
    struct stat stat_info;
    F *memory_map;
  };

}  // namespace sacred

#endif  // SACRED_MEMORY_MAPPED_LOOKUP_TABLE_HPP_
