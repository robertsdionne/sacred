#ifndef SACRED_MEMORY_MAPPED_LOOKUP_TABLE_HPP_
#define SACRED_MEMORY_MAPPED_LOOKUP_TABLE_HPP_

#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "blob.hpp"
#include "lookup_table.hpp"

namespace sacred {

  using std::string;

  template <typename F>
  class MemoryMappedLookupTable : public LookupTable<F> {
  public:
    MemoryMappedLookupTable(const std::string &file_to_map) {
      auto file_descriptor = open(file_to_map.c_str(), O_RDONLY);
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

    F at(int index) const {
      if (index >= stat_info.st_size / sizeof(F)) {
        return F(0.0);
      }
      return memory_map[index];
    }

    void Forward(const vector<string> &bottom, Blob<F> *top) override {
    }

    void Backward(const Blob<F> &top, const vector<string> &bottom) override {
    }

  private:
    struct stat stat_info;
    F *memory_map;
  };

}  // namespace sacred

#endif  // SACRED_MEMORY_MAPPED_LOOKUP_TABLE_HPP_
