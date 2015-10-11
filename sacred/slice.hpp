#ifndef SACRED_SLICE_HPP_
#define SACRED_SLICE_HPP_

namespace sacred {

  class Slice {
  public:
    Slice() {}

    Slice(int start):
        start_empty_(false), start_(start), stop_(start + 1), step_(1) {}

    Slice(int start, int stop, int step=1):
        start_empty_(false), stop_empty_(false), start_(start), stop_(stop), step_(step) {}

    Slice(void *start, int stop, int step=1):
        stop_empty_(false), stop_(stop), step_(step) {}

    Slice(int start, void *stop, int step):
        start_empty_(false), start_(start), step_(step) {}

    Slice(void *start, void *stop, int step):
        step_(step) {}

    ~Slice() = default;

    bool has_start() {
      return !start_empty_;
    }

    bool has_stop() {
      return !stop_empty_;
    }

    int start(int n) {
      if (start_empty_) {
        if (step_ < 0) {
          return n;
        }
        return 0;
      }
      return start_;
    }

    int stop(int n) {
      if (stop_empty_) {
        if (step_ < 0) {
          return -1;
        }
        return n;
      }
      return stop_;
    }

    int step() {
      return step_;
    }

  private:
    bool start_empty_ = true, stop_empty_ = true;
    int start_, stop_, step_ = 1;
  };

}  // namespace sacred

#endif  // SACRED_SLICE_HPP_
