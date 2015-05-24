#ifndef SACRED_DUAL_HPP_
#define SACRED_DUAL_HPP_

namespace sacred {

  template <typename F>
  struct Dual {
    constexpr Dual() = default;

    constexpr Dual(const F real) : real(real), dual() {}

    constexpr Dual(const F real, F dual) : real(real), dual(dual) {}

    template <typename G>
    constexpr operator Dual<G>() const {
      return Dual<G>(static_cast<G>(real), static_cast<G>(dual));
    }

    F real, dual;
  };

  // User-defined string-literals
  constexpr Dual<long double> operator "" _ɛ(unsigned long long dual) {
    return Dual<long double>(0.0, dual);
  }

  constexpr Dual<long double> operator "" _ɛ(long double dual) {
    return Dual<long double>(0.0, dual);
  }

  // Negation
  template <typename F>
  constexpr Dual<F> operator -(const Dual<F> &d) {
    return Dual<F>(-d.real, -d.dual);
  }

  // Addition
  template <typename F, typename G>
  constexpr Dual<decltype(F() + G())> operator +(const Dual<F> &a, const Dual<G> &b) {
    return Dual<decltype(F() + G())>(a.real + b.real, a.dual + b.dual);
  }

  template <typename F, typename G>
  constexpr Dual<decltype(F() + G())> operator +(const F &a, const Dual<G> &b) {
    return Dual<decltype(F() + G())>(a + b.real, b.dual);
  }

  template <typename F, typename G>
  constexpr Dual<decltype(F() + G())> operator +(const Dual<F> &a, G &b) {
    return Dual<decltype(F() + G())>(a.real + b, a.dual);
  }

  // Subtraction
  template <typename F, typename G>
  constexpr Dual<decltype(F() - G())> operator -(const Dual<F> &a, const Dual<G> &b) {
    return Dual<decltype(F() - G())>(a.real - b.real, a.dual - b.dual);
  }

  template <typename F, typename G>
  constexpr Dual<decltype(F() - G())> operator -(const F &a, const Dual<G> &b) {
    return Dual<decltype(F() - G())>(a - b.real, -b.dual);
  }

  template <typename F, typename G>
  constexpr Dual<decltype(F() - G())> operator -(const Dual<F> &a, G &b) {
    return Dual<decltype(F() - G())>(a.real - b, a.dual);
  }

  // Multiplication
  template <typename F, typename G>
  constexpr Dual<decltype(F() * G())> operator *(const Dual<F> &a, const Dual<G> &b) {
    return Dual<decltype(F() * G())>(a.real * b.real, a.real * b.dual + a.dual * b.real);
  }

  template <typename F, typename G>
  constexpr Dual<decltype(F() * G())> operator *(const F &a, const Dual<G> &b) {
    return Dual<decltype(F() * G())>(a * b.real, a * b.dual);
  }

  template <typename F, typename G>
  constexpr Dual<decltype(F() * G())> operator *(const Dual<F> &a, G &b) {
    return Dual<decltype(F() * G())>(a.real * b, a.dual * b);
  }

  // Division
  template <typename F>
  constexpr Dual<F> Reciprocal(const Dual<F> &d) {
    return Dual<F>(1.0 / d.real, -d.dual / d.real / d.real);
  }

  template <typename F, typename G>
  constexpr Dual<decltype(F() / G())> operator /(const Dual<F> &a, const Dual<G> &b) {
    return a * Reciprocal(b);
  }

  template <typename F, typename G>
  constexpr Dual<decltype(F() / G())> operator /(const F &a, const Dual<G> &b) {
    return a * Reciprocal(b);
  }

  template <typename F, typename G>
  constexpr Dual<decltype(F() / G())> operator /(const Dual<F> &a, G &b) {
    return Dual<decltype(F() / G())>(a.real / b, a.dual / b);
  }

}  // namespace sacred

#endif  // SACRED_DUAL_HPP_
