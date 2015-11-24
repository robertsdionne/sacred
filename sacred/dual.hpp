#ifndef SACRED_DUAL_HPP_
#define SACRED_DUAL_HPP_

#include <ostream>

namespace sacred {

struct Dual {
  constexpr Dual() : real(), dual() {}

  constexpr Dual(const float real) : real(real), dual() {}

  constexpr Dual(const float real, float dual) : real(real), dual(dual) {}

  float real, dual;
};

using std::ostream;

ostream &operator <<(ostream &out, const Dual d) {
  return out << d.real << " + " << d.dual << "_ɛ";
}

// User-defined string-literals
constexpr Dual operator "" _ɛ(unsigned long long dual) {
  return Dual(0.0, dual);
}

constexpr Dual operator "" _ɛ(long double dual) {
  return Dual(0.0, dual);
}

constexpr bool operator ==(const Dual &a, const Dual &b) {
  return a.real == b.real && a.dual == b.dual;
}

constexpr bool operator <(const Dual &a, const Dual &b) {
  return a.real < b.real;
}

constexpr bool operator !=(const Dual &a, const Dual &b) {
  return !(a == b);
}

// Negation
constexpr Dual operator -(const Dual &d) {
  return Dual(-d.real, -d.dual);
}

// Addition
constexpr Dual operator +(const Dual &a, const Dual &b) {
  return Dual(a.real + b.real, a.dual + b.dual);
}

constexpr Dual operator +(const float &a, const Dual &b) {
  return Dual(a + b.real, b.dual);
}

constexpr Dual operator +(const Dual &a, float &b) {
  return Dual(a.real + b, a.dual);
}

inline Dual &operator +=(Dual &a, const Dual &b) {
  return a = a + b;
}

// Subtraction
constexpr Dual operator -(const Dual &a, const Dual &b) {
  return Dual(a.real - b.real, a.dual - b.dual);
}

constexpr Dual operator -(const float &a, const Dual &b) {
  return Dual(a - b.real, -b.dual);
}

constexpr Dual operator -(const Dual &a, float &b) {
  return Dual(a.real - b, a.dual);
}

inline Dual &operator -=(Dual &a, const Dual &b) {
  return a = a - b;
}

// Multiplication
constexpr Dual operator *(const Dual &a, const Dual &b) {
  return Dual(a.real * b.real, a.real * b.dual + a.dual * b.real);
}

constexpr Dual operator *(const float &a, const Dual &b) {
  return Dual(a * b.real, a * b.dual);
}

constexpr Dual operator *(const Dual &a, float &b) {
  return Dual(a.real * b, a.dual * b);
}

inline Dual &operator *=(Dual &a, const Dual &b) {
  return a = a * b;
}

// Division
constexpr Dual Reciprocal(const Dual &d) {
  return Dual(1.0 / d.real, -d.dual / d.real / d.real);
}

constexpr Dual operator /(const Dual &a, const Dual &b) {
  return a * Reciprocal(b);
}

constexpr Dual operator /(const float &a, const Dual &b) {
  return a * Reciprocal(b);
}

constexpr Dual operator /(const Dual &a, float &b) {
  return Dual(a.real / b, a.dual / b);
}

inline Dual &operator /=(Dual &a, const Dual &b) {
  return a = a / b;
}

}  // namespace sacred

#endif  // SACRED_DUAL_HPP_
