#pragma once
#include <pdqsort.h>
#include <robin_hood.h>

#include <Eigen/Core>
#include <functional>
#include <type_traits>

namespace cpz {
namespace {
  // Adapted from:
  // https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
  template <typename M> std::size_t matrix_hash(const M& matrix) {
    std::size_t result = 0;
    for (auto i = 0; i < matrix.size(); ++i) {
      result ^= std::hash<typename M::Scalar>()(*matrix.data() + i) + 0x9e3779b9 + (result << 6) +
                (result >> 2);
    }

    return result;
  }

  template <typename Derived> inline bool is_regular(const Eigen::MatrixBase<Derived>& exponents) {
    robin_hood::unordered_set<std::size_t> column_hashes;
    column_hashes.reserve(exponents.cols());
    for (auto& col : exponents.colwise()) {
      if (col.isZero()) {
        return false;
      }

      if (!column_hashes.insert(matrix_hash(col)).second) {
        return false;
      };
    }

    return true;
  }

  template <typename D1, typename D2>
  void regularize(Eigen::MatrixBase<D1>& exponents, Eigen::MatrixBase<D2>& generators) {}
  template <typename D1, typename D2>
  inline void ensure_regular(Eigen::MatrixBase<D1>& exponents, Eigen::MatrixBase<D2>& generators) {
    if (!is_regular(exponents)) {
      regularize(exponents, generators);
    }
  }
}  // namespace

template <typename F = float> struct ConstrainedPolynomialZonotope {
  using Vector = Eigen::Matrix<F, Eigen::Dynamic, 1>;
  using Matrix = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic>;
  Vector center;
  Matrix generators;
  Matrix exponents;
  Vector constraints;
  Matrix constraint_generators;
  Matrix constraint_exponents;

  ConstrainedPolynomialZonotope(const Vector& center,
                                const Matrix& generators,
                                const Matrix& exponents,
                                const Vector& constraints,
                                const Matrix& constraint_generators,
                                const Matrix& constraint_exponents)
  : center(center)
  , generators(generators)
  , exponents(exponents)
  , constraints(constraints)
  , constraint_generators(constraint_generators)
  , constraint_exponents(constraint_exponents) {
    ensure_regular(this->exponents, this->generators);
    ensure_regular(this->constraint_exponents, this->constraint_generators);
  }

  ConstrainedPolynomialZonotope(const Vector&& center,
                                const Matrix&& generators,
                                const Matrix&& exponents,
                                const Vector&& constraints,
                                const Matrix&& constraint_generators,
                                const Matrix&& constraint_exponents)
  : center(center)
  , generators(generators)
  , exponents(exponents)
  , constraints(constraints)
  , constraint_generators(constraint_generators)
  , constraint_exponents(constraint_exponents) {
    ensure_regular(this->exponents, this->generators);
    ensure_regular(this->constraint_exponents, this->constraint_generators);
  }

  ConstrainedPolynomialZonotope(ConstrainedPolynomialZonotope&& o) = default;
};
}  // namespace cpz
