#pragma once
#include <pdqsort.h>
#include <robin_hood.h>

#include <Eigen/Core>
#include <functional>
#include <numeric>
#include <type_traits>

namespace cpz {
namespace {
  template <typename Derived> inline bool is_regular(const Eigen::MatrixBase<Derived>& exponents) {
    auto next_col       = 1;
    const auto num_cols = exponents.cols();
    for (auto& col : exponents.colwise()) {
      if (col.isZero()) {
        return false;
      }

      if (next_col < num_cols && col == exponents.col(next_col)) {
        return false;
      }
    }

    return true;
  }

  template <typename D1, typename D2>
  void regularize(Eigen::MatrixBase<D1>& exponents, Eigen::MatrixBase<D2>& generators) {}
  template <typename D1, typename D2>
  void ensure_regular(Eigen::MatrixBase<D1>& exponents, Eigen::MatrixBase<D2>& generators) {
    // First, sort the exponents and generators according to the exponents
    const int num_cols = exponents.cols();
    std::vector<int> permutation;
    permutation.reserve(exponents.cols());
    for (int i = 0; i < num_cols; ++i) {
      permutation.push_back(i);
    }

    pdqsort(permutation.begin(),
            permutation.end(),
            [&exponents](const auto& i, const auto& j) -> bool {
              return exponents.col(i) < exponents.col(j);
            });

    // TODO: Apply the permutation in place if possible

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
