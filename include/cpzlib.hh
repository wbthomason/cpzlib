#pragma once
#include <pdqsort.h>
#include <robin_hood.h>

#include <Eigen/Core>
#include <functional>
#include <numeric>
#include <type_traits>

namespace cpz {
namespace {
  template <typename Derived> bool is_regular(const Eigen::MatrixBase<Derived>& exponents) {
    const unsigned int num_cols = exponents.cols();
    for (unsigned int i = 0; i < num_cols - 1; ++i) {
      const auto& col = exponents.col(i);
      if (col.isZero()) {
        return false;
      }

      if (col == exponents.col(i + 1)) {
        return false;
      }
    }

    return true;
  }

  template <typename Derived>
  void permute_cols(Eigen::MatrixBase<Derived>& mat,
                    const std::vector<int>& permutation,
                    const std::vector<int>& inverse_permutation) {
    const auto num_cols  = mat.cols();
    auto placeholder_col = mat.col(0);
    int idx              = 0;
    int last_unswapped   = 0;
    while (last_unswapped < num_cols) {
      const auto permutation_idx = permutation[idx];
      if (permutation_idx != idx) {
        placeholder_col = mat.col(idx);
        mat.col(idx)    = mat.col(permutation_idx);
      } else {
        ++last_unswapped;
      }
    }
  }

  template <typename D1, typename D2>
  void regularize(Eigen::MatrixBase<D1>& exponents, Eigen::MatrixBase<D2>& generators) {}
  template <typename D1, typename D2>
  void ensure_regular(Eigen::MatrixBase<D1>& exponents, Eigen::MatrixBase<D2>& generators) {
    // First, sort the exponents and generators according to the exponents
    std::vector<int> permutation;
    const auto num_cols = exponents.cols();
    permutation.reserve(num_cols);
    std::iota(permutation.begin(), permutation.end() + num_cols, 0);
    pdqsort(permutation.begin(),
            permutation.end(),
            [&exponents](const auto& i, const auto& j) -> bool {
              return exponents.col(i) < exponents.col(j);
            });


    std::vector<int> inverse_permutation(num_cols);
    for (int i = 0; i < num_cols; ++i) {
      inverse_permutation[permutation[i]] = i;
    }

    permute_cols(exponents, permutation, inverse_permutation);
    permute_cols(generators, permutation, inverse_permutation);
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
