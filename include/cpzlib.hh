#pragma once
#include <pdqsort.h>

#include <Eigen/Core>
#include <numeric>

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
  void permute_cols(Eigen::MatrixBase<Derived>& mat, std::vector<int>& permutation) {
    const unsigned int num_cols = mat.cols();
    unsigned int idx            = 0;
    int swap_start              = -1;
    unsigned int count          = 0;
    int permutation_idx         = -1;
    while (count < num_cols) {
      // Find the start point of the next chain of swaps
      ++swap_start;
      while (swap_start < num_cols && permutation[swap_start] < 0) {
        ++swap_start;
      }

      // Follow the chain of swaps
      idx             = swap_start;
      permutation_idx = permutation[swap_start];
      ++count;
      while (permutation_idx != swap_start) {
        mat.col(idx).swap(mat.col(permutation_idx));
        permutation[idx] = -1;
        idx              = permutation_idx;
        permutation_idx  = permutation[permutation_idx];
        ++count;
      }

      permutation[idx] = -1;
    }
  }

  template <typename Derived> auto unique_columns(const Eigen::MatrixBase<Derived>& exponents) {
    std::vector<std::vector<unsigned int>> column_groups;
    std::vector<unsigned int> unique_indices = {0};
    int curr_col                             = 0;
    std::vector<unsigned int> curr_group     = {0};
    const unsigned int num_cols              = exponents.cols();
    for (unsigned int i = 1; i < num_cols; ++i) {
      if (exponents.col(curr_col) != exponents.col(i)) {
        column_groups.push_back(curr_group);
        unique_indices.push_back(i);
        curr_group.clear();
        curr_col = i;
      }

      curr_group.push_back(i);
    }

    column_groups.push_back(curr_group);
    return std::make_pair(column_groups, exponents(Eigen::all, unique_indices));
  }

  template <typename D1, typename D2>
  auto
  regularize(const Eigen::MatrixBase<D1>& exponents, const Eigen::MatrixBase<D2>& generators) {
    const auto [column_groups, new_exponents] = unique_columns(exponents);
    const unsigned int num_groups             = column_groups.size();
    D2 new_generators(generators.rows(), num_groups);
    for (unsigned int i = 0; i < num_groups; ++i) {
      new_generators.col(i).noalias() = generators(Eigen::all, column_groups[i]).rowwise().sum();
    }

    return std::make_pair(new_exponents, new_generators);
  }

  template <typename D1, typename D2>
  void ensure_regular(Eigen::MatrixBase<D1>& exponents, Eigen::MatrixBase<D2>& generators) {
    // First, sort the exponents and generators according to the exponents
    const unsigned int num_cols = exponents.cols();
    const unsigned int num_rows = exponents.rows();
    if (num_cols == 0 || num_rows == 0) {
      return;
    }

    std::vector<int> permutation(num_cols);
    std::iota(permutation.begin(), permutation.end(), 0);
    pdqsort(permutation.begin(),
            permutation.end(),
            [&exponents, num_rows](const int i, const int j) -> bool {
              const auto col_a = exponents.col(i);
              const auto col_b = exponents.col(j);
              for (unsigned int i = 0; i < num_rows; ++i) {
                const auto val_a = col_a[i];
                const auto val_b = col_b[i];
                if (val_a != val_b) {
                  return val_a < val_b;
                }
              }

              return false;
            });

    // Then, apply the permutation in-place to sort the matrices
    // Because permute_cols modifies the permutation vector, we make a copy
    std::vector<int> permutation_copy(permutation);
    permute_cols(exponents, permutation_copy);
    permute_cols(generators, permutation);

    // Finally, check if the exponents matrix is regular and apply regularization if it is not
    if (!is_regular(exponents)) {
      std::tie(exponents, generators) = regularize(exponents, generators);
    }
  }
}  // namespace

template <typename F                  = float,
          int Dims                    = Eigen::Dynamic,
          int NumGenerators           = Eigen::Dynamic,
          int NumFactors              = Eigen::Dynamic,
          int NumConstraints          = Eigen::Dynamic,
          int NumConstraintGenerators = Eigen::Dynamic>
struct ConstrainedPolynomialZonotope {
 protected:
  inline void regularize_cpz() noexcept {
    ensure_regular(this->exponents, this->generators);
    ensure_regular(this->constraint_exponents, this->constraint_generators);
  }

  template <int Size> using Vector     = Eigen::Matrix<F, Size, 1>;
  template <int R, int C> using Matrix = Eigen::Matrix<F, R, C>;

 public:
  Vector<Dims> center;
  Matrix<Dims, NumGenerators> generators;
  Matrix<NumFactors, NumGenerators> exponents;
  Vector<NumConstraints> constraints;
  Matrix<NumConstraints, NumConstraintGenerators> constraint_generators;
  Matrix<NumFactors, NumConstraintGenerators> constraint_exponents;

  ConstrainedPolynomialZonotope(
  const Vector<Dims>& center,
  const Matrix<Dims, NumGenerators>& generators,
  const Matrix<NumFactors, NumGenerators>& exponents,
  const Vector<NumConstraints>& constraints,
  const Matrix<NumConstraints, NumConstraintGenerators>& constraint_generators,
  const Matrix<NumFactors, NumConstraintGenerators>& constraint_exponents)
  : center(center)
  , generators(generators)
  , exponents(exponents)
  , constraints(constraints)
  , constraint_generators(constraint_generators)
  , constraint_exponents(constraint_exponents) {
    regularize_cpz();
  }

  ConstrainedPolynomialZonotope(
  const Vector<Dims>&& center,
  const Matrix<Dims, NumGenerators>&& generators,
  const Matrix<NumFactors, NumGenerators>&& exponents,
  const Vector<NumConstraints>&& constraints,
  const Matrix<NumConstraints, NumConstraintGenerators>&& constraint_generators,
  const Matrix<NumFactors, NumConstraintGenerators>&& constraint_exponents)
  : center(center)
  , generators(generators)
  , exponents(exponents)
  , constraints(constraints)
  , constraint_generators(constraint_generators)
  , constraint_exponents(constraint_exponents) {
    regularize_cpz();
  }
};
}  // namespace cpz
