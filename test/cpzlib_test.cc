#include <Eigen/Core>
#include <cpzlib.hh>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 1) {
    std::cout << argv[0] << " takes no arguments.\n";
    return 1;
  }

  Eigen::Matrix<float, 3, 2> generators;
  Eigen::Matrix<float, 3, 1> center;
  Eigen::Matrix<float, 2, 2> exponents;
  Eigen::Matrix<float, 0, 1> constraints;
  Eigen::Matrix<float, 1, 0> constraint_generators;
  Eigen::Matrix<float, 2, 0> constraint_exponents;
  cpz::ConstrainedPolynomialZonotope<> z(
  center, generators, exponents, constraints, constraint_generators, constraint_exponents);
}
