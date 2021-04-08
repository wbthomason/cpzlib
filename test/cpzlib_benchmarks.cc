#include <celero/Celero.h>

#include <Eigen/Core>
#include <cpzlib.hh>
#include <random>

CELERO_MAIN

// TODO: Add test setup for matrices that definitely need regularization
class BasicFixture : public celero::TestFixture {
 public:
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> generators;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> center;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> exponents;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> constraints;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> constraint_generators;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> constraint_exponents;
  void setUp(const celero::TestFixture::ExperimentValue&) override {
    std::mt19937 gen(0);
    std::uniform_int_distribution<> dist(1, 50);
    const int dim             = dist(gen);
    const int num_gens        = dist(gen);
    const int num_coeffs      = dist(gen);
    const int num_constrs     = dist(gen);
    const int num_constr_gens = dist(gen);

    center     = Eigen::Matrix<float, Eigen::Dynamic, 1>::Random(dim);
    generators = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Random(dim, num_gens);
    exponents = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Random(num_coeffs, num_gens);
    constraints = Eigen::Matrix<float, Eigen::Dynamic, 1>::Random(num_constrs);
    constraint_generators =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Random(num_constrs, num_constr_gens);
    constraint_exponents =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Random(num_coeffs, num_constr_gens);
  }
};

BASELINE_F(Regularization, Baseline, BasicFixture, 1000, 0) {
  cpz::ConstrainedPolynomialZonotope<> z(
  center, generators, exponents, constraints, constraint_generators, constraint_exponents);
}
