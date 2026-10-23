
// Analytic residuals and SE(3) local parameterization used by RIA-LO.

#ifndef RIA_LO_LIDAR_OPTIMIZATION_H_
#define RIA_LO_LIDAR_OPTIMIZATION_H_

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

void getTransformFromSe3(
    const Eigen::Matrix<double, 6, 1>& se3,
    Eigen::Quaterniond& q,
    Eigen::Vector3d& t);

Eigen::Matrix3d skew(const Eigen::Vector3d& vector);

// Weighted point-to-line residual.
// curr_point is p_i, last_point_a/b define the fitted map line, and
// sqrt_weight is sqrt(w_i^e). Multiplying both residual and Jacobian by
// sqrt(w_i^e) makes the squared Ceres term equal to w_i^e*r_i^2.
class EdgeAnalyticCostFunction : public ceres::SizedCostFunction<1, 7>
{
public:
    EdgeAnalyticCostFunction(
        Eigen::Vector3d curr_point,
        Eigen::Vector3d last_point_a,
        Eigen::Vector3d last_point_b,
        double sqrt_weight = 1.0);
    ~EdgeAnalyticCostFunction() override = default;

    bool Evaluate(
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override;

private:
    Eigen::Vector3d curr_point;
    Eigen::Vector3d last_point_a;
    Eigen::Vector3d last_point_b;
    double sqrt_weight;
};

// Weighted signed point-to-plane
// residual. plane_unit_norm is n_j and negative_OA_dot_norm is d_j in
// n_j^T(Rp_j+t)+d_j; sqrt_weight is sqrt(w_j^s).
class SurfNormAnalyticCostFunction : public ceres::SizedCostFunction<1, 7>
{
public:
    SurfNormAnalyticCostFunction(
        Eigen::Vector3d curr_point,
        Eigen::Vector3d plane_unit_norm,
        double negative_OA_dot_norm,
        double sqrt_weight = 1.0);
    ~SurfNormAnalyticCostFunction() override = default;

    bool Evaluate(
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override;

private:
    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm;
    double sqrt_weight;
};

// Left-multiplicative SE(3) local parameterization.
class PoseSE3Parameterization : public ceres::LocalParameterization
{
public:
    PoseSE3Parameterization() = default;
    ~PoseSE3Parameterization() override = default;

    bool Plus(
        const double* x,
        const double* delta,
        double* x_plus_delta) const override;
    bool ComputeJacobian(
        const double* x,
        double* jacobian) const override;
    int GlobalSize() const override { return 7; }
    int LocalSize() const override { return 6; }
};

#endif  // RIA_LO_LIDAR_OPTIMIZATION_H_
