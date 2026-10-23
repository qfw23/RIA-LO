// RIA-LO analytic scan-to-map residuals.

#include "lidarOptimization.h"

#include <algorithm>
#include <cmath>

EdgeAnalyticCostFunction::EdgeAnalyticCostFunction(
    Eigen::Vector3d curr_point_in,
    Eigen::Vector3d last_point_a_in,
    Eigen::Vector3d last_point_b_in,
    double sqrt_weight_in)
    : curr_point(curr_point_in),
      last_point_a(last_point_a_in),
      last_point_b(last_point_b_in),
      sqrt_weight(sqrt_weight_in)
{
}

bool EdgeAnalyticCostFunction::Evaluate(
    double const* const* parameters,
    double* residuals,
    double** jacobians) const
{
    // Transform p_i into the map frame, then divide the
    // parallelogram area ||(p-a)x(p-b)|| by line length ||a-b||. The result is
    // the Euclidean distance from p_i to the fitted line.
    const Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
    const Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[0] + 4);
    const Eigen::Vector3d point_in_map =
        q_last_curr * curr_point + t_last_curr;

    const Eigen::Vector3d numerator =
        (point_in_map - last_point_a).cross(
            point_in_map - last_point_b);
    const Eigen::Vector3d line = last_point_a - last_point_b;
    const double line_norm = line.norm();
    const double numerator_norm = std::max(numerator.norm(), 1e-12);
    residuals[0] = sqrt_weight * numerator_norm / line_norm;

    if (jacobians != nullptr && jacobians[0] != nullptr)
    {
        // Chain rule for a left SE(3) perturbation: [-[p]_x I] maps a local
        // pose increment to point motion. The same sqrt_weight as the residual
        // is mandatory for the information-weighted objective.
        const Eigen::Matrix3d skew_point = skew(point_in_map);
        Eigen::Matrix<double, 3, 6> point_jacobian;
        point_jacobian.block<3, 3>(0, 0) = -skew_point;
        point_jacobian.block<3, 3>(0, 3).setIdentity();

        Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>
            pose_jacobian(jacobians[0]);
        pose_jacobian.setZero();
        const Eigen::Matrix3d skew_line = skew(line);
        pose_jacobian.block<1, 6>(0, 0) =
            -sqrt_weight * numerator.transpose() / numerator_norm *
            skew_line * point_jacobian / line_norm;
    }

    return true;
}

SurfNormAnalyticCostFunction::SurfNormAnalyticCostFunction(
    Eigen::Vector3d curr_point_in,
    Eigen::Vector3d plane_unit_norm_in,
    double negative_OA_dot_norm_in,
    double sqrt_weight_in)
    : curr_point(curr_point_in),
      plane_unit_norm(plane_unit_norm_in),
      negative_OA_dot_norm(negative_OA_dot_norm_in),
      sqrt_weight(sqrt_weight_in)
{
}

bool SurfNormAnalyticCostFunction::Evaluate(
    double const* const* parameters,
    double* residuals,
    double** jacobians) const
{
    // After transforming p_j to the map frame, project it onto
    // the unit normal n_j and add plane offset d_j. Keeping the sign preserves
    // the correct descent direction during pose optimization.
    const Eigen::Map<const Eigen::Quaterniond> q_w_curr(parameters[0]);
    const Eigen::Map<const Eigen::Vector3d> t_w_curr(parameters[0] + 4);
    const Eigen::Vector3d point_w =
        q_w_curr * curr_point + t_w_curr;
    residuals[0] =
        sqrt_weight *
        (plane_unit_norm.dot(point_w) + negative_OA_dot_norm);

    if (jacobians != nullptr && jacobians[0] != nullptr)
    {
        const Eigen::Matrix3d skew_point_w = skew(point_w);
        Eigen::Matrix<double, 3, 6> point_jacobian;
        point_jacobian.block<3, 3>(0, 0) = -skew_point_w;
        point_jacobian.block<3, 3>(0, 3).setIdentity();

        Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>
            pose_jacobian(jacobians[0]);
        pose_jacobian.setZero();
        pose_jacobian.block<1, 6>(0, 0) =
            sqrt_weight *
            plane_unit_norm.transpose() * point_jacobian;
    }

    return true;
}

bool PoseSE3Parameterization::Plus(
    const double* x,
    const double* delta,
    double* x_plus_delta) const
{
    // Convert the six-dimensional tangent increment xi=[omega,upsilon] into
    // Exp(xi), then left-compose it with the current world pose.
    const Eigen::Map<const Eigen::Vector3d> translation(x + 4);

    Eigen::Quaterniond delta_rotation;
    Eigen::Vector3d delta_translation;
    getTransformFromSe3(
        Eigen::Map<const Eigen::Matrix<double, 6, 1>>(delta),
        delta_rotation,
        delta_translation);

    const Eigen::Map<const Eigen::Quaterniond> rotation(x);
    Eigen::Map<Eigen::Quaterniond> updated_rotation(x_plus_delta);
    Eigen::Map<Eigen::Vector3d> updated_translation(x_plus_delta + 4);
    updated_rotation = delta_rotation * rotation;
    updated_translation =
        delta_rotation * translation + delta_translation;
    return true;
}

bool PoseSE3Parameterization::ComputeJacobian(
    const double* /*x*/,
    double* jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>>
        local_jacobian(jacobian);
    local_jacobian.topRows(6).setIdentity();
    local_jacobian.bottomRows(1).setZero();
    return true;
}

void getTransformFromSe3(
    const Eigen::Matrix<double, 6, 1>& se3,
    Eigen::Quaterniond& rotation,
    Eigen::Vector3d& translation)
{
    // Closed-form SE(3) exponential map. The small-angle branch avoids loss of
    // precision in sin(theta)/theta and in the translation left Jacobian.
    const Eigen::Vector3d omega(se3.data());
    const Eigen::Vector3d upsilon(se3.data() + 3);
    const Eigen::Matrix3d omega_skew = skew(omega);

    const double theta = omega.norm();
    const double half_theta = 0.5 * theta;
    double imaginary_factor;
    const double real_factor = std::cos(half_theta);
    if (theta < 1e-10)
    {
        const double theta_squared = theta * theta;
        const double theta_fourth = theta_squared * theta_squared;
        imaginary_factor =
            0.5 - 0.0208333 * theta_squared +
            0.000260417 * theta_fourth;
    }
    else
    {
        imaginary_factor = std::sin(half_theta) / theta;
    }

    rotation = Eigen::Quaterniond(
        real_factor,
        imaginary_factor * omega.x(),
        imaginary_factor * omega.y(),
        imaginary_factor * omega.z());

    Eigen::Matrix3d translation_jacobian;
    if (theta < 1e-10)
    {
        translation_jacobian = rotation.matrix();
    }
    else
    {
        const Eigen::Matrix3d omega_skew_squared =
            omega_skew * omega_skew;
        translation_jacobian =
            Eigen::Matrix3d::Identity() +
            (1 - std::cos(theta)) / (theta * theta) * omega_skew +
            (theta - std::sin(theta)) / std::pow(theta, 3) *
                omega_skew_squared;
    }

    translation = translation_jacobian * upsilon;
}

Eigen::Matrix3d skew(const Eigen::Vector3d& vector)
{
    Eigen::Matrix3d skew_matrix;
    skew_matrix.setZero();
    skew_matrix(0, 1) = -vector(2);
    skew_matrix(0, 2) = vector(1);
    skew_matrix(1, 2) = -vector(0);
    skew_matrix(1, 0) = vector(2);
    skew_matrix(2, 0) = -vector(1);
    skew_matrix(2, 1) = vector(0);
    return skew_matrix;
}
