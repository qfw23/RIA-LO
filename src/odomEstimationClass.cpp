// RIA-LO core scan-to-map estimator.

#include "odomEstimationClass.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <utility>
#include <unordered_map>

#include <ros/ros.h>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr int kMotionPhaseBins = 512;
constexpr float kAssociationRadius = 1.0f;
constexpr float kAssociationCellSize = 2.0f;
constexpr float kAssociationRadiusSquared =
    kAssociationRadius * kAssociationRadius;

// Exact correspondence index: a 2 m cell bounds each 1 m-radius query to at
// most eight hash buckets. The first high-iteration frame uses the initial
// map KD-tree; steady frames use the incrementally maintained voxel index.

// Residual-domain polar grid.
constexpr int kResidualAzimuthCells = 72;
constexpr int kResidualRangeCells = 24;
constexpr double kResidualRangeStep = 5.0;

// Conservative free-space visibility grid for map-evidence confirmation.
constexpr int kVisibilityAzimuthCells = 360;
constexpr int kVisibilityElevationCells = 16;
constexpr double kVisibilityElevationMin = -30.0 * kPi / 180.0;
constexpr double kVisibilityElevationMax = 10.0 * kPi / 180.0;

// Event-driven local-map maintenance thresholds.
constexpr int kMaintenanceFrameLimit = 4;
constexpr double kMaintenanceTravelMeters = 4.0;
constexpr std::size_t kMapHardLimit = 190000;
constexpr double kFreeSpaceMarginMeters = 1.25;
constexpr double kEvidenceWeightFloor = 0.30;

template <typename T>
T clampValue(T value, T lower, T upper)
{
    return std::max(lower, std::min(value, upper));
}

Eigen::Matrix<double, 6, 1> poseLog(const Eigen::Isometry3d& pose)
{
    // Inverse of getTransformFromSe3: recover xi=[omega,upsilon] from a
    // relative pose. Motion compensation scales it by each point's phase offset.
    const Eigen::AngleAxisd angle_axis(pose.rotation());
    const double theta = angle_axis.angle();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    if (std::isfinite(theta) && std::abs(theta) > 1e-12)
        omega = angle_axis.axis() * theta;

    const Eigen::Matrix3d omega_skew = skew(omega);
    Eigen::Matrix3d translation_jacobian = Eigen::Matrix3d::Identity();
    if (std::abs(theta) < 1e-5)
    {
        translation_jacobian +=
            0.5 * omega_skew +
            (1.0 / 6.0) * omega_skew * omega_skew;
    }
    else
    {
        const double theta_squared = theta * theta;
        translation_jacobian +=
            (1.0 - std::cos(theta)) / theta_squared * omega_skew +
            (theta - std::sin(theta)) /
                (theta_squared * theta) * omega_skew * omega_skew;
    }

    Eigen::Matrix<double, 6, 1> tangent;
    tangent.head<3>() = omega;
    tangent.tail<3>() =
        translation_jacobian.colPivHouseholderQr().solve(pose.translation());
    return tangent;
}

double median(std::vector<double> values)
{
    if (values.empty())
        return 0.0;

    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2 != 0)
        return upper;

    std::nth_element(
        values.begin(), values.begin() + middle - 1, values.begin() + middle);
    return 0.5 * (values[middle - 1] + upper);
}

double pointStaticProbability(const PointType& point)
{
    // A symmetric Beta(3, 3) prior prevents a few early
    // observations from dominating the persistent map-point probability.
    const double support = static_cast<double>(point.support);
    const double conflict = static_cast<double>(point.conflict);
    return clampValue(
        (support + 3.0) / (support + conflict + 6.0),
        0.10,
        0.99);
}

float saturatedIncrement(float value, int increment = 1)
{
    return std::min(255.0f, value + static_cast<float>(increment));
}

int visibilityCell(const Eigen::Vector3d& point)
{
    const double range = point.norm();
    if (range < 1e-6)
        return -1;

    const double azimuth = std::atan2(point.y(), point.x());
    const double elevation = std::asin(
        clampValue(point.z() / range, -1.0, 1.0));
    if (elevation < kVisibilityElevationMin ||
        elevation >= kVisibilityElevationMax)
    {
        return -1;
    }

    int azimuth_index = static_cast<int>(
        (azimuth + kPi) / (2.0 * kPi) * kVisibilityAzimuthCells);
    azimuth_index =
        clampValue(azimuth_index, 0, kVisibilityAzimuthCells - 1);
    int elevation_index = static_cast<int>(
        (elevation - kVisibilityElevationMin) /
        (kVisibilityElevationMax - kVisibilityElevationMin) *
        kVisibilityElevationCells);
    elevation_index =
        clampValue(elevation_index, 0, kVisibilityElevationCells - 1);
    return elevation_index * kVisibilityAzimuthCells + azimuth_index;
}
}  // namespace

OdomEstimationClass::OdomEstimationClass()
    : odom(Eigen::Isometry3d::Identity()),
      frame_id(0),
      frame_candidate_constraints(0),
      frame_direction_bins(0),
      frame_residual_median(0.0),
      frame_residual_scale(0.0),
      frame_weighting_active(true),
      last_odom(Eigen::Isometry3d::Identity()),
      frames_since_maintenance(0),
      last_maintenance_position(Eigen::Vector3d::Zero()),
      robust_loss_delta(0.1),
      execution_mode(ExecutionMode::Full),
      adaptive_weighting_enabled(false),
      weighting_enter_constraints(900),
      weighting_exit_constraints(1300),
      weighting_enter_direction_bins(8),
      weighting_exit_direction_bins(10)
{
}

std::size_t OdomEstimationClass::AssociationVoxelKeyHash::operator()(
    const AssociationVoxelKey& key) const
{
    std::size_t seed = std::hash<int>()(key.x);
    seed ^= std::hash<int>()(key.y) + 0x9e3779b9U +
        (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<int>()(key.z) + 0x9e3779b9U +
        (seed << 6U) + (seed >> 2U);
    return seed;
}

OdomEstimationClass::AssociationVoxelKey
OdomEstimationClass::associationVoxelKey(const PointType& point) const
{
    AssociationVoxelKey key;
    key.x = static_cast<int>(std::floor(point.x / kAssociationCellSize));
    key.y = static_cast<int>(std::floor(point.y / kAssociationCellSize));
    key.z = static_cast<int>(std::floor(point.z / kAssociationCellSize));
    return key;
}

void OdomEstimationClass::insertAssociationPoint(
    const PointType& point,
    int point_index,
    AssociationVoxelIndex& index)
{
    if (!std::isfinite(point.x) ||
        !std::isfinite(point.y) ||
        !std::isfinite(point.z))
    {
        return;
    }
    index[associationVoxelKey(point)].push_back(point_index);
}

void OdomEstimationClass::rebuildAssociationIndex(
    const pcl::PointCloud<PointType>::Ptr& map,
    AssociationVoxelIndex& index)
{
    index.clear();
    index.reserve(map->size() / 16 + 1);
    for (int i = 0; i < static_cast<int>(map->size()); ++i)
        insertAssociationPoint(map->points[i], i, index);
}

int OdomEstimationClass::radiusNearestSearch(
    const PointType& query,
    const pcl::PointCloud<PointType>::Ptr& map,
    const AssociationVoxelIndex& index,
    std::array<int, kAssociationSearchCount>& indices,
    std::array<float, kAssociationSearchCount>& squared_distances,
    float& outside_distance) const
{
    // Enumerate every voxel intersecting the one-meter search
    // ball, keep the nearest candidates in deterministic (distance,index)
    // order, and return a conservative lower bound for every excluded point.
    using RankedPoint = std::pair<float, int>;
    std::array<RankedPoint, kAssociationSearchCount> nearest;
    int found = 0;
    const int min_x = static_cast<int>(std::floor(
        (query.x - kAssociationRadius) / kAssociationCellSize));
    const int max_x = static_cast<int>(std::floor(
        (query.x + kAssociationRadius) / kAssociationCellSize));
    const int min_y = static_cast<int>(std::floor(
        (query.y - kAssociationRadius) / kAssociationCellSize));
    const int max_y = static_cast<int>(std::floor(
        (query.y + kAssociationRadius) / kAssociationCellSize));
    const int min_z = static_cast<int>(std::floor(
        (query.z - kAssociationRadius) / kAssociationCellSize));
    const int max_z = static_cast<int>(std::floor(
        (query.z + kAssociationRadius) / kAssociationCellSize));

    // Lower bound for every point outside the visited voxel block. Together
    // with the nearest rejected point, it supports certified negative reuse.
    const float lower_x = query.x - min_x * kAssociationCellSize;
    const float upper_x = (max_x + 1) * kAssociationCellSize - query.x;
    const float lower_y = query.y - min_y * kAssociationCellSize;
    const float upper_y = (max_y + 1) * kAssociationCellSize - query.y;
    const float lower_z = query.z - min_z * kAssociationCellSize;
    const float upper_z = (max_z + 1) * kAssociationCellSize - query.z;
    const float unvisited_voxel_distance = std::min(
        std::min(std::min(lower_x, upper_x), std::min(lower_y, upper_y)),
        std::min(lower_z, upper_z));
    float nearest_rejected_squared_distance =
        std::numeric_limits<float>::infinity();
    const auto better = [](const RankedPoint& left, const RankedPoint& right)
    {
        if (left.first != right.first)
            return left.first < right.first;
        return left.second < right.second;
    };

    for (int x = min_x; x <= max_x; ++x)
    {
        for (int y = min_y; y <= max_y; ++y)
        {
            for (int z = min_z; z <= max_z; ++z)
            {
                AssociationVoxelKey key;
                key.x = x;
                key.y = y;
                key.z = z;
                const auto bucket = index.find(key);
                if (bucket == index.end())
                    continue;

                for (int point_index : bucket->second)
                {
                    if (point_index < 0 ||
                        point_index >= static_cast<int>(map->size()))
                    {
                        continue;
                    }
                    const PointType& point = map->points[point_index];
                    const float point_dx = query.x - point.x;
                    const float point_dy = query.y - point.y;
                    const float point_dz = query.z - point.z;
                    const float squared_distance =
                        point_dx * point_dx +
                        point_dy * point_dy +
                        point_dz * point_dz;
                    if (!(squared_distance < kAssociationRadiusSquared))
                    {
                        nearest_rejected_squared_distance = std::min(
                            nearest_rejected_squared_distance,
                            squared_distance);
                        continue;
                    }

                    const RankedPoint candidate(
                        squared_distance, point_index);
                    int position = 0;
                    if (found < static_cast<int>(kAssociationSearchCount))
                    {
                        position = found;
                        nearest[found++] = candidate;
                    }
                    else
                    {
                        position = found - 1;
                        if (!better(candidate, nearest[position]))
                            continue;
                        nearest[position] = candidate;
                    }
                    while (position > 0 &&
                           better(nearest[position], nearest[position - 1]))
                    {
                        std::swap(nearest[position], nearest[position - 1]);
                        --position;
                    }
                }
            }
        }
    }

    for (int i = 0; i < found; ++i)
    {
        squared_distances[i] = nearest[i].first;
        indices[i] = nearest[i].second;
    }
    outside_distance = std::min(
        unvisited_voxel_distance,
        std::sqrt(nearest_rejected_squared_distance));
    return found;
}

void OdomEstimationClass::init(
    lidar::Lidar lidar_param,
    double map_resolution_in,
    const std::string& pipeline_mode_in)
{
    if (pipeline_mode_in == "base")
        execution_mode = ExecutionMode::Base;
    else if (pipeline_mode_in == "weighted")
        execution_mode = ExecutionMode::Weighted;
    else if (pipeline_mode_in == "full_global")
        execution_mode = ExecutionMode::FullGlobal;
    else if (pipeline_mode_in == "full")
        execution_mode = ExecutionMode::Full;
    else
    {
        ROS_WARN(
            "Unknown RIA-LO pipeline mode '%s'; using full",
            pipeline_mode_in.c_str());
        execution_mode = ExecutionMode::Full;
    }

    laserCloudCornerMap.reset(new pcl::PointCloud<PointType>());
    laserCloudSurfMap.reset(new pcl::PointCloud<PointType>());

    map_resolution = static_cast<float>(map_resolution_in);
    downSizeFilterEdge.setLeafSize(
        map_resolution, map_resolution, map_resolution);
    downSizeFilterSurf.setLeafSize(
        map_resolution * surface_resolution_scale,
        map_resolution * surface_resolution_scale,
        map_resolution * surface_resolution_scale);
    downSizeFilterEdge.setDownsampleAllData(true);
    downSizeFilterSurf.setDownsampleAllData(true);

    kdtreeEdgeMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfMap.reset(new pcl::KdTreeFLANN<PointType>());
    croppedCornerScratch.reset(new pcl::PointCloud<PointType>());
    croppedSurfScratch.reset(new pcl::PointCloud<PointType>());
    compactedCornerScratch.reset(new pcl::PointCloud<PointType>());
    compactedSurfScratch.reset(new pcl::PointCloud<PointType>());
    currentEdgeScratch.reset(new pcl::PointCloud<PointType>());
    currentSurfScratch.reset(new pcl::PointCloud<PointType>());
    deskewedEdgeScratch.reset(new pcl::PointCloud<PointType>());
    deskewedSurfScratch.reset(new pcl::PointCloud<PointType>());
    scan_period = std::max(1e-3, lidar_param.scan_period);
    odom = Eigen::Isometry3d::Identity();
    last_odom = Eigen::Isometry3d::Identity();
    frames_since_maintenance = 0;
    last_maintenance_position = Eigen::Vector3d::Zero();
    motion_turn_activity = 0.0;
    motion_turn_duty = 0.0;
    edge_association_index.clear();
    surf_association_index.clear();
    optimization_count = steady_optimization_count;
}

void OdomEstimationClass::configureAccuracyProfile(
    double surface_resolution_scale_in,
    int optimization_iterations,
    int ceres_max_iterations_in,
    double robust_delta_scale_multiplier_in,
    double robust_delta_min_in,
    double robust_delta_max_in,
    double direction_balance_exponent_in,
    double direction_weight_min_in,
    double direction_weight_max_in)
{
    surface_resolution_scale = clampValue(
        surface_resolution_scale_in, 1.0, 3.0);
    steady_optimization_count = clampValue(
        optimization_iterations, 1, 4);
    ceres_max_iterations = clampValue(
        ceres_max_iterations_in, 1, 12);
    robust_delta_scale_multiplier = clampValue(
        robust_delta_scale_multiplier_in, 0.5, 3.0);
    robust_delta_min = clampValue(
        robust_delta_min_in, 0.01, 0.30);
    robust_delta_max = clampValue(
        robust_delta_max_in, robust_delta_min + 0.001, 0.50);
    direction_balance_exponent = clampValue(
        direction_balance_exponent_in, 0.0, 1.5);
    direction_weight_min = clampValue(
        direction_weight_min_in, 0.20, 1.0);
    direction_weight_max = clampValue(
        direction_weight_max_in, 1.0, 3.0);
}

void OdomEstimationClass::configureScanMotionCompensation(
    bool enabled,
    double reference_phase,
    double translation_scale,
    double rotation_scale,
    int warmup_frames)
{
    scan_motion_compensation = enabled;
    motion_reference_phase = clampValue(reference_phase, 0.0, 1.0);
    motion_translation_scale =
        clampValue(translation_scale, 0.0, 1.5);
    motion_rotation_scale =
        clampValue(rotation_scale, 0.0, 1.5);
    motion_compensation_warmup_frames = std::max(0, warmup_frames);
}

void OdomEstimationClass::configureAdaptiveMotionScaling(
    bool enabled,
    double high_turn_translation_scale_in,
    double high_turn_rotation_scale_in,
    double turn_start,
    double turn_end,
    double smoothing,
    double duty_start,
    double duty_end,
    double duty_smoothing)
{
    adaptive_motion_scaling = enabled;
    high_turn_translation_scale =
        clampValue(high_turn_translation_scale_in, 0.0, 1.5);
    high_turn_rotation_scale =
        clampValue(high_turn_rotation_scale_in, 0.0, 1.5);
    motion_turn_start = clampValue(turn_start, 0.0, 0.20);
    motion_turn_end = clampValue(
        turn_end, motion_turn_start + 1e-4, 0.35);
    motion_turn_smoothing = clampValue(smoothing, 0.01, 1.0);
    motion_turn_duty_start = clampValue(duty_start, 0.0, 0.95);
    motion_turn_duty_end = clampValue(
        duty_end, motion_turn_duty_start + 1e-3, 1.0);
    motion_turn_duty_smoothing =
        clampValue(duty_smoothing, 0.001, 1.0);
}

void OdomEstimationClass::configureAdaptiveWeighting(
    bool enabled,
    int enter_constraints,
    int exit_constraints,
    int enter_direction_bins,
    int exit_direction_bins)
{
    adaptive_weighting_enabled = enabled;
    weighting_enter_constraints = std::max(1, enter_constraints);
    weighting_exit_constraints =
        std::max(weighting_enter_constraints + 1, exit_constraints);
    weighting_enter_direction_bins =
        clampValue(enter_direction_bins, 0, 12);
    weighting_exit_direction_bins = clampValue(
        std::max(weighting_enter_direction_bins, exit_direction_bins),
        0,
        12);
    // Optional hysteresis starts on the lightweight path and enables complete
    // correspondence weighting when the current frame is weakly constrained.
    frame_weighting_active = !adaptive_weighting_enabled;
}

bool OdomEstimationClass::weightingEnabled() const
{
    return execution_mode != ExecutionMode::Base;
}

bool OdomEstimationClass::weightingActive() const
{
    return weightingEnabled() &&
        (!adaptive_weighting_enabled || frame_weighting_active);
}

bool OdomEstimationClass::fullFactorPathEnabled() const
{
    return execution_mode == ExecutionMode::FullGlobal ||
        execution_mode == ExecutionMode::Full;
}

bool OdomEstimationClass::geometryReuseEnabled() const
{
    return execution_mode == ExecutionMode::Full;
}

void OdomEstimationClass::initializePointState(
    pcl::PointCloud<PointType>::Ptr& cloud) const
{
    for (PointType& point : cloud->points)
    {
        point.support = 2;
        point.conflict = 0;
    }
}

void OdomEstimationClass::initMapWithPoints(
    const pcl::PointCloud<PointType>::Ptr& edge_in,
    const pcl::PointCloud<PointType>::Ptr& surf_in)
{
    // The first feature frame defines the initial local map and identity pose.
    // Full mode also creates the incremental association index used from the
    // next frame onward; no optimization is performed on this seed frame.
    *laserCloudCornerMap = *edge_in;
    *laserCloudSurfMap = *surf_in;
    initializePointState(laserCloudCornerMap);
    initializePointState(laserCloudSurfMap);
    if (geometryReuseEnabled())
    {
        rebuildAssociationIndex(
            laserCloudCornerMap, edge_association_index);
        rebuildAssociationIndex(
            laserCloudSurfMap, surf_association_index);
        kdtreeEdgeMap->setInputCloud(laserCloudCornerMap);
        kdtreeSurfMap->setInputCloud(laserCloudSurfMap);
    }
    last_maintenance_position = Eigen::Vector3d::Zero();
    frames_since_maintenance = 0;
    optimization_count = std::max(12, steady_optimization_count);
}

void OdomEstimationClass::updatePointsToMap(
    const pcl::PointCloud<PointType>::Ptr& edge_in,
    const pcl::PointCloud<PointType>::Ptr& surf_in)
{
    // One complete estimator transaction. All frame-local statistics are
    // rebuilt here, while persistent map evidence changes only after the final
    // optimization pass. This boundary prevents intermediate nonlinear poses
    // from leaking into later frames.
    ++frame_id;
    if (optimization_count > steady_optimization_count)
        --optimization_count;

    // Constant-velocity prediction supplies the next scan-to-map initial pose.
    const Eigen::Isometry3d relative_motion =
        last_odom.inverse() * odom;
    const Eigen::Isometry3d odom_prediction =
        odom * relative_motion;
    last_odom = odom;
    odom = odom_prediction;
    q_w_curr = Eigen::Quaterniond(odom.rotation());
    t_w_curr = odom.translation();

    currentEdgeScratch->clear();
    currentSurfScratch->clear();
    pcl::PointCloud<PointType>::Ptr& downsampled_edge = currentEdgeScratch;
    pcl::PointCloud<PointType>::Ptr& downsampled_surf = currentSurfScratch;
    pcl::PointCloud<PointType>::Ptr processed_edge = edge_in;
    pcl::PointCloud<PointType>::Ptr processed_surf = surf_in;
    frame_motion_translation = relative_motion.translation().norm();
    frame_motion_rotation =
        Eigen::AngleAxisd(relative_motion.rotation()).angle();
    if (frame_id == 1)
    {
        motion_turn_activity = frame_motion_rotation;
    }
    else
    {
        motion_turn_activity =
            (1.0 - motion_turn_smoothing) * motion_turn_activity +
            motion_turn_smoothing * frame_motion_rotation;
    }
    frame_motion_turn_instant = 0.0;
    frame_motion_turn_blend = 0.0;
    if (adaptive_motion_scaling)
    {
        // Sustained turning, rather than one isolated angular increment,
        // controls the compensation scale. Two smoothstep filters avoid an
        // abrupt deskew regime switch around the configured thresholds.
        const double normalized_turn = clampValue(
            (motion_turn_activity - motion_turn_start) /
                (motion_turn_end - motion_turn_start),
            0.0,
            1.0);
        frame_motion_turn_instant =
            normalized_turn * normalized_turn *
            (3.0 - 2.0 * normalized_turn);
        motion_turn_duty =
            (1.0 - motion_turn_duty_smoothing) * motion_turn_duty +
            motion_turn_duty_smoothing * frame_motion_turn_instant;
        const double normalized_duty = clampValue(
            (motion_turn_duty - motion_turn_duty_start) /
                (motion_turn_duty_end - motion_turn_duty_start),
            0.0,
            1.0);
        frame_motion_turn_blend =
            normalized_duty * normalized_duty *
            (3.0 - 2.0 * normalized_duty);
    }
    frame_motion_translation_scale =
        motion_translation_scale +
        frame_motion_turn_blend *
            (high_turn_translation_scale - motion_translation_scale);
    frame_motion_rotation_scale =
        motion_rotation_scale +
        frame_motion_turn_blend *
            (high_turn_rotation_scale - motion_rotation_scale);
    const bool finite_motion =
        relative_motion.matrix().allFinite() &&
        std::isfinite(frame_motion_translation) &&
        std::isfinite(frame_motion_rotation);
    const bool plausible_motion =
        finite_motion &&
        frame_motion_translation <= 5.0 &&
        std::abs(frame_motion_rotation) <= 0.35;
    if (scan_motion_compensation &&
        frame_id > motion_compensation_warmup_frames &&
        plausible_motion)
    {
        compensateScanMotion(
            edge_in,
            surf_in,
            relative_motion,
            deskewedEdgeScratch,
            deskewedSurfScratch);
        processed_edge = deskewedEdgeScratch;
        processed_surf = deskewedSurfScratch;
    }
    downSamplingToMap(
        processed_edge,
        downsampled_edge,
        processed_surf,
        downsampled_surf);
    initializePointState(downsampled_edge);
    initializePointState(downsampled_surf);

    // Reset reliability fields before constructing this
    // frame's rough correspondences. residual_cell_history is intentionally
    // retained as reversible state across frames.
    frame_candidate_constraints = 0;
    frame_direction_bins = 0;
    frame_residual_median = 0.0;
    frame_residual_scale = 0.0;
    rough_observations.clear();
    if (fullFactorPathEnabled())
    {
        // Candidate neighborhoods are frame-local because the map is fixed
        // only during this frame's outer scan-to-map iterations.
        edge_geometry_cache.assign(
            downsampled_edge->size(), LocalGeometry());
        surf_geometry_cache.assign(
            downsampled_surf->size(), LocalGeometry());
    }
    else
    {
        edge_geometry_cache.clear();
        surf_geometry_cache.clear();
    }
    residual_cell_weights.clear();
    direction_balance_weights.clear();
    robust_loss_delta = 0.1;

    if (weightingEnabled())
    {
        prepareEvidenceVotes(
            edge_evidence_votes, laserCloudCornerMap->size());
        prepareEvidenceVotes(
            surf_evidence_votes, laserCloudSurfMap->size());
    }

    if (laserCloudCornerMap->size() > 10 &&
        laserCloudSurfMap->size() > 50)
    {
        if (!geometryReuseEnabled())
        {
            kdtreeEdgeMap->setInputCloud(laserCloudCornerMap);
            kdtreeSurfMap->setInputCloud(laserCloudSurfMap);
        }

        // Steady operation uses two scan-to-map passes for the
        // same robust objective. The startup transient adds weighted refinement
        // passes and decreases to two; only the final pass commits state.
        for (int iteration = 0; iteration < optimization_count; ++iteration)
        {
            const bool rough_pass = iteration == 0;
            const bool final_pass = iteration == optimization_count - 1;
            ceres::Problem problem;
            problem.AddParameterBlock(
                parameters, 7, new PoseSE3Parameterization());

            ceres::LossFunction* loss_function =
                new ceres::HuberLoss(robust_loss_delta);
            if (fullFactorPathEnabled())
            {
                addReusableWeightedEdgeCostFactor(
                    downsampled_edge,
                    laserCloudCornerMap,
                    problem,
                    loss_function,
                    rough_pass,
                    final_pass);
                addReusableWeightedSurfCostFactor(
                    downsampled_surf,
                    laserCloudSurfMap,
                    problem,
                    loss_function,
                    rough_pass,
                    final_pass);
            }
            else if (weightingEnabled())
            {
                addWeightedEdgeCostFactor(
                    downsampled_edge,
                    laserCloudCornerMap,
                    problem,
                    loss_function,
                    rough_pass,
                    final_pass);
                addWeightedSurfCostFactor(
                    downsampled_surf,
                    laserCloudSurfMap,
                    problem,
                    loss_function,
                    rough_pass,
                    final_pass);
            }
            else
            {
                addBaselineEdgeCostFactor(
                    downsampled_edge,
                    laserCloudCornerMap,
                    problem,
                    loss_function,
                    rough_pass);
                addBaselineSurfCostFactor(
                    downsampled_surf,
                    laserCloudSurfMap,
                    problem,
                    loss_function,
                    rough_pass);
            }

            ceres::Solver::Options options;
            options.linear_solver_type = execution_mode == ExecutionMode::Weighted
                ? ceres::DENSE_QR
                : ceres::DENSE_NORMAL_CHOLESKY;
            options.max_num_iterations = ceres_max_iterations;
            options.num_threads = 1;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            ceres::Solver::Summary summary;
            // Ceres updates q_w_curr/t_w_curr in place. The rough pass solves
            // ordinary geometry first; its converged pose then freezes the
            // statistics used by all later weighted passes in this frame.
            ceres::Solve(options, &problem, &summary);

            if (rough_pass && weightingEnabled())
            {
                // The fixed rough pose estimates robust
                // residual statistics and reliability terms. Later passes
                // consume them without relabeling intermediate nonlinear poses.
                buildResidualConsistencyField();
            }
        }
    }
    else
    {
        ROS_WARN_THROTTLE(
            2.0, "RIA-LO map has too few points for association");
    }

    // Materialize the optimized parameter block as the public odometry pose
    // before transforming accepted features into the map.
    odom = Eigen::Isometry3d::Identity();
    odom.linear() = q_w_curr.normalized().toRotationMatrix();
    odom.translation() = t_w_curr;

    if (weightingEnabled())
    {
        // Commit only final-pass votes so intermediate poses cannot pollute
        // persistent map evidence.
        updateActiveEvidence(laserCloudCornerMap, edge_evidence_votes);
        updateActiveEvidence(laserCloudSurfMap, surf_evidence_votes);
    }
    addPointsToMap(downsampled_edge, downsampled_surf);
}

void OdomEstimationClass::pointAssociateToMap(
    PointType const* const pi,
    PointType* const po) const
{
    const Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    const Eigen::Vector3d point_w =
        q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->intensity = pi->intensity;
    po->support = pi->support;
    po->conflict = pi->conflict;
}

void OdomEstimationClass::downSamplingToMap(
    const pcl::PointCloud<PointType>::Ptr& edge_pc_in,
    pcl::PointCloud<PointType>::Ptr& edge_pc_out,
    const pcl::PointCloud<PointType>::Ptr& surf_pc_in,
    pcl::PointCloud<PointType>::Ptr& surf_pc_out)
{
    downSizeFilterEdge.setInputCloud(edge_pc_in);
    downSizeFilterEdge.filter(*edge_pc_out);
    downSizeFilterSurf.setInputCloud(surf_pc_in);
    downSizeFilterSurf.filter(*surf_pc_out);
}

void OdomEstimationClass::compensateScanMotion(
    const pcl::PointCloud<PointType>::Ptr& edge_input,
    const pcl::PointCloud<PointType>::Ptr& surf_input,
    const Eigen::Isometry3d& relative_motion,
    pcl::PointCloud<PointType>::Ptr& edge_output,
    pcl::PointCloud<PointType>::Ptr& surf_output)
{
    Eigen::Matrix<double, 6, 1> motion = poseLog(relative_motion);
    motion.head<3>() *= frame_motion_rotation_scale;
    motion.tail<3>() *= frame_motion_translation_scale;

    // Apply the previous relative SE(3) motion at each point's
    // offset from the common reference phase. Nearby phases share a lookup.
    std::array<Eigen::Matrix3d, kMotionPhaseBins> phase_rotations;
    std::array<Eigen::Vector3d, kMotionPhaseBins> phase_translations;
    for (int bin = 0; bin < kMotionPhaseBins; ++bin)
    {
        const double phase =
            static_cast<double>(bin) /
            static_cast<double>(kMotionPhaseBins - 1);
        Eigen::Quaterniond phase_rotation;
        getTransformFromSe3(
            (phase - motion_reference_phase) * motion,
            phase_rotation,
            phase_translations[bin]);
        phase_rotations[bin] = phase_rotation.toRotationMatrix();
    }

    const auto compensate_cloud =
        [&](const pcl::PointCloud<PointType>::Ptr& input,
            pcl::PointCloud<PointType>::Ptr& output)
    {
        output->clear();
        output->reserve(input->size());
        for (const PointType& point : input->points)
        {
            const double scan_line = std::floor(
                static_cast<double>(point.intensity) + 1e-6);
            const double phase = clampValue(
                (static_cast<double>(point.intensity) - scan_line) /
                    scan_period,
                0.0,
                1.0);
            const int phase_bin = clampValue(
                static_cast<int>(std::lround(
                    phase *
                    static_cast<double>(kMotionPhaseBins - 1))),
                0,
                kMotionPhaseBins - 1);

            const Eigen::Vector3d raw(point.x, point.y, point.z);
            const Eigen::Vector3d corrected =
                phase_rotations[phase_bin] * raw +
                phase_translations[phase_bin];
            PointType compensated = point;
            compensated.x = static_cast<float>(corrected.x());
            compensated.y = static_cast<float>(corrected.y());
            compensated.z = static_cast<float>(corrected.z());
            output->push_back(compensated);
        }
        output->width = static_cast<std::uint32_t>(output->size());
        output->height = 1;
        output->is_dense = input->is_dense;
    };

    compensate_cloud(edge_input, edge_output);
    compensate_cloud(surf_input, surf_output);
}

int OdomEstimationClass::polarCell(const Eigen::Vector3d& point) const
{
    // Group residuals by sensor-frame azimuth and range, not by
    // world position. Moving objects therefore remain in a coherent local
    // observation region while the platform travels.
    const double range = point.head<2>().norm();
    if (range < 1e-6)
        return -1;

    int azimuth_index = static_cast<int>(
        (std::atan2(point.y(), point.x()) + kPi) /
        (2.0 * kPi) * kResidualAzimuthCells);
    azimuth_index =
        clampValue(azimuth_index, 0, kResidualAzimuthCells - 1);
    const int range_index = clampValue(
        static_cast<int>(range / kResidualRangeStep),
        0,
        kResidualRangeCells - 1);
    return range_index * kResidualAzimuthCells + azimuth_index;
}

int OdomEstimationClass::geometricDirectionCell(
    bool edge,
    const Eigen::Vector3d& direction) const
{
    // Use an unsigned coarse orientation label: the dominant and
    // secondary axes identify one of nine bins for each feature type. Opposite
    // normals/line directions remain equivalent geometric constraints.
    if (direction.squaredNorm() < 1e-12)
        return -1;

    Eigen::Vector3d magnitudes = direction.normalized().cwiseAbs();
    Eigen::Index primary = 0;
    magnitudes.maxCoeff(&primary);
    magnitudes[primary] = -1.0;
    Eigen::Index secondary = 0;
    magnitudes.maxCoeff(&secondary);
    return (edge ? 9 : 0) +
        static_cast<int>(primary) * 3 +
        static_cast<int>(secondary);
}

double OdomEstimationClass::residualCellWeight(
    const Eigen::Vector3d& point) const
{
    const auto found = residual_cell_weights.find(polarCell(point));
    if (found == residual_cell_weights.end())
        return 1.0;
    return found->second;
}

double OdomEstimationClass::directionBalanceWeight(
    bool edge,
    const Eigen::Vector3d& direction) const
{
    const auto found = direction_balance_weights.find(
        geometricDirectionCell(edge, direction));
    return found == direction_balance_weights.end() ? 1.0 : found->second;
}

double OdomEstimationClass::mapEvidenceWeight(
    const pcl::PointCloud<PointType>::Ptr& map,
    const std::array<int, 5>& indices) const
{
    // Map-point support/conflict yields a bounded stability
    // probability; the five-neighbor median rejects one anomalous map point.
    std::array<double, 5> weights;
    for (std::size_t position = 0; position < indices.size(); ++position)
    {
        const int index = indices[position];
        const PointType& point = map->points[index];
        const double explicit_evidence =
            static_cast<double>(point.support + point.conflict);
        if (explicit_evidence < 8.0)
        {
            weights[position] = 1.0;
            continue;
        }
        const double probability = pointStaticProbability(point);
        weights[position] = clampValue(
            (probability - 0.25) / 0.60,
            kEvidenceWeightFloor,
            1.0);
    }
    std::sort(weights.begin(), weights.end());
    return weights[2];
}

void OdomEstimationClass::inheritPointState(
    PointType& point,
    const pcl::PointCloud<PointType>::Ptr& map,
    const std::array<int, 5>& indices) const
{
    double support = 0.0;
    double conflict = 0.0;
    int valid = 0;
    for (int index : indices)
    {
        if (index < 0 || index >= static_cast<int>(map->size()))
            continue;
        support += map->points[index].support;
        conflict += map->points[index].conflict;
        ++valid;
    }
    if (valid == 0)
        return;

    point.support = static_cast<float>(
        clampValue(support / valid, 2.0, 255.0));
    point.conflict = static_cast<float>(
        clampValue(conflict / valid, 0.0, 255.0));
}

double OdomEstimationClass::observationResidual(
    const ResidualObservation& observation,
    Eigen::Vector3d* innovation) const
{
    // Re-evaluate point-to-line or point-to-plane
    // innovation at one fixed rough pose before constructing reliability.
    const Eigen::Vector3d point_w =
        q_w_curr * observation.curr_point + t_w_curr;
    if (observation.edge)
    {
        const Eigen::Vector3d direction =
            observation.model_b - observation.model_a;
        const double denominator = direction.squaredNorm();
        if (denominator < 1e-12)
        {
            *innovation = Eigen::Vector3d::Zero();
            return 0.0;
        }
        const double alpha =
            (point_w - observation.model_a).dot(direction) / denominator;
        const Eigen::Vector3d projection =
            observation.model_a + alpha * direction;
        *innovation = point_w - projection;
    }
    else
    {
        const double signed_distance =
            observation.model_a.dot(point_w) + observation.model_b.x();
        *innovation = signed_distance * observation.model_a;
    }
    return innovation->norm();
}

void OdomEstimationClass::buildResidualConsistencyField()
{
    // Reliability construction is deliberately separated from Ceres:
    // 1) evaluate every rough residual at one common pose;
    // 2) estimate frame scale and directional occupancy;
    // 3) aggregate spatially coherent residual tails;
    // 4) update reversible history and expose weights to the final pass.
    struct CellStats
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int count = 0;
        int tail_count = 0;
        double residual_sum = 0.0;
        double normalized_excess_sum = 0.0;
        Eigen::Vector3d innovation_sum = Eigen::Vector3d::Zero();
    };

    std::vector<double> residuals;
    residuals.reserve(rough_observations.size());
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        innovations;
    innovations.reserve(rough_observations.size());

    for (const ResidualObservation& observation : rough_observations)
    {
        Eigen::Vector3d innovation;
        const double residual =
            observationResidual(observation, &innovation);
        residuals.push_back(residual);
        innovations.push_back(innovation);
    }

    frame_candidate_constraints = static_cast<int>(residuals.size());
    if (residuals.empty())
        return;

    // Median/MAD gives the frame center and robust scale; the
    // same scale defines the residual-tail threshold and adaptive Huber delta.
    frame_residual_median = median(residuals);
    std::vector<double> deviations;
    deviations.reserve(residuals.size());
    for (double residual : residuals)
        deviations.push_back(std::abs(residual - frame_residual_median));
    frame_residual_scale = std::max(0.015, 1.4826 * median(deviations));
    const double tail_threshold = std::max(
        0.12, frame_residual_median + 3.0 * frame_residual_scale);
    robust_loss_delta = clampValue(
        frame_residual_median +
            robust_delta_scale_multiplier * frame_residual_scale,
        robust_delta_min,
        robust_delta_max);

    // Bounded inverse occupancy balances valid orientation
    // bins so dominant ground planes do not overwhelm complementary geometry.
    std::unordered_map<int, int> direction_counts;
    int balanced_observations = 0;
    for (std::size_t index = 0;
         index < rough_observations.size();
         ++index)
    {
        if (residuals[index] > tail_threshold)
            continue;
        const ResidualObservation& observation =
            rough_observations[index];
        const Eigen::Vector3d direction = observation.edge
            ? observation.model_b - observation.model_a
            : observation.model_a;
        const int direction_cell =
            geometricDirectionCell(observation.edge, direction);
        if (direction_cell < 0)
            continue;
        ++direction_counts[direction_cell];
        ++balanced_observations;
    }
    if (!direction_counts.empty())
    {
        const double target_count =
            static_cast<double>(balanced_observations) /
            static_cast<double>(direction_counts.size());
        for (const auto& entry : direction_counts)
        {
            direction_balance_weights[entry.first] = clampValue(
                std::pow(
                    target_count /
                        static_cast<double>(std::max(entry.second, 1)),
                    direction_balance_exponent),
                direction_weight_min,
                direction_weight_max);
        }
    }
    frame_direction_bins =
        static_cast<int>(direction_balance_weights.size());

    if (adaptive_weighting_enabled)
    {
        if (frame_candidate_constraints <= weighting_enter_constraints ||
            frame_direction_bins <= weighting_enter_direction_bins)
        {
            frame_weighting_active = true;
        }
        else if (
            frame_candidate_constraints >= weighting_exit_constraints &&
            frame_direction_bins >= weighting_exit_direction_bins)
        {
            frame_weighting_active = false;
        }
    }

    std::unordered_map<int, CellStats> cells;
    for (std::size_t index = 0; index < rough_observations.size(); ++index)
    {
        const int cell = rough_observations[index].polar_cell;
        if (cell < 0)
            continue;
        CellStats& stats = cells[cell];
        ++stats.count;
        stats.residual_sum += residuals[index];
        stats.innovation_sum += innovations[index];
        const double normalized_excess = std::max(
            0.0,
            (residuals[index] - tail_threshold) /
                std::max(frame_residual_scale, 0.025));
        stats.normalized_excess_sum += normalized_excess;
        if (residuals[index] > tail_threshold)
            ++stats.tail_count;
    }

    // Cell aggregation is essential: one large residual is insufficient.
    // A cell is downweighted only when many points lie in the robust tail and
    // their innovation vectors agree.
    std::unordered_map<int, double> current_inconsistency_scores;
    for (const auto& entry : cells)
    {
        const CellStats& stats = entry.second;
        if (stats.count < 10)
            continue;

        const double tail_fraction =
            static_cast<double>(stats.tail_count) /
            static_cast<double>(stats.count);
        const double coherence =
            stats.innovation_sum.norm() /
            std::max(stats.residual_sum, 1e-9);
        const double excess = clampValue(
            stats.normalized_excess_sum /
                static_cast<double>(stats.count) / 2.0,
            0.0,
            1.0);
        const double consensus = clampValue(
            tail_fraction / 0.45,
            0.0,
            1.0);
        // Tail excess, tail support, and innovation coherence
        // jointly form the residual-cell inconsistency score.
        const double dynamic_probability =
            excess * consensus * (0.45 + 0.55 * coherence);
        if (dynamic_probability < 0.45)
            continue;

        current_inconsistency_scores[entry.first] = dynamic_probability;
        double& history = residual_cell_history[entry.first];
        history = 0.55 * history + 0.45 * dynamic_probability;
    }

    // Reversible temporal filtering suppresses isolated spikes;
    // an unobserved or recovered cell decays back toward unit weight.
    for (auto iterator = residual_cell_history.begin();
         iterator != residual_cell_history.end();)
    {
        const bool observed_now =
            current_inconsistency_scores.find(iterator->first) !=
            current_inconsistency_scores.end();
        if (!observed_now)
            iterator->second *= 0.65;
        if (iterator->second < 0.05)
        {
            iterator = residual_cell_history.erase(iterator);
            continue;
        }
        if (iterator->second >= 0.50)
        {
            const double confirmed_probability = clampValue(
                (iterator->second - 0.35) / 0.65,
                0.0,
                1.0);
            residual_cell_weights[iterator->first] = clampValue(
                1.0 - 0.85 * confirmed_probability,
                0.15,
                1.0);
        }
        ++iterator;
    }

    if (adaptive_weighting_enabled && !frame_weighting_active)
    {
        robust_loss_delta = 0.1;
        residual_cell_weights.clear();
        direction_balance_weights.clear();
    }
}

void OdomEstimationClass::prepareEvidenceVotes(
    SparseEvidenceVotes& votes,
    std::size_t map_size)
{
    if (votes.values.size() < map_size)
        votes.values.resize(map_size);
    if (votes.epochs.size() < map_size)
        votes.epochs.resize(map_size, 0);
    votes.touched_indices.clear();
}

void OdomEstimationClass::recordMapVotes(
    const ResidualObservation& observation,
    bool static_observation)
{
    // Final-pass residual classification contributes support or
    // conflict only to the three closest map correspondences. Sparse epoch
    // storage makes repeated observations accumulate without clearing the map.
    SparseEvidenceVotes& votes =
        observation.edge ? edge_evidence_votes : surf_evidence_votes;
    for (int neighbor = 0; neighbor < 3; ++neighbor)
    {
        const int index = observation.map_indices[neighbor];
        if (index < 0 ||
            index >= static_cast<int>(votes.values.size()))
        {
            continue;
        }
        if (votes.epochs[index] != static_cast<std::uint32_t>(frame_id))
        {
            votes.epochs[index] = static_cast<std::uint32_t>(frame_id);
            votes.values[index] = EvidenceVotes();
            votes.touched_indices.push_back(index);
        }
        EvidenceVotes& vote = votes.values[index];
        std::uint16_t& value =
            static_observation ? vote.support : vote.conflict;
        value = static_cast<std::uint16_t>(
            std::min(255, static_cast<int>(value) + 1));
    }
}

void OdomEstimationClass::addBaselineEdgeCostFactor(
    const pcl::PointCloud<PointType>::Ptr& pc_in,
    const pcl::PointCloud<PointType>::Ptr& map_in,
    ceres::Problem& problem,
    ceres::LossFunction* loss_function,
    bool rough_pass)
{
    std::vector<int> indices(5);
    std::vector<float> squared_distances(5);
    for (int i = 0; i < static_cast<int>(pc_in->size()); ++i)
    {
        PointType point_temp;
        pointAssociateToMap(&pc_in->points[i], &point_temp);
        if (kdtreeEdgeMap->nearestKSearch(
                point_temp, 5, indices, squared_distances) < 5 ||
            squared_distances[4] >= 1.0f)
        {
            continue;
        }

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        std::array<int, 5> map_indices;
        std::array<Eigen::Vector3d, 5> neighbors;
        for (int j = 0; j < 5; ++j)
        {
            map_indices[j] = indices[j];
            const PointType& point = map_in->points[indices[j]];
            neighbors[j] = Eigen::Vector3d(point.x, point.y, point.z);
            center += neighbors[j];
        }
        center /= 5.0;

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const Eigen::Vector3d& point : neighbors)
        {
            const Eigen::Vector3d centered = point - center;
            covariance += centered * centered.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() != Eigen::Success ||
            solver.eigenvalues()[2] <= 3.0 * solver.eigenvalues()[1])
        {
            continue;
        }

        const Eigen::Vector3d direction = solver.eigenvectors().col(2);
        const Eigen::Vector3d point_a = center + 0.1 * direction;
        const Eigen::Vector3d point_b = center - 0.1 * direction;
        const Eigen::Vector3d current(
            pc_in->points[i].x,
            pc_in->points[i].y,
            pc_in->points[i].z);

        if (rough_pass)
            ++frame_candidate_constraints;

        ceres::CostFunction* cost_function =
            new EdgeAnalyticCostFunction(
                current,
                point_a,
                point_b,
                1.0);
        problem.AddResidualBlock(
            cost_function, loss_function, parameters);
    }
}

void OdomEstimationClass::addBaselineSurfCostFactor(
    const pcl::PointCloud<PointType>::Ptr& pc_in,
    const pcl::PointCloud<PointType>::Ptr& map_in,
    ceres::Problem& problem,
    ceres::LossFunction* loss_function,
    bool rough_pass)
{
    std::vector<int> indices(5);
    std::vector<float> squared_distances(5);
    for (int i = 0; i < static_cast<int>(pc_in->size()); ++i)
    {
        PointType point_temp;
        pointAssociateToMap(&pc_in->points[i], &point_temp);
        if (kdtreeSurfMap->nearestKSearch(
                point_temp, 5, indices, squared_distances) < 5 ||
            squared_distances[4] >= 1.0f)
        {
            continue;
        }

        Eigen::Matrix<double, 5, 3> matrix_a;
        std::array<int, 5> map_indices;
        for (int j = 0; j < 5; ++j)
        {
            map_indices[j] = indices[j];
            const PointType& point = map_in->points[indices[j]];
            matrix_a(j, 0) = point.x;
            matrix_a(j, 1) = point.y;
            matrix_a(j, 2) = point.z;
        }
        const Eigen::Matrix<double, 5, 1> matrix_b =
            -Eigen::Matrix<double, 5, 1>::Ones();
        Eigen::Vector3d normal =
            matrix_a.colPivHouseholderQr().solve(matrix_b);
        const double normal_norm = normal.norm();
        if (!std::isfinite(normal_norm) || normal_norm < 1e-12)
            continue;
        const double offset = 1.0 / normal_norm;
        normal /= normal_norm;

        bool valid_plane = true;
        for (int index : map_indices)
        {
            const PointType& point = map_in->points[index];
            if (std::abs(
                    normal.x() * point.x +
                    normal.y() * point.y +
                    normal.z() * point.z + offset) > 0.2)
            {
                valid_plane = false;
                break;
            }
        }
        if (!valid_plane)
            continue;

        const Eigen::Vector3d current(
            pc_in->points[i].x,
            pc_in->points[i].y,
            pc_in->points[i].z);
        if (rough_pass)
            ++frame_candidate_constraints;

        ceres::CostFunction* cost_function =
            new SurfNormAnalyticCostFunction(
                current,
                normal,
                offset,
                1.0);
        problem.AddResidualBlock(
            cost_function, loss_function, parameters);
    }
}

// Weighted execution path without geometry reuse.
void OdomEstimationClass::addWeightedEdgeCostFactor(
    const pcl::PointCloud<PointType>::Ptr& pc_in,
    const pcl::PointCloud<PointType>::Ptr& map_in,
    ceres::Problem& problem,
    ceres::LossFunction* loss_function,
    bool rough_pass,
    bool final_pass)
{
    std::vector<int> indices(5);
    std::vector<float> squared_distances(5);
    for (int i = 0; i < static_cast<int>(pc_in->size()); ++i)
    {
        PointType point_temp;
        pointAssociateToMap(&pc_in->points[i], &point_temp);
        if (kdtreeEdgeMap->nearestKSearch(
                point_temp, 5, indices, squared_distances) < 5 ||
            squared_distances[4] >= 1.0f)
        {
            continue;
        }

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        std::array<int, 5> map_indices;
        for (int j = 0; j < 5; ++j)
        {
            map_indices[j] = indices[j];
            center += Eigen::Vector3d(
                map_in->points[indices[j]].x,
                map_in->points[indices[j]].y,
                map_in->points[indices[j]].z);
        }
        center /= 5.0;

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (int index : map_indices)
        {
            const Eigen::Vector3d point(
                map_in->points[index].x,
                map_in->points[index].y,
                map_in->points[index].z);
            const Eigen::Vector3d centered = point - center;
            covariance += centered * centered.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        const Eigen::Vector3d eigenvalues = solver.eigenvalues();
        const double largest_eigenvalue =
            std::max(eigenvalues[2], 1e-9);
        // Line quality from ordered covariance eigenvalues.
        const double linearity =
            (eigenvalues[2] - eigenvalues[1]) / largest_eigenvalue;
        if (linearity <= 2.0 / 3.0)
            continue;

        const Eigen::Vector3d direction = solver.eigenvectors().col(2);
        const Eigen::Vector3d point_a = center + 0.1 * direction;
        const Eigen::Vector3d point_b = center - 0.1 * direction;
        const Eigen::Vector3d current(
            pc_in->points[i].x,
            pc_in->points[i].y,
            pc_in->points[i].z);

        ResidualObservation observation;
        observation.edge = true;
        observation.polar_cell = polarCell(current);
        observation.curr_point = current;
        observation.model_a = point_a;
        observation.model_b = point_b;
        observation.map_indices = map_indices;
        if (rough_pass)
            rough_observations.push_back(observation);

        const double group_weight =
            rough_pass ? 1.0 : residualCellWeight(current);
        const double evidence_weight =
            rough_pass ? 1.0 : mapEvidenceWeight(map_in, map_indices);
        const double geometry_weight =
            0.55 + 0.45 * clampValue(
                (linearity - 0.60) / 0.40,
                0.0,
                1.0);
        const double range = current.norm();
        // Smoothly reduce distant edge information.
        const double range_weight = clampValue(
            1.0 / std::sqrt(1.0 + std::pow(range / 100.0, 2.0)),
            0.75,
            1.0);
        const double direction_weight = rough_pass
            ? 1.0
            : directionBalanceWeight(true, direction);
        // Shared reliability is multiplied by the
        // edge-specific line-quality and range evidence.
        const double information_weight = weightingActive()
            ? group_weight * evidence_weight * geometry_weight *
                range_weight * direction_weight
            : 1.0;

        const bool static_observation =
            !weightingActive() || group_weight >= 0.45;
        if (final_pass)
        {
            recordMapVotes(observation, static_observation);
            if (static_observation)
            {
                inheritPointState(
                    pc_in->points[i], map_in, map_indices);
                pc_in->points[i].support =
                    saturatedIncrement(pc_in->points[i].support, 2);
                pc_in->points[i].conflict = std::max(
                    0.0f, pc_in->points[i].conflict - 1.0f);
            }
            else
            {
                pc_in->points[i].support = 0;
                pc_in->points[i].conflict = 8;
            }
        }

        if (information_weight < 0.10)
            continue;
        ceres::CostFunction* cost_function =
            new EdgeAnalyticCostFunction(
                current,
                point_a,
                point_b,
                std::sqrt(information_weight));
        problem.AddResidualBlock(
            cost_function, loss_function, parameters);
    }
}

void OdomEstimationClass::addWeightedSurfCostFactor(
    const pcl::PointCloud<PointType>::Ptr& pc_in,
    const pcl::PointCloud<PointType>::Ptr& map_in,
    ceres::Problem& problem,
    ceres::LossFunction* loss_function,
    bool rough_pass,
    bool final_pass)
{
    std::vector<int> indices(5);
    std::vector<float> squared_distances(5);
    for (int i = 0; i < static_cast<int>(pc_in->size()); ++i)
    {
        PointType point_temp;
        pointAssociateToMap(&pc_in->points[i], &point_temp);
        if (kdtreeSurfMap->nearestKSearch(
                point_temp, 5, indices, squared_distances) < 5 ||
            squared_distances[4] >= 1.0f)
        {
            continue;
        }

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        std::array<int, 5> map_indices;
        for (int j = 0; j < 5; ++j)
        {
            map_indices[j] = indices[j];
            center += Eigen::Vector3d(
                map_in->points[indices[j]].x,
                map_in->points[indices[j]].y,
                map_in->points[indices[j]].z);
        }
        center /= 5.0;

        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (int index : map_indices)
        {
            const Eigen::Vector3d point(
                map_in->points[index].x,
                map_in->points[index].y,
                map_in->points[index].z);
            const Eigen::Vector3d centered = point - center;
            covariance += centered * centered.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(
            covariance);
        if (eigen_solver.info() != Eigen::Success)
            continue;
        const Eigen::Vector3d eigenvalues = eigen_solver.eigenvalues();
        const double largest_eigenvalue =
            std::max(eigenvalues[2], 1e-9);
        // Plane quality from ordered covariance eigenvalues.
        const double planarity =
            (eigenvalues[1] - eigenvalues[0]) / largest_eigenvalue;
        if (planarity < 0.08)
            continue;
        const Eigen::Vector3d normal =
            eigen_solver.eigenvectors().col(0);
        const double offset = -normal.dot(center);

        bool valid_plane = true;
        for (int index : map_indices)
        {
            const PointType& point = map_in->points[index];
            if (std::abs(
                    normal.x() * point.x +
                    normal.y() * point.y +
                    normal.z() * point.z + offset) > 0.2)
            {
                valid_plane = false;
                break;
            }
        }
        if (!valid_plane)
            continue;

        const Eigen::Vector3d current(
            pc_in->points[i].x,
            pc_in->points[i].y,
            pc_in->points[i].z);
        ResidualObservation observation;
        observation.edge = false;
        observation.polar_cell = polarCell(current);
        observation.curr_point = current;
        observation.model_a = normal;
        observation.model_b = Eigen::Vector3d(offset, 0.0, 0.0);
        observation.map_indices = map_indices;
        if (rough_pass)
            rough_observations.push_back(observation);

        const double group_weight =
            rough_pass ? 1.0 : residualCellWeight(current);
        const double evidence_weight =
            rough_pass ? 1.0 : mapEvidenceWeight(map_in, map_indices);
        const double geometry_weight =
            0.50 + 0.50 * clampValue(
                (planarity - 0.20) / 0.65,
                0.0,
                1.0);
        const double range = current.norm();
        const Eigen::Vector3d ray_world =
            range > 1e-6
                ? q_w_curr * (current / range)
                : Eigen::Vector3d::UnitX();
        const double incidence =
            clampValue(std::abs(normal.dot(ray_world)), 0.0, 1.0);
        const double radial_sigma = 0.02;
        const double tangential_sigma = 0.0015 * range;
        // Project radial and tangential LiDAR uncertainty
        // onto the plane normal using range and incidence angle.
        const double projected_sigma = std::sqrt(
            radial_sigma * radial_sigma * incidence * incidence +
            tangential_sigma * tangential_sigma *
                (1.0 - incidence * incidence));
        const double noise_weight = clampValue(
            0.05 / std::max(projected_sigma, 0.02),
            0.55,
            1.0);
        const double direction_weight = rough_pass
            ? 1.0
            : directionBalanceWeight(false, normal);
        // Shared reliability is multiplied by the
        // surface-specific plane-quality and projected-noise evidence.
        const double information_weight = weightingActive()
            ? group_weight * evidence_weight * geometry_weight *
                noise_weight * direction_weight
            : 1.0;

        const bool static_observation =
            !weightingActive() || group_weight >= 0.45;
        if (final_pass)
        {
            recordMapVotes(observation, static_observation);
            if (static_observation)
            {
                inheritPointState(
                    pc_in->points[i], map_in, map_indices);
                pc_in->points[i].support =
                    saturatedIncrement(pc_in->points[i].support, 2);
                pc_in->points[i].conflict = std::max(
                    0.0f, pc_in->points[i].conflict - 1.0f);
            }
            else
            {
                pc_in->points[i].support = 0;
                pc_in->points[i].conflict = 8;
            }
        }

        if (information_weight < 0.10)
            continue;
        ceres::CostFunction* cost_function =
            new SurfNormAnalyticCostFunction(
                current,
                normal,
                offset,
                std::sqrt(information_weight));
        problem.AddResidualBlock(
            cost_function, loss_function, parameters);
    }
}

bool OdomEstimationClass::findCertifiedNearestNeighbors(
    const PointType& query,
    const pcl::PointCloud<PointType>::Ptr& map,
    const AssociationVoxelIndex& index,
    const pcl::KdTreeFLANN<PointType>::Ptr& tree,
    bool use_global_tree,
    LocalGeometry& cache,
    std::array<int, 5>& map_indices,
    std::array<float, 5>& squared_distances,
    std::vector<int>& tree_search_indices,
    std::vector<float>& tree_search_distances)
{
    // Exact neighbor search has three outcomes:
    // (a) certify that fewer than five accepted neighbors remain;
    // (b) certify the exact current top-five set by the triangle inequality;
    // (c) if neither proof succeeds, run a complete exact search and refresh
    //     the certificate. No approximate correspondence is returned.
    const Eigen::Vector3f query_position(query.x, query.y, query.z);

    if (cache.candidates_initialized)
    {
        std::array<
            std::pair<float, int>,
            kAssociationCandidateCount> ranked_candidates;
        const std::size_t candidate_count =
            static_cast<std::size_t>(cache.candidate_count);
        bool valid_candidates =
            candidate_count <= kAssociationCandidateCount;
        for (std::size_t j = 0;
             valid_candidates && j < candidate_count;
             ++j)
        {
            const int index = cache.candidate_indices[j];
            if (index < 0 || index >= static_cast<int>(map->size()))
            {
                valid_candidates = false;
                break;
            }
            const PointType& candidate = map->points[index];
            const double dx = query.x - candidate.x;
            const double dy = query.y - candidate.y;
            const double dz = query.z - candidate.z;
            ranked_candidates[j] = std::make_pair(
                static_cast<float>(dx * dx + dy * dy + dz * dz),
                index);
        }

        if (valid_candidates)
        {
            const double displacement =
                (query_position - cache.query_position).norm();
            const double outside_lower_bound =
                cache.outside_distance - displacement;
            if (candidate_count < kLocalGeometryNeighborCount)
            {
                // Every non-candidate remains outside the acceptance radius,
                // so fewer than five valid neighbors is an exact rejection.
                if (outside_lower_bound >= kAssociationRadius)
                {
                    return false;
                }
            }
            else
            {
                std::partial_sort(
                    ranked_candidates.begin(),
                    ranked_candidates.begin() + kLocalGeometryNeighborCount,
                    ranked_candidates.begin() + candidate_count,
                    [](const std::pair<float, int>& left,
                       const std::pair<float, int>& right)
                    {
                        if (left.first != right.first)
                            return left.first < right.first;
                        return left.second < right.second;
                    });
                const double candidate_fifth_distance = std::sqrt(
                    std::max(0.0f, ranked_candidates[4].first));

                // Triangle inequality certifies exact top-five membership.
                if (candidate_fifth_distance + 1e-6 < outside_lower_bound)
                {
                    for (std::size_t j = 0;
                         j < kLocalGeometryNeighborCount;
                         ++j)
                    {
                        map_indices[j] = ranked_candidates[j].second;
                        squared_distances[j] = ranked_candidates[j].first;
                    }
                    return squared_distances.back() < 1.0f;
                }
            }
        }
    }

    std::array<int, kAssociationSearchCount> search_indices;
    std::array<float, kAssociationSearchCount> search_squared_distances;
    float voxel_outside_distance = kAssociationRadius;
    int found = 0;
    if (use_global_tree)
    {
        // The initial map is static during the high-iteration bootstrap frame.
        const int query_count = std::min(
            static_cast<int>(kAssociationSearchCount),
            static_cast<int>(map->size()));
        tree_search_indices.resize(query_count);
        tree_search_distances.resize(query_count);
        found = tree->nearestKSearch(
            query,
            query_count,
            tree_search_indices,
            tree_search_distances);
        for (int j = 0; j < found; ++j)
        {
            search_indices[j] = tree_search_indices[j];
            search_squared_distances[j] = tree_search_distances[j];
        }
    }
    else
    {
        // Steady-state queries use the exact, incrementally updated voxel hash.
        found = radiusNearestSearch(
            query,
            map,
            index,
            search_indices,
            search_squared_distances,
            voxel_outside_distance);
    }
    if (found < static_cast<int>(kLocalGeometryNeighborCount))
    {
        cache.candidates_initialized = true;
        cache.candidate_count = std::min(
            found, static_cast<int>(kAssociationCandidateCount));
        cache.query_position = query_position;
        cache.outside_distance = use_global_tree
            ? std::numeric_limits<float>::infinity()
            : voxel_outside_distance;
        for (int j = 0; j < cache.candidate_count; ++j)
            cache.candidate_indices[j] = search_indices[j];
        return false;
    }

    cache.candidates_initialized = true;
    cache.candidate_count = std::min(
        found, static_cast<int>(kAssociationCandidateCount));
    cache.query_position = query_position;
    if (found > static_cast<int>(kAssociationCandidateCount))
    {
        cache.outside_distance = std::sqrt(std::max(
            0.0f, search_squared_distances[kAssociationCandidateCount]));
    }
    else
    {
        cache.outside_distance = use_global_tree
            ? std::numeric_limits<float>::infinity()
            : voxel_outside_distance;
    }
    for (int j = 0; j < cache.candidate_count; ++j)
        cache.candidate_indices[j] = search_indices[j];
    for (std::size_t j = 0; j < kLocalGeometryNeighborCount; ++j)
    {
        map_indices[j] = search_indices[j];
        squared_distances[j] = search_squared_distances[j];
    }
    return squared_distances.back() < 1.0f;
}

void OdomEstimationClass::addReusableWeightedEdgeCostFactor(
    const pcl::PointCloud<PointType>::Ptr& pc_in,
    const pcl::PointCloud<PointType>::Ptr& map_in,
    ceres::Problem& problem,
    ceres::LossFunction* loss_function,
    bool rough_pass,
    bool final_pass)
{
    // Full certifies neighbor membership and reuses geometry; FullGlobal resets the
    // cache for every factor and performs an exact global KD-tree query. Both
    // paths use the same direct eigensolver, residual, weight, and Ceres solver.
    // Workload-aware exact dispatch: the first frame has many outer passes,
    // while steady frames have only the configured small iteration count.
    const bool use_global_tree =
        !geometryReuseEnabled() || frame_id == 1;
    std::vector<int> tree_search_indices(kAssociationSearchCount);
    std::vector<float> tree_search_distances(kAssociationSearchCount);
    for (int i = 0; i < static_cast<int>(pc_in->size()); ++i)
    {
        PointType point_temp;
        pointAssociateToMap(&pc_in->points[i], &point_temp);
        std::array<int, 5> map_indices;
        std::array<float, 5> squared_distances;
        LocalGeometry& cached_geometry = edge_geometry_cache[i];
        if (!geometryReuseEnabled())
            cached_geometry = LocalGeometry();
        if (!findCertifiedNearestNeighbors(
                point_temp,
                map_in,
                edge_association_index,
                kdtreeEdgeMap,
                use_global_tree,
                cached_geometry,
                map_indices,
                squared_distances,
                tree_search_indices,
                tree_search_distances))
        {
            continue;
        }

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        Eigen::Vector3d direction = Eigen::Vector3d::Zero();
        double linearity = 0.0;

        std::array<int, 5> geometry_key = map_indices;
        std::sort(geometry_key.begin(), geometry_key.end());
        const bool same_geometry = cached_geometry.geometry_initialized &&
            cached_geometry.map_indices == geometry_key;
        if (same_geometry)
        {
            if (!cached_geometry.valid)
                continue;
            center = cached_geometry.center;
            direction = cached_geometry.direction;
            linearity = cached_geometry.quality;
        }
        else
        {
            cached_geometry.geometry_initialized = true;
            cached_geometry.valid = false;
            cached_geometry.map_indices = geometry_key;

            for (int index : map_indices)
            {
                center += Eigen::Vector3d(
                    map_in->points[index].x,
                    map_in->points[index].y,
                    map_in->points[index].z);
            }
            center /= 5.0;

            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (int index : map_indices)
            {
                const Eigen::Vector3d point(
                    map_in->points[index].x,
                    map_in->points[index].y,
                    map_in->points[index].z);
                const Eigen::Vector3d centered = point - center;
                covariance += centered * centered.transpose();
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
            solver.computeDirect(covariance, Eigen::ComputeEigenvectors);
            const Eigen::Vector3d eigenvalues = solver.eigenvalues();
            const double largest_eigenvalue =
                std::max(eigenvalues[2], 1e-9);
            // Evaluate geometric quality only on an exact-key cache miss.
            linearity =
                (eigenvalues[2] - eigenvalues[1]) / largest_eigenvalue;
            if (linearity <= 2.0 / 3.0)
                continue;

            direction = solver.eigenvectors().col(2);
            cached_geometry.valid = true;
            cached_geometry.center = center;
            cached_geometry.direction = direction;
            cached_geometry.quality = linearity;
        }

        const Eigen::Vector3d point_a = center + 0.1 * direction;
        const Eigen::Vector3d point_b = center - 0.1 * direction;
        const Eigen::Vector3d current(
            pc_in->points[i].x,
            pc_in->points[i].y,
            pc_in->points[i].z);

        ResidualObservation observation;
        observation.edge = true;
        observation.polar_cell = polarCell(current);
        observation.curr_point = current;
        observation.model_a = point_a;
        observation.model_b = point_b;
        observation.map_indices = map_indices;
        if (rough_pass)
            rough_observations.push_back(observation);

        const double group_weight =
            rough_pass ? 1.0 : residualCellWeight(current);
        const double evidence_weight =
            rough_pass ? 1.0 : mapEvidenceWeight(map_in, map_indices);
        const double geometry_weight =
            0.55 + 0.45 * clampValue(
                (linearity - 0.60) / 0.40,
                0.0,
                1.0);
        const double range = current.norm();
        const double range_weight = clampValue(
            1.0 / std::sqrt(1.0 + std::pow(range / 100.0, 2.0)),
            0.75,
            1.0);
        const double direction_weight = rough_pass
            ? 1.0
            : directionBalanceWeight(true, direction);

        // Recompute every pose-dependent weight and
        // residual; only identical map-neighborhood geometry is reused, so the
        // weighted objective remains equal to the non-reuse path.
        const double information_weight = weightingActive()
            ? group_weight * evidence_weight * geometry_weight *
                range_weight * direction_weight
            : 1.0;

        const bool static_observation =
            !weightingActive() || group_weight >= 0.45;

        if (final_pass)
        {
            recordMapVotes(observation, static_observation);
            if (static_observation)
            {
                inheritPointState(
                    pc_in->points[i], map_in, map_indices);
                pc_in->points[i].support =
                    saturatedIncrement(pc_in->points[i].support, 2);
                pc_in->points[i].conflict = std::max(
                    0.0f, pc_in->points[i].conflict - 1.0f);
            }
            else
            {
                pc_in->points[i].support = 0;
                pc_in->points[i].conflict = 8;
            }
        }

        if (information_weight < 0.10)
            continue;
        ceres::CostFunction* cost_function =
            new EdgeAnalyticCostFunction(
                current,
                point_a,
                point_b,
                std::sqrt(information_weight));
        problem.AddResidualBlock(
            cost_function, loss_function, parameters);
    }
}

void OdomEstimationClass::addReusableWeightedSurfCostFactor(
    const pcl::PointCloud<PointType>::Ptr& pc_in,
    const pcl::PointCloud<PointType>::Ptr& map_in,
    ceres::Problem& problem,
    ceres::LossFunction* loss_function,
    bool rough_pass,
    bool final_pass)
{
    // Shared Full/FullGlobal surface path. Full uses the certified association
    // and canonical-key reuse; FullGlobal always performs an
    // exact global query and refits the same direct-eigensolver plane model.
    const bool use_global_tree =
        !geometryReuseEnabled() || frame_id == 1;
    std::vector<int> tree_search_indices(kAssociationSearchCount);
    std::vector<float> tree_search_distances(kAssociationSearchCount);
    for (int i = 0; i < static_cast<int>(pc_in->size()); ++i)
    {
        PointType point_temp;
        pointAssociateToMap(&pc_in->points[i], &point_temp);
        std::array<int, 5> map_indices;
        std::array<float, 5> squared_distances;
        LocalGeometry& cached_geometry = surf_geometry_cache[i];
        if (!geometryReuseEnabled())
            cached_geometry = LocalGeometry();
        if (!findCertifiedNearestNeighbors(
                point_temp,
                map_in,
                surf_association_index,
                kdtreeSurfMap,
                use_global_tree,
                cached_geometry,
                map_indices,
                squared_distances,
                tree_search_indices,
                tree_search_distances))
        {
            continue;
        }

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        double planarity = 0.0;
        double offset = 0.0;

        std::array<int, 5> geometry_key = map_indices;
        std::sort(geometry_key.begin(), geometry_key.end());
        const bool same_geometry = cached_geometry.geometry_initialized &&
            cached_geometry.map_indices == geometry_key;
        if (same_geometry)
        {
            if (!cached_geometry.valid)
                continue;
            center = cached_geometry.center;
            normal = cached_geometry.direction;
            planarity = cached_geometry.quality;
            offset = cached_geometry.offset;
        }
        else
        {
            cached_geometry.geometry_initialized = true;
            cached_geometry.valid = false;
            cached_geometry.map_indices = geometry_key;

            for (int index : map_indices)
            {
                center += Eigen::Vector3d(
                    map_in->points[index].x,
                    map_in->points[index].y,
                    map_in->points[index].z);
            }
            center /= 5.0;

            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (int index : map_indices)
            {
                const Eigen::Vector3d point(
                    map_in->points[index].x,
                    map_in->points[index].y,
                    map_in->points[index].z);
                const Eigen::Vector3d centered = point - center;
                covariance += centered * centered.transpose();
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver;
            eigen_solver.computeDirect(
                covariance, Eigen::ComputeEigenvectors);
            if (eigen_solver.info() != Eigen::Success)
                continue;
            const Eigen::Vector3d eigenvalues =
                eigen_solver.eigenvalues();
            const double largest_eigenvalue =
                std::max(eigenvalues[2], 1e-9);
            // Evaluate geometric quality only on an exact-key cache miss.
            planarity =
                (eigenvalues[1] - eigenvalues[0]) /
                largest_eigenvalue;
            if (planarity < 0.08)
                continue;
            normal = eigen_solver.eigenvectors().col(0);
            offset = -normal.dot(center);

            bool valid_plane = true;
            for (int index : map_indices)
            {
                const PointType& point = map_in->points[index];
                if (std::abs(
                        normal.x() * point.x +
                        normal.y() * point.y +
                        normal.z() * point.z + offset) > 0.2)
                {
                    valid_plane = false;
                    break;
                }
            }
            if (!valid_plane)
                continue;

            cached_geometry.valid = true;
            cached_geometry.center = center;
            cached_geometry.direction = normal;
            cached_geometry.quality = planarity;
            cached_geometry.offset = offset;
        }

        const Eigen::Vector3d current(
            pc_in->points[i].x,
            pc_in->points[i].y,
            pc_in->points[i].z);
        ResidualObservation observation;
        observation.edge = false;
        observation.polar_cell = polarCell(current);
        observation.curr_point = current;
        observation.model_a = normal;
        observation.model_b = Eigen::Vector3d(offset, 0.0, 0.0);
        observation.map_indices = map_indices;
        if (rough_pass)
            rough_observations.push_back(observation);

        const double group_weight =
            rough_pass ? 1.0 : residualCellWeight(current);
        const double evidence_weight =
            rough_pass ? 1.0 : mapEvidenceWeight(map_in, map_indices);
        const double geometry_weight =
            0.50 + 0.50 * clampValue(
                (planarity - 0.20) / 0.65,
                0.0,
                1.0);
        const double range = current.norm();
        const Eigen::Vector3d ray_world =
            range > 1e-6
                ? q_w_curr * (current / range)
                : Eigen::Vector3d::UnitX();
        const double incidence =
            clampValue(std::abs(normal.dot(ray_world)), 0.0, 1.0);
        const double radial_sigma = 0.02;
        const double tangential_sigma = 0.0015 * range;
        // Range/incidence projected plane uncertainty.
        const double projected_sigma = std::sqrt(
            radial_sigma * radial_sigma * incidence * incidence +
            tangential_sigma * tangential_sigma *
                (1.0 - incidence * incidence));
        const double noise_weight = clampValue(
            0.05 / std::max(projected_sigma, 0.02),
            0.55,
            1.0);
        const double direction_weight = rough_pass
            ? 1.0
            : directionBalanceWeight(false, normal);

        // All surface weights and residuals are
        // recomputed after safe geometry reuse, preserving the objective.
        const double information_weight = weightingActive()
            ? group_weight * evidence_weight * geometry_weight *
                noise_weight * direction_weight
            : 1.0;

        const bool static_observation =
            !weightingActive() || group_weight >= 0.45;

        if (final_pass)
        {
            recordMapVotes(observation, static_observation);
            if (static_observation)
            {
                inheritPointState(
                    pc_in->points[i], map_in, map_indices);
                pc_in->points[i].support =
                    saturatedIncrement(pc_in->points[i].support, 2);
                pc_in->points[i].conflict = std::max(
                    0.0f, pc_in->points[i].conflict - 1.0f);
            }
            else
            {
                pc_in->points[i].support = 0;
                pc_in->points[i].conflict = 8;
            }
        }

        if (information_weight < 0.10)
            continue;
        ceres::CostFunction* cost_function =
            new SurfNormAnalyticCostFunction(
                current,
                normal,
                offset,
                std::sqrt(information_weight));
        problem.AddResidualBlock(
            cost_function, loss_function, parameters);
    }
}

void OdomEstimationClass::buildVisibilityGrid(
    const pcl::PointCloud<PointType>::Ptr& current_edge,
    const pcl::PointCloud<PointType>::Ptr& current_surf)
{
    // Current edge and surface returns form a conservative range image. It is
    // consulted only during event-driven maintenance to confirm that a
    // previously conflicting map point lies in observed free space.
    const int grid_size =
        kVisibilityAzimuthCells * kVisibilityElevationCells;
    observed_ranges.assign(
        grid_size, std::numeric_limits<float>::infinity());

    const auto insert_current = [this](
                                    const pcl::PointCloud<PointType>::Ptr& cloud)
    {
        for (const PointType& point : cloud->points)
        {
            const Eigen::Vector3d vector(point.x, point.y, point.z);
            const int cell = visibilityCell(vector);
            if (cell >= 0)
            {
                observed_ranges[cell] = std::min(
                    observed_ranges[cell],
                    static_cast<float>(vector.norm()));
            }
        }
    };
    insert_current(current_edge);
    insert_current(current_surf);
}

void OdomEstimationClass::decayPointEvidence(PointType& point) const
{
    if (point.support + point.conflict < 200.0f)
        return;
    point.support = std::max(1.0f, point.support * 0.5f);
    point.conflict *= 0.5f;
}

void OdomEstimationClass::updateActiveEvidence(
    pcl::PointCloud<PointType>::Ptr& map,
    const SparseEvidenceVotes& votes)
{
    // Evidence update: support strengthens persistence, conflict weakens
    // it, and saturation-triggered decay preserves the same support/conflict
    // ratio without letting counters overflow.
    for (int index : votes.touched_indices)
    {
        if (index < 0 || index >= static_cast<int>(map->size()))
        {
            continue;
        }

        PointType& point = map->points[index];
        const EvidenceVotes& vote = votes.values[index];
        if (vote.support > 0)
        {
            point.support = saturatedIncrement(
                point.support, std::min(3, static_cast<int>(vote.support)));
            point.conflict = std::max(0.0f, point.conflict - 1.0f);
        }
        if (vote.conflict > 0)
        {
            point.conflict = saturatedIncrement(
                point.conflict,
                std::min(3, static_cast<int>(vote.conflict)));
        }
        decayPointEvidence(point);
    }
}

void OdomEstimationClass::updateFreeSpaceEvidence(
    pcl::PointCloud<PointType>::Ptr& map)
{
    const Eigen::Quaterniond inverse_rotation = q_w_curr.conjugate();
    for (PointType& point : map->points)
    {
        // Ray evidence confirms a residual-seeded suspicion; it cannot create
        // a dynamic label by itself in sparse or partially observed regions.
        if (point.conflict < 2.0f ||
            point.conflict * 2.0f < point.support)
        {
            continue;
        }

        const Eigen::Vector3d point_w(point.x, point.y, point.z);
        const Eigen::Vector3d point_sensor =
            inverse_rotation * (point_w - t_w_curr);
        const double map_range = point_sensor.norm();
        if (map_range < 3.0 || map_range > 95.0)
            continue;

        const int cell = visibilityCell(point_sensor);
        if (cell < 0 || !std::isfinite(observed_ranges[cell]))
            continue;

        const double observed_range =
            static_cast<double>(observed_ranges[cell]);
        if (observed_range - map_range > kFreeSpaceMarginMeters)
        {
            point.conflict = saturatedIncrement(point.conflict);
            decayPointEvidence(point);
        }
    }
}

bool OdomEstimationClass::shouldMaintainMap() const
{
    const std::size_t map_points =
        laserCloudCornerMap->size() + laserCloudSurfMap->size();
    const double travel =
        (odom.translation() - last_maintenance_position).norm();
    return frames_since_maintenance >= kMaintenanceFrameLimit ||
        travel >= kMaintenanceTravelMeters ||
        map_points >= kMapHardLimit;
}

void OdomEstimationClass::maintainMap()
{
    // Crop and voxelize only when triggered by time, travel, or map size.
    const double x = odom.translation().x();
    const double y = odom.translation().y();
    const double z = odom.translation().z();
    cropBoxFilter.setMin(
        Eigen::Vector4f(x - 100.0, y - 100.0, z - 100.0, 1.0));
    cropBoxFilter.setMax(
        Eigen::Vector4f(x + 100.0, y + 100.0, z + 100.0, 1.0));
    cropBoxFilter.setNegative(false);

    croppedCornerScratch->clear();
    croppedSurfScratch->clear();
    compactedCornerScratch->clear();
    compactedSurfScratch->clear();

    cropBoxFilter.setInputCloud(laserCloudCornerMap);
    cropBoxFilter.filter(*croppedCornerScratch);
    cropBoxFilter.setInputCloud(laserCloudSurfMap);
    cropBoxFilter.filter(*croppedSurfScratch);

    downSizeFilterEdge.setInputCloud(croppedCornerScratch);
    downSizeFilterEdge.filter(*compactedCornerScratch);
    downSizeFilterSurf.setInputCloud(croppedSurfScratch);
    downSizeFilterSurf.filter(*compactedSurfScratch);

    laserCloudCornerMap.swap(compactedCornerScratch);
    laserCloudSurfMap.swap(compactedSurfScratch);
    if (geometryReuseEnabled())
    {
        rebuildAssociationIndex(
            laserCloudCornerMap, edge_association_index);
        rebuildAssociationIndex(
            laserCloudSurfMap, surf_association_index);
    }
    last_maintenance_position = odom.translation();
    frames_since_maintenance = 0;
}

void OdomEstimationClass::addPointsToMap(
    const pcl::PointCloud<PointType>::Ptr& downsampled_edge,
    const pcl::PointCloud<PointType>::Ptr& downsampled_surf)
{
    // Map commit policy: reject only features already classified as strongly
    // conflicting, transform accepted points with the final pose, incrementally
    // append them to the exact index, and compact the local map only on an event.
    ++frames_since_maintenance;
    const bool maintenance_due = shouldMaintainMap();
    if (maintenance_due && weightingEnabled())
    {
        buildVisibilityGrid(downsampled_edge, downsampled_surf);
        updateFreeSpaceEvidence(laserCloudCornerMap);
        updateFreeSpaceEvidence(laserCloudSurfMap);
    }

    for (const PointType& point : downsampled_edge->points)
    {
        if (weightingEnabled() && point.conflict >= 6 && point.support == 0)
            continue;
        PointType point_w;
        pointAssociateToMap(&point, &point_w);
        laserCloudCornerMap->push_back(point_w);
        if (geometryReuseEnabled())
        {
            insertAssociationPoint(
                point_w,
                static_cast<int>(laserCloudCornerMap->size()) - 1,
                edge_association_index);
        }
    }
    for (const PointType& point : downsampled_surf->points)
    {
        if (weightingEnabled() && point.conflict >= 6 && point.support == 0)
            continue;
        PointType point_w;
        pointAssociateToMap(&point, &point_w);
        laserCloudSurfMap->push_back(point_w);
        if (geometryReuseEnabled())
        {
            insertAssociationPoint(
                point_w,
                static_cast<int>(laserCloudSurfMap->size()) - 1,
                surf_association_index);
        }
    }

    const std::size_t map_points =
        laserCloudCornerMap->size() + laserCloudSurfMap->size();
    if (maintenance_due || map_points >= kMapHardLimit)
        maintainMap();
}

void OdomEstimationClass::getMap(
    pcl::PointCloud<PointType>::Ptr& laserCloudMap)
{
    *laserCloudMap += *laserCloudSurfMap;
    *laserCloudMap += *laserCloudCornerMap;
}
