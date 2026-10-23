#ifndef RIA_LO_ODOM_ESTIMATION_CLASS_H_
#define RIA_LO_ODOM_ESTIMATION_CLASS_H_

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "lidar.h"
#include "lidarOptimization.h"

struct EIGEN_ALIGN16 RstPoint
{
    PCL_ADD_POINT4D;
    float intensity = 0.0f;
    // Reversible map evidence: static support and motion conflict.
    float support = 0.0f;
    float conflict = 0.0f;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    RstPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, support, support)
    (float, conflict, conflict))

using PointType = RstPoint;

// Core RIA-LO estimator. One update predicts the pose, optionally compensates
// scan motion, extracts rough residual statistics, builds reliability-aware
// edge/plane factors, solves the weighted scan-to-map objective, and commits
// map evidence only after the final pass.
class OdomEstimationClass
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum class ExecutionMode
    {
        // Unweighted point-to-line/point-to-plane geometric baseline.
        Base,
        // Complete adaptive correspondence weighting without geometry reuse.
        Weighted,
        // Full weighted objective with global KD-tree association.
        FullGlobal,
        // Complete weighting with certified correspondence and geometry reuse.
        Full
    };

    OdomEstimationClass();

    void init(
        lidar::Lidar lidar_param,
        double map_resolution,
        const std::string& pipeline_mode = "full");
    void configureAdaptiveWeighting(
        bool enabled,
        int enter_constraints,
        int exit_constraints,
        int enter_direction_bins,
        int exit_direction_bins);
    void configureAccuracyProfile(
        double surface_resolution_scale,
        int optimization_iterations,
        int ceres_max_iterations,
        double robust_delta_scale_multiplier,
        double robust_delta_min,
        double robust_delta_max,
        double direction_balance_exponent,
        double direction_weight_min,
        double direction_weight_max);
    void configureScanMotionCompensation(
        bool enabled,
        double reference_phase,
        double translation_scale,
        double rotation_scale,
        int warmup_frames);
    void configureAdaptiveMotionScaling(
        bool enabled,
        double high_turn_translation_scale,
        double high_turn_rotation_scale,
        double turn_start,
        double turn_end,
        double smoothing,
        double duty_start,
        double duty_end,
        double duty_smoothing);
    void initMapWithPoints(
        const pcl::PointCloud<PointType>::Ptr& edge_in,
        const pcl::PointCloud<PointType>::Ptr& surf_in);
    void updatePointsToMap(
        const pcl::PointCloud<PointType>::Ptr& edge_in,
        const pcl::PointCloud<PointType>::Ptr& surf_in);
    void getMap(pcl::PointCloud<PointType>::Ptr& laserCloudMap);

    Eigen::Isometry3d odom;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfMap;
    float map_resolution;

private:
    // Per-frame state used by motion compensation and adaptive weighting.
    int frame_id;
    int frame_candidate_constraints;
    int frame_direction_bins;
    double frame_residual_median;
    double frame_residual_scale;
    bool frame_weighting_active;

    static constexpr std::size_t kLocalGeometryNeighborCount = 5;
    static constexpr std::size_t kAssociationCandidateCount = 6;
    static constexpr std::size_t kAssociationSearchCount =
        kAssociationCandidateCount + 1;

    struct AssociationVoxelKey
    {
        int x = 0;
        int y = 0;
        int z = 0;

        bool operator==(const AssociationVoxelKey& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct AssociationVoxelKeyHash
    {
        std::size_t operator()(const AssociationVoxelKey& key) const;
    };

    using AssociationVoxelIndex = std::unordered_map<
        AssociationVoxelKey,
        std::vector<int>,
        AssociationVoxelKeyHash>;

    struct ResidualObservation
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // Rough-pass observation used to estimate reliability.
        // model_a/model_b store line endpoints for an edge, or plane normal
        // and offset for a surface; map_indices identify map evidence.
        bool edge = false;
        int polar_cell = -1;
        Eigen::Vector3d curr_point = Eigen::Vector3d::Zero();
        Eigen::Vector3d model_a = Eigen::Vector3d::Zero();
        Eigen::Vector3d model_b = Eigen::Vector3d::Zero();
        std::array<int, 5> map_indices{{-1, -1, -1, -1, -1}};
    };

    using ResidualVector = std::vector<
        ResidualObservation,
        Eigen::aligned_allocator<ResidualObservation>>;

    struct LocalGeometry
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // Candidate distances and an outside-set lower
        // bound certify exact top-five membership or exact rejection. Geometry
        // is reusable only when the sorted five-neighbor identity key matches.
        bool candidates_initialized = false;
        bool geometry_initialized = false;
        bool valid = false;
        int candidate_count = 0;
        std::array<int, kAssociationCandidateCount> candidate_indices{};
        Eigen::Vector3f query_position = Eigen::Vector3f::Zero();
        float outside_distance = 0.0f;
        std::array<int, 5> map_indices{{-1, -1, -1, -1, -1}};
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        Eigen::Vector3d direction = Eigen::Vector3d::Zero();
        double quality = 0.0;
        double offset = 0.0;
    };

    using LocalGeometryVector = std::vector<
        LocalGeometry,
        Eigen::aligned_allocator<LocalGeometry>>;

    struct EvidenceVotes
    {
        std::uint16_t support = 0;
        std::uint16_t conflict = 0;
    };

    struct SparseEvidenceVotes
    {
        // Epoch tags clear only touched entries, avoiding a full-map reset.
        std::vector<EvidenceVotes> values;
        std::vector<std::uint32_t> epochs;
        std::vector<int> touched_indices;
    };

    double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
    // Ceres global pose storage [qx,qy,qz,qw,tx,ty,tz]. The optimizer exposes
    // a six-dimensional left-multiplicative SE(3) increment through
    // PoseSE3Parameterization.
    Eigen::Map<Eigen::Quaterniond> q_w_curr =
        Eigen::Map<Eigen::Quaterniond>(parameters);
    Eigen::Map<Eigen::Vector3d> t_w_curr =
        Eigen::Map<Eigen::Vector3d>(parameters + 4);

    Eigen::Isometry3d last_odom;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeEdgeMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfMap;
    pcl::VoxelGrid<PointType> downSizeFilterEdge;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::CropBox<PointType> cropBoxFilter;
    pcl::PointCloud<PointType>::Ptr croppedCornerScratch;
    pcl::PointCloud<PointType>::Ptr croppedSurfScratch;
    pcl::PointCloud<PointType>::Ptr compactedCornerScratch;
    pcl::PointCloud<PointType>::Ptr compactedSurfScratch;
    pcl::PointCloud<PointType>::Ptr currentEdgeScratch;
    pcl::PointCloud<PointType>::Ptr currentSurfScratch;
    pcl::PointCloud<PointType>::Ptr deskewedEdgeScratch;
    pcl::PointCloud<PointType>::Ptr deskewedSurfScratch;
    int optimization_count;
    int steady_optimization_count = 2;
    int ceres_max_iterations = 4;
    double surface_resolution_scale = 2.0;
    int frames_since_maintenance;
    Eigen::Vector3d last_maintenance_position;

    ResidualVector rough_observations;
    LocalGeometryVector edge_geometry_cache;
    LocalGeometryVector surf_geometry_cache;
    AssociationVoxelIndex edge_association_index;
    AssociationVoxelIndex surf_association_index;
    std::unordered_map<int, double> residual_cell_weights;
    std::unordered_map<int, double> residual_cell_history;
    std::unordered_map<int, double> direction_balance_weights;
    double robust_loss_delta;
    double robust_delta_scale_multiplier = 1.5;
    double robust_delta_min = 0.055;
    double robust_delta_max = 0.15;
    double direction_balance_exponent = 0.5;
    double direction_weight_min = 0.65;
    double direction_weight_max = 1.55;
    bool scan_motion_compensation = false;
    double scan_period = 0.1;
    double motion_reference_phase = 0.5;
    double motion_translation_scale = 1.0;
    double motion_rotation_scale = 1.0;
    int motion_compensation_warmup_frames = 10;
    bool adaptive_motion_scaling = false;
    double high_turn_translation_scale = 0.10;
    double high_turn_rotation_scale = 0.20;
    double motion_turn_start = 0.004;
    double motion_turn_end = 0.008;
    double motion_turn_smoothing = 0.20;
    double motion_turn_activity = 0.0;
    double motion_turn_duty_start = 0.45;
    double motion_turn_duty_end = 0.60;
    double motion_turn_duty_smoothing = 0.02;
    double motion_turn_duty = 0.0;
    double frame_motion_turn_instant = 0.0;
    double frame_motion_translation_scale = 1.0;
    double frame_motion_rotation_scale = 1.0;
    double frame_motion_turn_blend = 0.0;
    double frame_motion_translation = 0.0;
    double frame_motion_rotation = 0.0;
    ExecutionMode execution_mode;
    bool adaptive_weighting_enabled;
    int weighting_enter_constraints;
    int weighting_exit_constraints;
    int weighting_enter_direction_bins;
    int weighting_exit_direction_bins;
    SparseEvidenceVotes edge_evidence_votes;
    SparseEvidenceVotes surf_evidence_votes;
    std::vector<float> observed_ranges;

    bool weightingEnabled() const;
    bool weightingActive() const;
    bool fullFactorPathEnabled() const;
    bool geometryReuseEnabled() const;
    void addBaselineEdgeCostFactor(
        const pcl::PointCloud<PointType>::Ptr& pc_in,
        const pcl::PointCloud<PointType>::Ptr& map_in,
        ceres::Problem& problem,
        ceres::LossFunction* loss_function,
        bool rough_pass);
    void addBaselineSurfCostFactor(
        const pcl::PointCloud<PointType>::Ptr& pc_in,
        const pcl::PointCloud<PointType>::Ptr& map_in,
        ceres::Problem& problem,
        ceres::LossFunction* loss_function,
        bool rough_pass);
    void addWeightedEdgeCostFactor(
        const pcl::PointCloud<PointType>::Ptr& pc_in,
        const pcl::PointCloud<PointType>::Ptr& map_in,
        ceres::Problem& problem,
        ceres::LossFunction* loss_function,
        bool rough_pass,
        bool final_pass);
    void addWeightedSurfCostFactor(
        const pcl::PointCloud<PointType>::Ptr& pc_in,
        const pcl::PointCloud<PointType>::Ptr& map_in,
        ceres::Problem& problem,
        ceres::LossFunction* loss_function,
        bool rough_pass,
        bool final_pass);
    void addReusableWeightedEdgeCostFactor(
        const pcl::PointCloud<PointType>::Ptr& pc_in,
        const pcl::PointCloud<PointType>::Ptr& map_in,
        ceres::Problem& problem,
        ceres::LossFunction* loss_function,
        bool rough_pass,
        bool final_pass);
    void addReusableWeightedSurfCostFactor(
        const pcl::PointCloud<PointType>::Ptr& pc_in,
        const pcl::PointCloud<PointType>::Ptr& map_in,
        ceres::Problem& problem,
        ceres::LossFunction* loss_function,
        bool rough_pass,
        bool final_pass);
    bool findCertifiedNearestNeighbors(
        const PointType& query,
        const pcl::PointCloud<PointType>::Ptr& map,
        const AssociationVoxelIndex& index,
        const pcl::KdTreeFLANN<PointType>::Ptr& tree,
        bool use_global_tree,
        LocalGeometry& cache,
        std::array<int, 5>& map_indices,
        std::array<float, 5>& squared_distances,
        std::vector<int>& tree_search_indices,
        std::vector<float>& tree_search_distances);
    AssociationVoxelKey associationVoxelKey(const PointType& point) const;
    void rebuildAssociationIndex(
        const pcl::PointCloud<PointType>::Ptr& map,
        AssociationVoxelIndex& index);
    void insertAssociationPoint(
        const PointType& point,
        int point_index,
        AssociationVoxelIndex& index);
    int radiusNearestSearch(
        const PointType& query,
        const pcl::PointCloud<PointType>::Ptr& map,
        const AssociationVoxelIndex& index,
        std::array<int, kAssociationSearchCount>& indices,
        std::array<float, kAssociationSearchCount>& squared_distances,
        float& outside_distance) const;
    void addPointsToMap(
        const pcl::PointCloud<PointType>::Ptr& downsampledEdgeCloud,
        const pcl::PointCloud<PointType>::Ptr& downsampledSurfCloud);
    void pointAssociateToMap(
        PointType const* const pi,
        PointType* const po) const;
    void downSamplingToMap(
        const pcl::PointCloud<PointType>::Ptr& edge_pc_in,
        pcl::PointCloud<PointType>::Ptr& edge_pc_out,
        const pcl::PointCloud<PointType>::Ptr& surf_pc_in,
        pcl::PointCloud<PointType>::Ptr& surf_pc_out);
    void compensateScanMotion(
        const pcl::PointCloud<PointType>::Ptr& edge_input,
        const pcl::PointCloud<PointType>::Ptr& surf_input,
        const Eigen::Isometry3d& relative_motion,
        pcl::PointCloud<PointType>::Ptr& edge_output,
        pcl::PointCloud<PointType>::Ptr& surf_output);

    void buildResidualConsistencyField();
    double observationResidual(
        const ResidualObservation& observation,
        Eigen::Vector3d* innovation) const;
    int polarCell(const Eigen::Vector3d& point) const;
    int geometricDirectionCell(
        bool edge,
        const Eigen::Vector3d& direction) const;
    double residualCellWeight(const Eigen::Vector3d& point) const;
    double directionBalanceWeight(
        bool edge,
        const Eigen::Vector3d& direction) const;
    double mapEvidenceWeight(
        const pcl::PointCloud<PointType>::Ptr& map,
        const std::array<int, 5>& indices) const;
    void inheritPointState(
        PointType& point,
        const pcl::PointCloud<PointType>::Ptr& map,
        const std::array<int, 5>& indices) const;
    void recordMapVotes(
        const ResidualObservation& observation,
        bool static_observation);
    void prepareEvidenceVotes(
        SparseEvidenceVotes& votes,
        std::size_t map_size);
    void updateActiveEvidence(
        pcl::PointCloud<PointType>::Ptr& map,
        const SparseEvidenceVotes& votes);
    void updateFreeSpaceEvidence(pcl::PointCloud<PointType>::Ptr& map);
    void buildVisibilityGrid(
        const pcl::PointCloud<PointType>::Ptr& current_edge,
        const pcl::PointCloud<PointType>::Ptr& current_surf);
    bool shouldMaintainMap() const;
    void maintainMap();
    void decayPointEvidence(PointType& point) const;
    void initializePointState(pcl::PointCloud<PointType>::Ptr& cloud) const;
};

#endif  // RIA_LO_ODOM_ESTIMATION_CLASS_H_
