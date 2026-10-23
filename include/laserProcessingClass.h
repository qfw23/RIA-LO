// RIA-LO LiDAR feature front end.

#ifndef RIA_LO_LASER_PROCESSING_CLASS_H_
#define RIA_LO_LASER_PROCESSING_CLASS_H_

#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "lidar.h"

struct Double2d
{
    int id;
    double value;

    Double2d(int id_in, double value_in);
};

class LaserProcessingClass
{
public:
    LaserProcessingClass() = default;

    void init(lidar::Lidar lidar_param_in);
    void configureFeatureProfile(
        double edge_curvature_threshold,
        int edge_features_per_sector,
        double edge_suppression_distance_sq);
    void configureScanPhaseEncoding(bool enabled);
    void featureExtraction(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_edge,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_surf);

private:
    // Sort one ring sector by curvature, retain spatially separated sharp
    // points as edge features, and send the remaining points to the surface
    // stream. This front-end ordering is shared by all pipeline modes.
    void featureExtractionFromSector(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
        std::vector<Double2d>& cloud_curvature,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_edge,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_surf);

    lidar::Lidar lidar_param;
    double edge_curvature_threshold = 0.1;
    int edge_features_per_sector = 20;
    double edge_suppression_distance_sq = 0.05;
    bool scan_phase_encoding = false;
};

#endif  // RIA_LO_LASER_PROCESSING_CLASS_H_
