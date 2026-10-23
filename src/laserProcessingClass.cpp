// RIA-LO LiDAR feature front end.

#include "laserProcessingClass.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <pcl/filters/filter_indices.h>

void LaserProcessingClass::init(lidar::Lidar lidar_param_in)
{
    lidar_param = lidar_param_in;
}

void LaserProcessingClass::configureFeatureProfile(
    double edge_curvature_threshold_in,
    int edge_features_per_sector_in,
    double edge_suppression_distance_sq_in)
{
    edge_curvature_threshold =
        std::max(0.0, edge_curvature_threshold_in);
    edge_features_per_sector =
        std::max(1, edge_features_per_sector_in);
    edge_suppression_distance_sq =
        std::max(0.0, edge_suppression_distance_sq_in);
}

void LaserProcessingClass::configureScanPhaseEncoding(bool enabled)
{
    scan_phase_encoding = enabled;
}

void LaserProcessingClass::featureExtraction(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_edge,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_surf)
{
    // Front-end data path:
    // raw cloud -> range/ring filtering -> phase encoding ->
    // per-ring curvature -> six-sector edge selection -> surface remainder.
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pc_in, indices);

    const int scan_count = lidar_param.num_lines;
    if (pc_in->points.size() < 2)
        return;

    // For standard azimuth phi=atan2(y,x), the expression
    // start_orientation-orientation equals phi_i-phi_0. Its wrapped angular
    // progress defines normalized scan phase, which is packed with the ring
    // in the intensity field.
    const double start_orientation =
        -std::atan2(pc_in->points.front().y, pc_in->points.front().x);
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> laser_cloud_scans;
    for (int i = 0; i < scan_count; ++i)
    {
        laser_cloud_scans.push_back(
            pcl::PointCloud<pcl::PointXYZI>::Ptr(
                new pcl::PointCloud<pcl::PointXYZI>()));
    }

    // Split the input by physical LiDAR ring. The elevation-to-ring mapping
    // follows the calibrated vertical layouts of 16-, 32-, and 64-line units;
    // points outside a supported field of view never enter the optimizer.
    for (int i = 0; i < static_cast<int>(pc_in->points.size()); ++i)
    {
        int scan_id = 0;
        const double distance = std::sqrt(
            pc_in->points[i].x * pc_in->points[i].x +
            pc_in->points[i].y * pc_in->points[i].y);
        if (distance < lidar_param.min_distance ||
            distance > lidar_param.max_distance)
        {
            continue;
        }

        const double angle =
            std::atan(pc_in->points[i].z / distance) * 180 / M_PI;
        if (scan_count == 16)
        {
            scan_id = static_cast<int>((angle + 15) / 2 + 0.5);
            if (scan_id > scan_count - 1 || scan_id < 0)
                continue;
        }
        else if (scan_count == 32)
        {
            scan_id =
                static_cast<int>((angle + 92.0 / 3.0) * 3.0 / 4.0);
            if (scan_id > scan_count - 1 || scan_id < 0)
                continue;
        }
        else if (scan_count == 64)
        {
            if (angle >= -8.83)
                scan_id = static_cast<int>((2 - angle) * 3.0 + 0.5);
            else
            {
                scan_id = scan_count / 2 +
                    static_cast<int>((-8.83 - angle) * 2.0 + 0.5);
            }

            if (angle > 2 || angle < -24.33 ||
                scan_id > 63 || scan_id < 0)
            {
                continue;
            }
        }
        else
        {
            std::printf("wrong scan number\n");
        }
        pcl::PointXYZI point = pc_in->points[i];
        if (scan_phase_encoding)
        {
            const double orientation = -std::atan2(point.y, point.x);
            double phase_angle = std::fmod(
                start_orientation - orientation, 2.0 * M_PI);
            if (phase_angle < 0.0)
                phase_angle += 2.0 * M_PI;
            const double relative_time = phase_angle / (2.0 * M_PI);
            point.intensity = static_cast<float>(
                scan_id + lidar_param.scan_period * relative_time);
        }
        laser_cloud_scans[scan_id]->push_back(point);
    }

    for (int scan = 0; scan < scan_count; ++scan)
    {
        if (laser_cloud_scans[scan]->points.size() < 131)
            continue;

        // The geometric baseline uses a symmetric ten-neighbor difference.
        // Six azimuthal sectors prevent one textured direction from consuming
        // the entire edge budget and keep all pipeline modes on the same features.
        std::vector<Double2d> cloud_curvature;
        const int total_points =
            static_cast<int>(laser_cloud_scans[scan]->points.size()) - 10;
        for (int point = 5;
             point <
                 static_cast<int>(laser_cloud_scans[scan]->points.size()) - 5;
             ++point)
        {
            const double diff_x =
                laser_cloud_scans[scan]->points[point - 5].x +
                laser_cloud_scans[scan]->points[point - 4].x +
                laser_cloud_scans[scan]->points[point - 3].x +
                laser_cloud_scans[scan]->points[point - 2].x +
                laser_cloud_scans[scan]->points[point - 1].x -
                10 * laser_cloud_scans[scan]->points[point].x +
                laser_cloud_scans[scan]->points[point + 1].x +
                laser_cloud_scans[scan]->points[point + 2].x +
                laser_cloud_scans[scan]->points[point + 3].x +
                laser_cloud_scans[scan]->points[point + 4].x +
                laser_cloud_scans[scan]->points[point + 5].x;
            const double diff_y =
                laser_cloud_scans[scan]->points[point - 5].y +
                laser_cloud_scans[scan]->points[point - 4].y +
                laser_cloud_scans[scan]->points[point - 3].y +
                laser_cloud_scans[scan]->points[point - 2].y +
                laser_cloud_scans[scan]->points[point - 1].y -
                10 * laser_cloud_scans[scan]->points[point].y +
                laser_cloud_scans[scan]->points[point + 1].y +
                laser_cloud_scans[scan]->points[point + 2].y +
                laser_cloud_scans[scan]->points[point + 3].y +
                laser_cloud_scans[scan]->points[point + 4].y +
                laser_cloud_scans[scan]->points[point + 5].y;
            const double diff_z =
                laser_cloud_scans[scan]->points[point - 5].z +
                laser_cloud_scans[scan]->points[point - 4].z +
                laser_cloud_scans[scan]->points[point - 3].z +
                laser_cloud_scans[scan]->points[point - 2].z +
                laser_cloud_scans[scan]->points[point - 1].z -
                10 * laser_cloud_scans[scan]->points[point].z +
                laser_cloud_scans[scan]->points[point + 1].z +
                laser_cloud_scans[scan]->points[point + 2].z +
                laser_cloud_scans[scan]->points[point + 3].z +
                laser_cloud_scans[scan]->points[point + 4].z +
                laser_cloud_scans[scan]->points[point + 5].z;
            cloud_curvature.emplace_back(
                point,
                diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
        }

        for (int sector = 0; sector < 6; ++sector)
        {
            const int sector_length = total_points / 6;
            const int sector_start = sector_length * sector;
            int sector_end = sector_length * (sector + 1) - 1;
            if (sector == 5)
                sector_end = total_points - 1;

            std::vector<Double2d> sector_curvature(
                cloud_curvature.begin() + sector_start,
                cloud_curvature.begin() + sector_end);
            featureExtractionFromSector(
                laser_cloud_scans[scan],
                sector_curvature,
                pc_out_edge,
                pc_out_surf);
        }
    }
}

void LaserProcessingClass::featureExtractionFromSector(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
    std::vector<Double2d>& cloud_curvature,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_edge,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_out_surf)
{
    // High-curvature points are selected from large to small. Once an edge is
    // accepted, nearby points on the same ring are suppressed until a depth
    // discontinuity is reached; this avoids nearly duplicate constraints.
    std::sort(
        cloud_curvature.begin(),
        cloud_curvature.end(),
        [](const Double2d& lhs, const Double2d& rhs)
        {
            return lhs.value < rhs.value;
        });

    int largest_picked_num = 0;
    std::vector<int> picked_points;
    for (int i = static_cast<int>(cloud_curvature.size()) - 1;
         i >= 0;
         --i)
    {
        const int index = cloud_curvature[i].id;
        if (std::find(
                picked_points.begin(),
                picked_points.end(),
                index) != picked_points.end())
        {
            continue;
        }
        if (cloud_curvature[i].value <= edge_curvature_threshold)
            break;

        ++largest_picked_num;
        picked_points.push_back(index);
        if (largest_picked_num <= edge_features_per_sector)
            pc_out_edge->push_back(pc_in->points[index]);
        else
            break;

        for (int offset = 1; offset <= 5; ++offset)
        {
            const double diff_x =
                pc_in->points[index + offset].x -
                pc_in->points[index + offset - 1].x;
            const double diff_y =
                pc_in->points[index + offset].y -
                pc_in->points[index + offset - 1].y;
            const double diff_z =
                pc_in->points[index + offset].z -
                pc_in->points[index + offset - 1].z;
            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z >
                edge_suppression_distance_sq)
                break;
            picked_points.push_back(index + offset);
        }
        for (int offset = -1; offset >= -5; --offset)
        {
            const double diff_x =
                pc_in->points[index + offset].x -
                pc_in->points[index + offset + 1].x;
            const double diff_y =
                pc_in->points[index + offset].y -
                pc_in->points[index + offset + 1].y;
            const double diff_z =
                pc_in->points[index + offset].z -
                pc_in->points[index + offset + 1].z;
            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z >
                edge_suppression_distance_sq)
                break;
            picked_points.push_back(index + offset);
        }
    }

    // Every non-suppressed sample becomes a surface candidate. The odometry
    // stage later downsamples this denser stream before plane fitting.
    for (const Double2d& curvature : cloud_curvature)
    {
        if (std::find(
                picked_points.begin(),
                picked_points.end(),
                curvature.id) == picked_points.end())
        {
            pc_out_surf->push_back(pc_in->points[curvature.id]);
        }
    }
}

Double2d::Double2d(int id_in, double value_in)
    : id(id_in),
      value(value_in)
{
}
