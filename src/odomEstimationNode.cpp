// RIA-LO scan-to-map odometry ROS node.

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "lidar.h"
#include "odomEstimationClass.h"

namespace
{
OdomEstimationClass odom_estimation;
std::mutex queue_mutex;
std::condition_variable queue_condition;
std::queue<sensor_msgs::PointCloud2ConstPtr> edge_queue;
std::queue<sensor_msgs::PointCloud2ConstPtr> surf_queue;
std::atomic<bool> running(true);
lidar::Lidar lidar_param;
ros::Publisher odometry_publisher;

void surfHandler(const sensor_msgs::PointCloud2ConstPtr& message)
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        surf_queue.push(message);
    }
    queue_condition.notify_one();
}

void edgeHandler(const sensor_msgs::PointCloud2ConstPtr& message)
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        edge_queue.push(message);
    }
    queue_condition.notify_one();
}

bool popAlignedPair(
    sensor_msgs::PointCloud2ConstPtr& edge,
    sensor_msgs::PointCloud2ConstPtr& surf)
{
    // Preserve every aligned feature pair. Frames are discarded only when
    // their timestamps cannot represent the same LiDAR scan.
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_condition.wait(lock, [] {
        return !running.load() || (!edge_queue.empty() && !surf_queue.empty());
    });

    while (!edge_queue.empty() && !surf_queue.empty())
    {
        const double edge_time = edge_queue.front()->header.stamp.toSec();
        const double surf_time = surf_queue.front()->header.stamp.toSec();
        const double tolerance = 0.5 * lidar_param.scan_period;
        if (surf_time < edge_time - tolerance)
        {
            surf_queue.pop();
            ROS_WARN_THROTTLE(
                2.0, "RIA-LO discarded an unaligned surface cloud");
            continue;
        }
        if (edge_time < surf_time - tolerance)
        {
            edge_queue.pop();
            ROS_WARN_THROTTLE(
                2.0, "RIA-LO discarded an unaligned edge cloud");
            continue;
        }

        edge = edge_queue.front();
        surf = surf_queue.front();
        edge_queue.pop();
        surf_queue.pop();
        return true;
    }
    return false;
}

void publishOdometry(const ros::Time& stamp)
{
    const Eigen::Quaterniond orientation(odom_estimation.odom.rotation());
    const Eigen::Vector3d position = odom_estimation.odom.translation();

    static tf::TransformBroadcaster broadcaster;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(position.x(), position.y(), position.z()));
    transform.setRotation(tf::Quaternion(
        orientation.x(),
        orientation.y(),
        orientation.z(),
        orientation.w()));
    broadcaster.sendTransform(
        tf::StampedTransform(transform, stamp, "map", "base_link"));

    nav_msgs::Odometry message;
    message.header.frame_id = "map";
    message.child_frame_id = "base_link";
    message.header.stamp = stamp;
    message.pose.pose.orientation.x = orientation.x();
    message.pose.pose.orientation.y = orientation.y();
    message.pose.pose.orientation.z = orientation.z();
    message.pose.pose.orientation.w = orientation.w();
    message.pose.pose.position.x = position.x();
    message.pose.pose.position.y = position.y();
    message.pose.pose.position.z = position.z();
    odometry_publisher.publish(message);
}

void processOdometry()
{
    // Runtime data path:
    // aligned edge/surface clouds -> first-frame map initialization ->
    // RIA-LO scan-to-map update -> /odom and map-to-base_link output.
    bool initialized = false;
    pcl::PointCloud<pcl::PointXYZI>::Ptr temporary(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<PointType>::Ptr edge_cloud(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surf_cloud(
        new pcl::PointCloud<PointType>());

    while (running.load() || !edge_queue.empty() || !surf_queue.empty())
    {
        sensor_msgs::PointCloud2ConstPtr edge_message;
        sensor_msgs::PointCloud2ConstPtr surf_message;
        if (!popAlignedPair(edge_message, surf_message))
        {
            if (!running.load())
                break;
            continue;
        }

        temporary->clear();
        edge_cloud->clear();
        pcl::fromROSMsg(*edge_message, *temporary);
        pcl::copyPointCloud(*temporary, *edge_cloud);
        temporary->clear();
        surf_cloud->clear();
        pcl::fromROSMsg(*surf_message, *temporary);
        pcl::copyPointCloud(*temporary, *surf_cloud);

        if (!initialized)
        {
            odom_estimation.initMapWithPoints(edge_cloud, surf_cloud);
            initialized = true;
            ROS_INFO("RIA-LO odometry initialized");
        }
        else
        {
            odom_estimation.updatePointsToMap(edge_cloud, surf_cloud);
        }

        publishOdometry(surf_message->header.stamp);
    }
}
}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ria_lo_odometry");
    ros::NodeHandle node;
    ros::NodeHandle private_node("~");

    int scan_line = 64;
    double scan_period = 0.1;
    double max_distance = 60.0;
    double min_distance = 2.0;
    double map_resolution = 0.4;
    double surface_resolution_scale = 2.0;
    int optimization_iterations = 2;
    int ceres_max_iterations = 4;
    double robust_delta_scale_multiplier = 1.5;
    double robust_delta_min = 0.055;
    double robust_delta_max = 0.15;
    double direction_balance_exponent = 0.5;
    double direction_weight_min = 0.65;
    double direction_weight_max = 1.55;
    bool scan_motion_compensation = false;
    double motion_reference_phase = 0.5;
    double motion_compensation_scale = 1.0;
    double motion_translation_scale = 1.0;
    double motion_rotation_scale = 1.0;
    int motion_compensation_warmup_frames = 10;
    bool adaptive_motion_scaling = false;
    double high_turn_translation_scale = 0.10;
    double high_turn_rotation_scale = 0.20;
    double motion_turn_start = 0.004;
    double motion_turn_end = 0.008;
    double motion_turn_smoothing = 0.20;
    double motion_turn_duty_start = 0.45;
    double motion_turn_duty_end = 0.60;
    double motion_turn_duty_smoothing = 0.02;
    std::string pipeline_mode = "full";
    bool weighting_adaptive = false;
    int weighting_enter_constraints = 900;
    int weighting_exit_constraints = 1300;
    int weighting_enter_direction_bins = 8;
    int weighting_exit_direction_bins = 10;
    node.getParam("/scan_period", scan_period);
    node.getParam("/max_dis", max_distance);
    node.getParam("/min_dis", min_distance);
    node.getParam("/scan_line", scan_line);
    node.getParam("/map_resolution", map_resolution);
    node.param(
        "/surface_resolution_scale",
        surface_resolution_scale,
        surface_resolution_scale);
    node.param(
        "/optimization_iterations",
        optimization_iterations,
        optimization_iterations);
    node.param(
        "/ceres_max_iterations",
        ceres_max_iterations,
        ceres_max_iterations);
    node.param(
        "/robust_delta_scale_multiplier",
        robust_delta_scale_multiplier,
        robust_delta_scale_multiplier);
    node.param(
        "/robust_delta_min",
        robust_delta_min,
        robust_delta_min);
    node.param(
        "/robust_delta_max",
        robust_delta_max,
        robust_delta_max);
    node.param(
        "/direction_balance_exponent",
        direction_balance_exponent,
        direction_balance_exponent);
    node.param(
        "/direction_weight_min",
        direction_weight_min,
        direction_weight_min);
    node.param(
        "/direction_weight_max",
        direction_weight_max,
        direction_weight_max);
    node.param(
        "/scan_motion_compensation",
        scan_motion_compensation,
        scan_motion_compensation);
    node.param(
        "/motion_reference_phase",
        motion_reference_phase,
        motion_reference_phase);
    node.param(
        "/motion_compensation_scale",
        motion_compensation_scale,
        motion_compensation_scale);
    motion_translation_scale = motion_compensation_scale;
    motion_rotation_scale = motion_compensation_scale;
    node.param(
        "/motion_translation_scale",
        motion_translation_scale,
        motion_translation_scale);
    node.param(
        "/motion_rotation_scale",
        motion_rotation_scale,
        motion_rotation_scale);
    node.param(
        "/motion_compensation_warmup_frames",
        motion_compensation_warmup_frames,
        motion_compensation_warmup_frames);
    node.param(
        "/adaptive_motion_scaling",
        adaptive_motion_scaling,
        adaptive_motion_scaling);
    node.param(
        "/high_turn_translation_scale",
        high_turn_translation_scale,
        high_turn_translation_scale);
    node.param(
        "/high_turn_rotation_scale",
        high_turn_rotation_scale,
        high_turn_rotation_scale);
    node.param(
        "/motion_turn_start",
        motion_turn_start,
        motion_turn_start);
    node.param(
        "/motion_turn_end",
        motion_turn_end,
        motion_turn_end);
    node.param(
        "/motion_turn_smoothing",
        motion_turn_smoothing,
        motion_turn_smoothing);
    node.param(
        "/motion_turn_duty_start",
        motion_turn_duty_start,
        motion_turn_duty_start);
    node.param(
        "/motion_turn_duty_end",
        motion_turn_duty_end,
        motion_turn_duty_end);
    node.param(
        "/motion_turn_duty_smoothing",
        motion_turn_duty_smoothing,
        motion_turn_duty_smoothing);
    private_node.param<std::string>(
        "pipeline_mode", pipeline_mode, pipeline_mode);
    node.param(
        "/weighting_adaptive", weighting_adaptive, weighting_adaptive);
    node.param(
        "/weighting_enter_constraints",
        weighting_enter_constraints,
        weighting_enter_constraints);
    node.param(
        "/weighting_exit_constraints",
        weighting_exit_constraints,
        weighting_exit_constraints);
    node.param(
        "/weighting_enter_direction_bins",
        weighting_enter_direction_bins,
        weighting_enter_direction_bins);
    node.param(
        "/weighting_exit_direction_bins",
        weighting_exit_direction_bins,
        weighting_exit_direction_bins);

    lidar_param.setScanPeriod(scan_period);
    lidar_param.setLines(scan_line);
    lidar_param.setMaxDistance(max_distance);
    lidar_param.setMinDistance(min_distance);
    odom_estimation.configureAccuracyProfile(
        surface_resolution_scale,
        optimization_iterations,
        ceres_max_iterations,
        robust_delta_scale_multiplier,
        robust_delta_min,
        robust_delta_max,
        direction_balance_exponent,
        direction_weight_min,
        direction_weight_max);
    odom_estimation.init(lidar_param, map_resolution, pipeline_mode);
    odom_estimation.configureAdaptiveWeighting(
        weighting_adaptive,
        weighting_enter_constraints,
        weighting_exit_constraints,
        weighting_enter_direction_bins,
        weighting_exit_direction_bins);
    odom_estimation.configureScanMotionCompensation(
        scan_motion_compensation,
        motion_reference_phase,
        motion_translation_scale,
        motion_rotation_scale,
        motion_compensation_warmup_frames);
    odom_estimation.configureAdaptiveMotionScaling(
        adaptive_motion_scaling,
        high_turn_translation_scale,
        high_turn_rotation_scale,
        motion_turn_start,
        motion_turn_end,
        motion_turn_smoothing,
        motion_turn_duty_start,
        motion_turn_duty_end,
        motion_turn_duty_smoothing);
    ROS_INFO(
        "RIA-LO accuracy profile: surface scale %.3f, "
        "outer iterations %d, Ceres iterations %d, "
        "robust delta median + %.3f*scale, clamp [%.3f, %.3f], "
        "direction balance exponent %.3f, clamp [%.3f, %.3f]",
        surface_resolution_scale,
        optimization_iterations,
        ceres_max_iterations,
        robust_delta_scale_multiplier,
        robust_delta_min,
        robust_delta_max,
        direction_balance_exponent,
        direction_weight_min,
        direction_weight_max);
    ROS_INFO(
        "RIA-LO pipeline mode: %s, adaptive weighting: %s "
        "(constraints %d/%d, direction bins %d/%d)",
        pipeline_mode.c_str(),
        weighting_adaptive ? "on" : "off",
        weighting_enter_constraints,
        weighting_exit_constraints,
        weighting_enter_direction_bins,
        weighting_exit_direction_bins);
    ROS_INFO(
        "RIA-LO scan motion compensation: %s, reference %.3f, "
        "translation scale %.3f, rotation scale %.3f, warmup %d frames",
        scan_motion_compensation ? "on" : "off",
        motion_reference_phase,
        motion_translation_scale,
        motion_rotation_scale,
        motion_compensation_warmup_frames);
    ROS_INFO(
        "RIA-LO adaptive motion scaling: %s, high-turn scales "
        "(translation %.3f, rotation %.3f), turn range [%.5f, %.5f], "
        "smoothing %.3f, duty range [%.3f, %.3f], duty smoothing %.3f",
        adaptive_motion_scaling ? "on" : "off",
        high_turn_translation_scale,
        high_turn_rotation_scale,
        motion_turn_start,
        motion_turn_end,
        motion_turn_smoothing,
        motion_turn_duty_start,
        motion_turn_duty_end,
        motion_turn_duty_smoothing);

    ros::Subscriber edge_subscriber =
        node.subscribe<sensor_msgs::PointCloud2>(
            "/laser_cloud_edge", 1000, edgeHandler);
    ros::Subscriber surf_subscriber =
        node.subscribe<sensor_msgs::PointCloud2>(
            "/laser_cloud_surf", 1000, surfHandler);
    odometry_publisher = node.advertise<nav_msgs::Odometry>("/odom", 1000);

    std::thread worker(processOdometry);
    ros::spin();
    running.store(false);
    queue_condition.notify_all();
    worker.join();
    return 0;
}
