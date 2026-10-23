// RIA-LO feature extraction ROS node.

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "laserProcessingClass.h"
#include "lidar.h"

namespace
{
LaserProcessingClass laser_processing;
std::mutex queue_mutex;
std::condition_variable queue_condition;
std::queue<sensor_msgs::PointCloud2ConstPtr> point_queue;
std::atomic<bool> running(true);
lidar::Lidar lidar_param;

ros::Publisher edge_publisher;
ros::Publisher surf_publisher;
ros::Publisher filtered_publisher;

void velodyneHandler(const sensor_msgs::PointCloud2ConstPtr& message)
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        point_queue.push(message);
    }
    queue_condition.notify_one();
}

bool popCloud(
    sensor_msgs::PointCloud2ConstPtr& message)
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_condition.wait(lock, [] {
        return !running.load() || !point_queue.empty();
    });
    if (point_queue.empty())
        return false;
    message = point_queue.front();
    point_queue.pop();
    return true;
}

void processClouds()
{
    // The subscriber callback only enqueues messages. A single worker keeps
    // feature frames ordered and publishes edge/surface clouds with the exact
    // input timestamp required by the odometry synchronizer.
    while (running.load() || !point_queue.empty())
    {
        sensor_msgs::PointCloud2ConstPtr input_message;
        if (!popCloud(input_message))
        {
            if (!running.load())
                break;
            continue;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr input(
            new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr edge(
            new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr surf(
            new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(*input_message, *input);

        laser_processing.featureExtraction(input, edge, surf);

        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZI>());
        *filtered += *edge;
        *filtered += *surf;

        sensor_msgs::PointCloud2 filtered_message;
        pcl::toROSMsg(*filtered, filtered_message);
        filtered_message.header.stamp = input_message->header.stamp;
        filtered_message.header.frame_id = "base_link";
        filtered_publisher.publish(filtered_message);

        sensor_msgs::PointCloud2 edge_message;
        pcl::toROSMsg(*edge, edge_message);
        edge_message.header.stamp = input_message->header.stamp;
        edge_message.header.frame_id = "base_link";
        edge_publisher.publish(edge_message);

        sensor_msgs::PointCloud2 surf_message;
        pcl::toROSMsg(*surf, surf_message);
        surf_message.header.stamp = input_message->header.stamp;
        surf_message.header.frame_id = "base_link";
        surf_publisher.publish(surf_message);
    }
}
}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ria_lo_feature_extraction");
    ros::NodeHandle node;

    int scan_line = 64;
    double scan_period = 0.1;
    double max_distance = 60.0;
    double min_distance = 2.0;
    double edge_curvature_threshold = 0.1;
    int edge_features_per_sector = 20;
    double edge_suppression_distance_sq = 0.05;
    bool scan_phase_encoding = false;
    std::string input_topic = "/velodyne_points";
    node.getParam("/scan_period", scan_period);
    node.getParam("/max_dis", max_distance);
    node.getParam("/min_dis", min_distance);
    node.getParam("/scan_line", scan_line);
    node.param(
        "/edge_curvature_threshold",
        edge_curvature_threshold,
        edge_curvature_threshold);
    node.param(
        "/edge_features_per_sector",
        edge_features_per_sector,
        edge_features_per_sector);
    node.param(
        "/edge_suppression_distance_sq",
        edge_suppression_distance_sq,
        edge_suppression_distance_sq);
    node.param(
        "/scan_phase_encoding",
        scan_phase_encoding,
        scan_phase_encoding);
    node.param<std::string>("/topic", input_topic, input_topic);

    lidar_param.setScanPeriod(scan_period);
    lidar_param.setLines(scan_line);
    lidar_param.setMaxDistance(max_distance);
    lidar_param.setMinDistance(min_distance);
    laser_processing.init(lidar_param);
    laser_processing.configureFeatureProfile(
        edge_curvature_threshold,
        edge_features_per_sector,
        edge_suppression_distance_sq);
    laser_processing.configureScanPhaseEncoding(scan_phase_encoding);
    ROS_INFO(
        "RIA-LO feature profile: curvature threshold %.6f, "
        "edge features/sector %d, suppression distance squared %.6f",
        edge_curvature_threshold,
        edge_features_per_sector,
        edge_suppression_distance_sq);
    ROS_INFO(
        "RIA-LO scan phase encoding: %s, input topic: %s",
        scan_phase_encoding ? "on" : "off",
        input_topic.c_str());

    ros::Subscriber subscriber =
        node.subscribe<sensor_msgs::PointCloud2>(
            input_topic, 1000, velodyneHandler);
    filtered_publisher =
        node.advertise<sensor_msgs::PointCloud2>(
            "/velodyne_points_filtered", 1000);
    edge_publisher =
        node.advertise<sensor_msgs::PointCloud2>(
            "/laser_cloud_edge", 1000);
    surf_publisher =
        node.advertise<sensor_msgs::PointCloud2>(
            "/laser_cloud_surf", 1000);

    std::thread worker(processClouds);
    ros::spin();
    running.store(false);
    queue_condition.notify_all();
    worker.join();
    return 0;
}
