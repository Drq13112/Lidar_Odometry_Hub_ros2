// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <Eigen/Core>
#include <memory>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

// KISS-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

#include <chrono> 

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS 2 headers
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

namespace {
Sophus::SE3d LookupTransform(const std::string &target_frame,
                             const std::string &source_frame,
                             const std::unique_ptr<tf2_ros::Buffer> &tf2_buffer) {
    std::string err_msg;
    if (tf2_buffer->canTransform(target_frame, source_frame, tf2::TimePointZero, &err_msg)) {
        try {
            auto tf = tf2_buffer->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
            return tf2::transformToSophus(tf);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(rclcpp::get_logger("LookupTransform"), "%s", ex.what());
        }
    }
    RCLCPP_WARN(rclcpp::get_logger("LookupTransform"), "Failed to find tf. Reason=%s",
                err_msg.c_str());
    // default construction is the identity
    return Sophus::SE3d();
}
}  // namespace

namespace kiss_icp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

OdometryServer::OdometryServer(const rclcpp::NodeOptions &options)
    : rclcpp::Node("kiss_icp_node", options) {
    kiss_icp::pipeline::KISSConfig config;
    initializeParameters(config);

    // Construct the main KISS-ICP odometry node
    kiss_icp_ = std::make_unique<kiss_icp::pipeline::KissICP>(config);

    // Initialize subscribers
    pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud_topic", rclcpp::SensorDataQoS(),
        std::bind(&OdometryServer::RegisterFrame, this, std::placeholders::_1));

    // Initialize publishers
    rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("kiss/odometry", qos);
    if (publish_debug_clouds_) {
        frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/frame", qos);
        kpoints_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/keypoints", qos);
        map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("kiss/local_map", qos);
    }

    // Initialize the transform broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf2_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf2_buffer_->setUsingDedicatedThread(true);
    tf2_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf2_buffer_);

    RCLCPP_INFO(this->get_logger(), "KISS-ICP ROS 2 odometry node initialized");
}

void OdometryServer::initializeParameters(kiss_icp::pipeline::KISSConfig &config) {
    RCLCPP_INFO(this->get_logger(), "Initializing parameters");

    base_frame_ = declare_parameter<std::string>("base_frame", base_frame_);
    RCLCPP_INFO(this->get_logger(), "\tBase frame: %s", base_frame_.c_str());
    lidar_odom_frame_ = declare_parameter<std::string>("lidar_odom_frame", lidar_odom_frame_);
    RCLCPP_INFO(this->get_logger(), "\tLiDAR odometry frame: %s", lidar_odom_frame_.c_str());
    publish_odom_tf_ = declare_parameter<bool>("publish_odom_tf", publish_odom_tf_);
    RCLCPP_INFO(this->get_logger(), "\tPublish odometry transform: %d", publish_odom_tf_);
    invert_odom_tf_ = declare_parameter<bool>("invert_odom_tf", invert_odom_tf_);
    RCLCPP_INFO(this->get_logger(), "\tInvert odometry transform: %d", invert_odom_tf_);
    publish_debug_clouds_ = declare_parameter<bool>("publish_debug_clouds", publish_debug_clouds_);
    RCLCPP_INFO(this->get_logger(), "\tPublish debug clouds: %d", publish_debug_clouds_);
    position_covariance_ = declare_parameter<double>("position_covariance", 0.1);
    RCLCPP_INFO(this->get_logger(), "\tPosition covariance: %.2f", position_covariance_);
    orientation_covariance_ = declare_parameter<double>("orientation_covariance", 0.1);
    RCLCPP_INFO(this->get_logger(), "\tOrientation covariance: %.2f", orientation_covariance_);

    config.max_range = declare_parameter<double>("data.max_range", config.max_range);
    RCLCPP_INFO(this->get_logger(), "\tMax range: %.2f", config.max_range);
    config.min_range = declare_parameter<double>("data.min_range", config.min_range);
    RCLCPP_INFO(this->get_logger(), "\tMin range: %.2f", config.min_range);
    config.deskew = declare_parameter<bool>("data.deskew", config.deskew);
    RCLCPP_INFO(this->get_logger(), "\tDeskew: %d", config.deskew);
    config.voxel_size = declare_parameter<double>("mapping.voxel_size", config.max_range / 100.0);
    RCLCPP_INFO(this->get_logger(), "\tVoxel size: %.2f", config.voxel_size);
    config.max_points_per_voxel =
        declare_parameter<int>("mapping.max_points_per_voxel", config.max_points_per_voxel);
    RCLCPP_INFO(this->get_logger(), "\tMax points per voxel: %d", config.max_points_per_voxel);
    config.initial_threshold =
        declare_parameter<double>("adaptive_threshold.initial_threshold", config.initial_threshold);
    RCLCPP_INFO(this->get_logger(), "\tInitial threshold: %.2f", config.initial_threshold);
    config.min_motion_th =
        declare_parameter<double>("adaptive_threshold.min_motion_th", config.min_motion_th);
    RCLCPP_INFO(this->get_logger(), "\tMin motion threshold: %.2f", config.min_motion_th);
    config.max_num_iterations =
        declare_parameter<int>("registration.max_num_iterations", config.max_num_iterations);
    RCLCPP_INFO(this->get_logger(), "\tMax number of iterations: %d", config.max_num_iterations);
    config.convergence_criterion = declare_parameter<double>("registration.convergence_criterion",
                                                             config.convergence_criterion);
    RCLCPP_INFO(this->get_logger(), "\tConvergence criterion: %.2f", config.convergence_criterion);
    config.max_num_threads =
        declare_parameter<int>("registration.max_num_threads", config.max_num_threads);
    RCLCPP_INFO(this->get_logger(), "\tMax number of threads: %d", config.max_num_threads);
    if (config.max_range < config.min_range) {
        RCLCPP_WARN(get_logger(),
                    "[WARNING] max_range is smaller than min_range, settng min_range to 0.0");
        config.min_range = 0.0;
    }
}

void OdometryServer::RegisterFrame(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    static double total_processing_time_ms = 0.0;
    static double total_conversion_time_ms = 0.0;
    static long valid_frame_count = 0;
    static auto last_frame_time = std::chrono::high_resolution_clock::now();
    static bool first_frame = true;

    const auto current_time = std::chrono::high_resolution_clock::now();
    
    // Solo calcular frecuencia después del primer frame
    if (!first_frame) {
        const auto frame_interval_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            current_time - last_frame_time).count();
        const double frequency_hz = 1000.0 / frame_interval_ms;

        RCLCPP_INFO(this->get_logger(), "Frecuencia: %.1f Hz, Intervalo: %.2f ms", 
                frequency_hz, frame_interval_ms);
    } else {
        first_frame = false;
    }
    last_frame_time = current_time;
    
    const auto start_total = std::chrono::high_resolution_clock::now();
    
    // Medir conversión de datos
    const auto start_conversion = std::chrono::high_resolution_clock::now();
    const auto cloud_frame_id = msg->header.frame_id;
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = GetTimestamps(msg);
    const auto end_conversion = std::chrono::high_resolution_clock::now();
    
    // Medir procesamiento principal
    const auto start_processing = std::chrono::high_resolution_clock::now();
    const auto &[frame, keypoints] = kiss_icp_->RegisterFrame(points, timestamps);
    const auto end_processing = std::chrono::high_resolution_clock::now();
    
    const auto end_total = std::chrono::high_resolution_clock::now();
    
    // Calcular duraciones
    const auto conversion_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        end_conversion - start_conversion).count();
    const auto processing_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        end_processing - start_processing).count();
    const auto total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        end_total - start_total).count();
    
    if (processing_ms > 1e-3) {
        valid_frame_count++;
        total_processing_time_ms += processing_ms;
        total_conversion_time_ms += conversion_ms;
        
        const double avg_processing_ms = total_processing_time_ms / valid_frame_count;
        const double avg_conversion_ms = total_conversion_time_ms / valid_frame_count;
        
        RCLCPP_INFO(this->get_logger(), 
                   "Tiempos - Total: %.2f ms, Procesamiento: %.2f ms (avg: %.2f), "
                   "Conversión: %.2f ms (avg: %.2f), Puntos: %zu",
                   total_ms, processing_ms, avg_processing_ms, 
                   conversion_ms, avg_conversion_ms, points.size());
    }

    // Extract the last KISS-ICP pose, ego-centric to the LiDAR
    const Sophus::SE3d kiss_pose = kiss_icp_->pose();

    // Spit the current estimated pose to ROS msgs handling the desired target frame
    PublishOdometry(kiss_pose, msg->header);
    // Publishing these clouds is a bit costly, so do it only if we are debugging
    if (publish_debug_clouds_) {
        PublishClouds(frame, keypoints, msg->header);
    }
}

void OdometryServer::PublishOdometry(const Sophus::SE3d &kiss_pose,
                                     const std_msgs::msg::Header &header) {
    // If necessary, transform the ego-centric pose to the specified base_link/base_footprint frame
    const auto cloud_frame_id = header.frame_id;
    const auto egocentric_estimation = (base_frame_.empty() || base_frame_ == cloud_frame_id);
    const auto moving_frame = egocentric_estimation ? cloud_frame_id : base_frame_;
    const auto pose = [&]() -> Sophus::SE3d {
        if (egocentric_estimation) return kiss_pose;
        const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id, tf2_buffer_);
        return cloud2base * kiss_pose * cloud2base.inverse();
    }();

    // Broadcast the tf ---
    if (publish_odom_tf_) {
        geometry_msgs::msg::TransformStamped transform_msg;
        transform_msg.header.stamp = header.stamp;
        if (invert_odom_tf_) {
            transform_msg.header.frame_id = moving_frame;
            transform_msg.child_frame_id = lidar_odom_frame_;
            transform_msg.transform = tf2::sophusToTransform(pose.inverse());
        } else {
            transform_msg.header.frame_id = lidar_odom_frame_;
            transform_msg.child_frame_id = moving_frame;
            transform_msg.transform = tf2::sophusToTransform(pose);
        }
        tf_broadcaster_->sendTransform(transform_msg);
    }

    // publish odometry msg
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = header.stamp;
    odom_msg.header.frame_id = lidar_odom_frame_;
    odom_msg.child_frame_id = moving_frame;
    odom_msg.pose.pose = tf2::sophusToPose(pose);
    odom_msg.pose.covariance.fill(0.0);
    odom_msg.pose.covariance[0] = position_covariance_;
    odom_msg.pose.covariance[7] = position_covariance_;
    odom_msg.pose.covariance[14] = position_covariance_;
    odom_msg.pose.covariance[21] = orientation_covariance_;
    odom_msg.pose.covariance[28] = orientation_covariance_;
    odom_msg.pose.covariance[35] = orientation_covariance_;
    odom_publisher_->publish(std::move(odom_msg));
}

void OdometryServer::PublishClouds(const std::vector<Eigen::Vector3d> &frame,
                                   const std::vector<Eigen::Vector3d> &keypoints,
                                   const std_msgs::msg::Header &header) {
    const auto kiss_map = kiss_icp_->LocalMap();

    frame_publisher_->publish(std::move(EigenToPointCloud2(frame, header)));
    kpoints_publisher_->publish(std::move(EigenToPointCloud2(keypoints, header)));

    auto local_map_header = header;
    local_map_header.frame_id = lidar_odom_frame_;
    map_publisher_->publish(std::move(EigenToPointCloud2(kiss_map, local_map_header)));
}
}  // namespace kiss_icp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kiss_icp_ros::OdometryServer)
