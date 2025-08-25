#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "trajlo/core/odometry.h"
#include "trajlo/utils/config.h"

// Nodo de ROS que envuelve la lógica de Traj-LO
class TrajLOROS : public rclcpp::Node {
public:
    TrajLOROS() : Node("trajlo_ros_node") {
        // Cargar configuración (simplificado, idealmente usar parámetros de ROS 2)
        this->declare_parameter<std::string>("config_path", "");
        std::string config_file = this->get_parameter("config_path").as_string();
        if (config_file.empty()) {
            RCLCPP_ERROR(this->get_logger(), "It needs the config file path!");
            return;
        }
        traj::TrajConfig config;
        config.load(config_file);

        // Inicializar odometría y su hilo de procesamiento
        odometry_ = std::make_shared<traj::TrajLOdometry>(config);
        odometry_->Start();

        // Suscriptor para la nube de puntos del LiDAR
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rubyplus_points", 1, std::bind(&TrajLOROS::lidarCallback, this, std::placeholders::_1));

        // Publicador para el mensaje de odometría
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/lidar_odometry/pose", 10);

        // Broadcaster para la transformada TF2 (odom -> base_link)
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // Temporizador para publicar la pose a una frecuencia fija
        publish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50), // Publicar a 25Hz
            std::bind(&TrajLOROS::publishPose, this));
        
        RCLCPP_INFO(this->get_logger(), "Traj-LO node initialized. Waiting for LiDAR data...");
    }

private:
    // Callback que se ejecuta al recibir un mensaje del LiDAR
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto scan = std::make_shared<traj::Scan>();
        scan->timestamp = rclcpp::Time(msg->header.stamp).nanoseconds();
        scan->size = msg->width * msg->height;
        scan->points.reserve(scan->size);

        // --- INICIO DE LA MODIFICACIÓN: LECTURA PRECISA DE TIMESTAMPS ---
        // Buscar el campo de timestamp en la nube de puntos.
        // Los drivers de Hesai suelen llamarlo 'timestamp'.
        int point_time_offset = -1;
        for (const auto& field : msg->fields) {
            if (field.name == "timestamp" || field.name == "t") {
                point_time_offset = field.offset;
                break;
            }
        }

        if (point_time_offset == -1) {
            RCLCPP_WARN_ONCE(this->get_logger(), "It haven't found per-point timestamp field ('timestamp' or 't'). Using linear simulation. Accuracy may be severely affected.");
        }

        sensor_msgs::PointCloud2Iterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(*msg, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_intensity(*msg, "intensity");

        for (size_t i = 0; i < scan->size; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
            traj::PointXYZIT point;
            point.x = *iter_x;
            point.y = *iter_y;
            point.z = *iter_z;
            point.intensity = *iter_intensity;

            if (point_time_offset != -1) {
                
                double point_timestamp_sec;
                memcpy(&point_timestamp_sec, &msg->data[i * msg->point_step + point_time_offset], sizeof(double));
                point.ts = point_timestamp_sec;
            } else {
                
                const double scan_duration = 0.1; // Asumir 10Hz
                double point_fraction = static_cast<double>(i) / static_cast<double>(scan->size);
                point.ts = rclcpp::Time(msg->header.stamp).seconds() + (point_fraction * scan_duration);
            }
            
            scan->points.push_back(point);
        }
        // --- FIN DE LA MODIFICACIÓN ---
        
        odometry_->pushScan(scan);
    }

    // Publica la última pose estimada
    void publishPose() {
        Sophus::SE3d current_pose;
        if (odometry_->getLatestPose(current_pose)) {
            auto now = this->get_clock()->now();
            const auto& trans = current_pose.translation();
            const auto& quat = current_pose.unit_quaternion();

            // 1. Publicar Transformada TF2 (odom -> rubyplus)
            geometry_msgs::msg::TransformStamped t;
 

            // 2. Publicar Mensaje de Odometría
            nav_msgs::msg::Odometry odom_msg;
  
            odom_msg.pose.pose.position.x = trans.x();
            odom_msg.pose.pose.position.y = trans.y();
            odom_msg.pose.pose.position.z = trans.z();
            odom_msg.pose.pose.orientation.x = quat.x();
            odom_msg.pose.pose.orientation.y = quat.y();
            odom_msg.pose.pose.orientation.z = quat.z();
            odom_msg.pose.pose.orientation.w = quat.w();
            odom_pub_->publish(odom_msg);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    traj::TrajLOdometry::Ptr odometry_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrajLOROS>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}