#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Vector3.hpp>
#include <fstream>
#include <chrono>

#include "PointCloud.hpp"
#include "PointMap.hpp"
#include "Register.hpp"

using std::placeholders::_1;

class LidarOdometry : public rclcpp::Node {
public:
  LidarOdometry() : Node("SiMpLE_lidar_odometry") {
    subPointcloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "rubyplus_points", 10, std::bind(&LidarOdometry::pointcloudCallback, this, _1));
    pubOdom_ = this->create_publisher<nav_msgs::msg::Odometry>("/lidar_odometry/pose", 20);
    file_time_path_ = this->declare_parameter<std::string>("timing_log_path", "/home/david/ros2_ws/processing_times_simple.csv");
    rNew_ = this->declare_parameter<float>("rNew", 0.5);
    float rMap = this->declare_parameter<float>("rMap", 2.0);
    rMin_ = this->declare_parameter<float>("rMin", 5.0);
    rMax_ = this->declare_parameter<float>("rMax", 120);
    float sigma = this->declare_parameter<float>("sigma", 0.3);
    float epsilon = this->declare_parameter<float>("epsilon", 1e-3);
    odomMessage_.header.frame_id = this->declare_parameter<std::string>("odom_frame", "odom");

    subMap_ = std::make_unique<PointMap>(rMap, rMax_);
    scanToMapRegister_ = std::make_unique<Register>(epsilon, sigma);

    // Abrir archivo CSV para guardar los tiempos de procesamiento.
    processing_time_file_.open(file_time_path_, std::ios::out);
    if (!processing_time_file_.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "No se pudo abrir %s para escritura.",
                   file_time_path_.c_str());
    } else {
      processing_time_file_ << "timestamp,processing_time_ms\n";
    }
  }

  ~LidarOdometry() {
    if(processing_time_file_.is_open()){
      processing_time_file_.close();
    }
  }

private:
  void pointcloudCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Inicia la medición del tiempo de procesamiento.
    auto start_time = std::chrono::steady_clock::now();

    PointCloud newScan = PointCloud(rNew_, rMax_, rMin_, false);

    sensor_msgs::PointCloud2Iterator<float> iterX(*msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iterY(*msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iterZ(*msg, "z");

    for (; iterX != iterX.end(); ++iterX, ++iterY, ++iterZ) {
      newScan.addPoint(*iterX, *iterY, *iterZ);
    }

    newScan.processPointCloud();

    double x = 0, y = 0, z = 0, roll = 0, pitch = 0, yaw = 0;

    if (initialised_) {
      double timePassed = (msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9)
                         - (odomMessage_.header.stamp.sec + odomMessage_.header.stamp.nanosec / 1e9);
      scanToMapRegister_->registerScan(newScan.getPtCloud(), subMap_->getPcForKdTree());
      column_vector res = scanToMapRegister_->getRegResult();

      roll = res(0);
      pitch = res(1);
      yaw = res(2);
      x = res(3);
      y = res(4);
      z = res(5);

      tf2::Quaternion orientation, past_orientation;
      orientation.setRPY(roll, pitch, yaw);
      tf2::fromMsg(odomMessage_.pose.pose.orientation, past_orientation);

      tf2::Vector3 relativeTranslation = tf2::Vector3(x - odomMessage_.pose.pose.position.x,
                                                        y - odomMessage_.pose.pose.position.y,
                                                        z - odomMessage_.pose.pose.position.z);
      tf2::Vector3 transformedLinearVelocity = tf2::quatRotate(orientation.inverse(), relativeTranslation / timePassed);

      tf2::Quaternion relativeRotationQuat = orientation * past_orientation.inverse();
      tf2::Matrix3x3 m(relativeRotationQuat);
      tf2::Vector3 relativeRotation;
      m.getRPY(relativeRotation[0], relativeRotation[1], relativeRotation[2]);
      tf2::Vector3 transformedAngularVelocity = tf2::quatRotate(orientation.inverse(), relativeRotation / timePassed);

      odomMessage_.pose.pose.position.x = x;
      odomMessage_.pose.pose.position.y = y;
      odomMessage_.pose.pose.position.z = z;
      odomMessage_.twist.twist.linear.x = transformedLinearVelocity[0];
      odomMessage_.twist.twist.linear.y = transformedLinearVelocity[1];
      odomMessage_.twist.twist.linear.z = transformedLinearVelocity[2];

      odomMessage_.pose.pose.orientation = tf2::toMsg(orientation);
      odomMessage_.twist.twist.angular.x = transformedAngularVelocity[0];
      odomMessage_.twist.twist.angular.y = transformedAngularVelocity[1];
      odomMessage_.twist.twist.angular.z = transformedAngularVelocity[2];
    }

    Eigen::Matrix4d hypothesis = utils::homogeneous(roll, pitch, yaw, x, y, z);

    subMap_->updateMap(newScan.getPtCloud(), hypothesis);

    odomMessage_.header.stamp = msg->header.stamp;
    odomMessage_.child_frame_id = msg->header.frame_id;

    this->pubOdom_->publish(odomMessage_);

    initialised_ = true;

    // Finaliza la medición del tiempo y calcula la duración en milisegundos.
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Guarda el tiempo de procesamiento en el CSV.
    if (processing_time_file_.is_open()) {
      // Se usa la marca de tiempo actual del nodo para la línea.
      processing_time_file_ << this->now().seconds() << "," << duration_ms << "\n";
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPointcloud_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdom_;
  nav_msgs::msg::Odometry odomMessage_;

  float rNew_, rMin_, rMax_;
  bool initialised_ = false;

  std::unique_ptr<PointMap> subMap_;
  std::unique_ptr<Register> scanToMapRegister_;
  std::ofstream processing_time_file_;
  std::string file_time_path_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarOdometry>());
  rclcpp::shutdown();
  return 0;
}