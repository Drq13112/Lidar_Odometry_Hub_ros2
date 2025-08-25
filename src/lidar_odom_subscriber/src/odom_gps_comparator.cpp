#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cmath>
#include <mutex>
#include <fstream>
#include <vector>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <Eigen/Dense>

class OdomGpsComparator : public rclcpp::Node
{
public:
    OdomGpsComparator()
        : Node("odom_gps_comparator"), got_initial_gps_(false), got_initial_lidar_(false)
    {
        // Parámetros para los topics
        this->declare_parameter<bool>("calibrate_angle", true);
        calibrate_angle_ = this->get_parameter("calibrate_angle").as_bool();
        this->declare_parameter<std::string>("lidar_topic", "/lidar_odometry/pose");
        std::string lidar_topic = this->get_parameter("lidar_topic").as_string();
        this->declare_parameter<std::string>("gps_topic", "/zoe/localization/global");
        std::string gps_topic = this->get_parameter("gps_topic").as_string();

        // Subscripciones
        lidar_odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            lidar_topic, 1,
            std::bind(&OdomGpsComparator::lidar_odometry_callback, this, std::placeholders::_1));
        gps_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            gps_topic, 1,
            std::bind(&OdomGpsComparator::gps_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Nodo comparador iniciado. Esperando datos...");
    }

    ~OdomGpsComparator()
    {
        if (est_traj_.empty() || gt_traj_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No hay datos de trayectoria para guardar o graficar.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Guardando trayectorias en /home/david/ros2_ws/ ...");
        std::ofstream est_file("/home/david/ros2_ws/odom_est_traj.csv");
        std::ofstream gt_file("/home/david/ros2_ws/odom_gt_traj.csv");

        // Normalizar las trayectorias para que empiecen en (0,0)
        if (got_initial_gps_)
        {
            for (size_t i = 0; i < est_traj_.size(); ++i)
            {
                est_file << est_traj_[i].x - initial_gps_.x << "," << est_traj_[i].y - initial_gps_.y << "," << est_traj_[i].z - initial_gps_.z << std::endl;
                if (i < gt_traj_.size())
                {
                    gt_file << gt_traj_[i].x - initial_gps_.x << "," << gt_traj_[i].y - initial_gps_.y << "," << gt_traj_[i].z - initial_gps_.z << std::endl;
                }
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "No se recibió la pose inicial del GPS. Las trayectorias no se normalizarán.");
            for (size_t i = 0; i < est_traj_.size(); ++i)
            {
                est_file << est_traj_[i].x << "," << est_traj_[i].y << "," << est_traj_[i].z << std::endl;
                if (i < gt_traj_.size())
                {
                    gt_file << gt_traj_[i].x << "," << gt_traj_[i].y << "," << gt_traj_[i].z << std::endl;
                }
            }
        }

        est_file.close();
        gt_file.close();
        RCLCPP_INFO(this->get_logger(), "Trayectorias guardadas.");
        RCLCPP_INFO(this->get_logger(), "Generando gráfica de comparación...");
        try
        {
            std::string package_path = ament_index_cpp::get_package_share_directory("lidar_odom_subscriber");
            std::string script_path = package_path + "/scripts/plot_trajectories.py";
            std::string command = "python3 " + script_path;
            int result = std::system(command.c_str());
            if (result != 0)
            {
                RCLCPP_ERROR(this->get_logger(), "Falló la ejecución del script de ploteo.");
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "No se pudo encontrar la ruta del paquete 'lidar_odom_subscriber'.");
        }
    }

private:
    void gps_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_gps_ = msg->pose.pose.position;

        if (!got_initial_gps_)
        {
            initial_gps_ = msg->pose.pose.position;
            initial_gps_yaw_ = msg->pose.pose.orientation.z * M_PI / 180.0; // El yaw del GPS viene dentro del eje Z de la orientación
            got_initial_gps_ = true;
            RCLCPP_INFO(this->get_logger(), "Initial GPS UTM set: (%.3f, %.3f), Yaw: %.3f deg",
                        initial_gps_.x, initial_gps_.y, initial_gps_yaw_ * 180.0 / M_PI);
        }
        
    }

    void lidar_odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!got_initial_gps_)
        {
            RCLCPP_WARN_ONCE(this->get_logger(), "Esperando la primera pose del GPS para empezar a calcular...");
            return;
        }

        // --- FASE DE OPERACIÓN NORMAL ---
        process_odometry(msg);
        
    }

    void process_odometry(const nav_msgs::msg::Odometry::SharedPtr &msg)
    {

        double dx = msg->pose.pose.position.x - initial_lidar_.x;
        double dy = msg->pose.pose.position.y - initial_lidar_.y;
        double dz = msg->pose.pose.position.z - initial_lidar_.z;


        double dx_rot = dx * cos(initial_gps_yaw_) - dy * sin(initial_gps_yaw_);
        double dy_rot = dx * sin(initial_gps_yaw_) + dy * cos(initial_gps_yaw_);

        geometry_msgs::msg::Point est;
        est.x = initial_gps_.x + dx_rot;
        est.y = initial_gps_.y + dy_rot;
        est.z = initial_gps_.z + dz;

        est_traj_.push_back(est);
        gt_traj_.push_back(last_gps_);

        // ... (Cálculo de ATE y RPE no cambia) ...
        double ate = std::sqrt(std::pow(last_gps_.x - est.x, 2) + std::pow(last_gps_.y - est.y, 2));
        double rpe = 0.0;
        if (est_traj_.size() > 1)
        {
            const auto &p_est_curr = est_traj_.back();
            const auto &p_est_prev = est_traj_[est_traj_.size() - 2];
            double d_est_x = p_est_curr.x - p_est_prev.x;
            double d_est_y = p_est_curr.y - p_est_prev.y;

            const auto &p_gt_curr = gt_traj_.back();
            const auto &p_gt_prev = gt_traj_[gt_traj_.size() - 2];
            double d_gt_x = p_gt_curr.x - p_gt_prev.x;
            double d_gt_y = p_gt_curr.y - p_gt_prev.y;

            double ddx = d_gt_x - d_est_x;
            double ddy = d_gt_y - d_est_y;
            rpe = std::sqrt(ddx * ddx + ddy * ddy);
        }

        RCLCPP_INFO(this->get_logger(), "ATE: %.3f m | RPE: %.3f m | Est(x,y): (%.2f, %.2f) | GT(x,y): (%.2f, %.2f)",
                    ate, rpe, est.x, est.y, last_gps_.x, last_gps_.y);
    }

    // Variables miembro
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr lidar_odometry_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gps_sub_;
    std::mutex mutex_;

    // Datos iniciales
    geometry_msgs::msg::Point initial_lidar_;
    geometry_msgs::msg::Point initial_gps_;
    double initial_lidar_yaw_ = 0.0;
    double initial_gps_yaw_ = 0.0;
    bool got_initial_lidar_ = false;
    bool got_initial_gps_ = false;

    // Última pose del GPS
    geometry_msgs::msg::Point last_gps_;

    // Vectores para guardar las trayectorias
    std::vector<geometry_msgs::msg::Point> est_traj_;
    std::vector<geometry_msgs::msg::Point> gt_traj_;

    bool calibrate_angle_;
    bool is_calibrated_;
    size_t calibration_samples_;
    double theta_ = 0.0; // Ángulo de corrección final
    static const int CALIBRATION_WINDOW = 50;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OdomGpsComparator>());
    rclcpp::shutdown();
    return 0;
}