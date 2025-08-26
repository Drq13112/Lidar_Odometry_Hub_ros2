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
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>

struct PoseData {
    Eigen::Matrix4d transform;
    double timestamp;
};

class OdomGpsComparator : public rclcpp::Node
{
public:
    OdomGpsComparator()
        : Node("odom_gps_comparator"), got_initial_gps_(false), got_initial_lidar_(false), 
          is_calibrated_(false), calibration_samples_(0)
    {
        // Parámetros para los topics
        this->declare_parameter<bool>("calibrate_angle", true);
        calibrate_angle_ = this->get_parameter("calibrate_angle").as_bool();
        this->declare_parameter<std::string>("lidar_topic", "/lidar_odometry/pose");
        std::string lidar_topic = this->get_parameter("lidar_topic").as_string();
        this->declare_parameter<std::string>("gps_topic", "/zoe/localization/global");
        std::string gps_topic = this->get_parameter("gps_topic").as_string();
        this->declare_parameter<std::string>("dlo_pose_topic", "dlo/odom_node/pose");
        std::string dlo_pose_topic = this->get_parameter("dlo_pose_topic").as_string();

        // Subscripciones
        lidar_odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            lidar_topic, 1,
            std::bind(&OdomGpsComparator::lidar_odometry_callback, this, std::placeholders::_1));

        lidar_odometry_sub_dlo_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            dlo_pose_topic, 1,
            std::bind(&OdomGpsComparator::pose_stamped_callback, this, std::placeholders::_1));

        gps_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            gps_topic, 1,
            std::bind(&OdomGpsComparator::gps_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Nodo comparador iniciado. Esperando datos...");
    }

    ~OdomGpsComparator()
    {
        if (estimated_poses_.empty() || ground_truth_poses_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "No hay datos de trayectoria para guardar o calcular métricas.");
            return;
        }

        // Calcular y mostrar métricas finales
        //calculate_final_metrics();
        save_trajectories();
        plot_trajectories();
    }

private:
    void gps_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        last_gps_pose_ = *msg;

        if (!got_initial_gps_)
        {
            initial_gps_ = msg->pose.pose.position;
            initial_gps_yaw_ = msg-> pose.pose.orientation.z * M_PI / 180.0; // El yaw del GPS viene dentro del eje Z de la orientación en grados -> pasamos a radianes
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

        if (!got_initial_lidar_)
        {
            initial_lidar_ = msg->pose.pose.position;
            initial_lidar_yaw_ = extract_yaw_from_quaternion(msg->pose.pose.orientation);
            got_initial_lidar_ = true;
            RCLCPP_INFO(this->get_logger(), "Initial LIDAR pose set: (%.3f, %.3f), Yaw: %.3f deg",
                        initial_lidar_.x, initial_lidar_.y, initial_lidar_yaw_ * 180.0 / M_PI);
        }

        // --- FASE DE OPERACIÓN NORMAL ---
        process_estimated_pose(msg->pose.pose, msg->header.stamp);

    }

    void pose_stamped_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {

        if (!got_initial_gps_)
        {
            RCLCPP_WARN_ONCE(this->get_logger(), "Esperando la primera pose del GPS para empezar a calcular...");
            return;
        }

        if (!got_initial_lidar_)
        {
            initial_lidar_ = msg->pose.position;
            initial_lidar_yaw_ = extract_yaw_from_quaternion(msg->pose.orientation);
            got_initial_lidar_ = true;
            RCLCPP_INFO(this->get_logger(), "Initial LIDAR pose set: (%.3f, %.3f), Yaw: %.3f deg",
                        initial_lidar_.x, initial_lidar_.y, initial_lidar_yaw_ * 180.0 / M_PI);
        }

        process_estimated_pose(msg->pose, msg->header.stamp);
    }

    void process_estimated_pose(const geometry_msgs::msg::Pose& pose, const rclcpp::Time& stamp)
    {
            // Calcular desplazamiento desde posición inicial del estimador
        double dx = pose.position.x - initial_lidar_.x;
        double dy = pose.position.y - initial_lidar_.y;
        double dz = pose.position.z - initial_lidar_.z;

        // Aplicar rotación de calibración para alinear con el GPS inicial
        double dx_rot = dx * cos(initial_gps_yaw_) - dy * sin(initial_gps_yaw_);
        double dy_rot = dx * sin(initial_gps_yaw_) + dy * cos(initial_gps_yaw_);

        // Crear pose estimada alineada
        geometry_msgs::msg::Pose aligned_estimated_pose;
        aligned_estimated_pose.position.x = initial_gps_.x + dx_rot;
        aligned_estimated_pose.position.y = initial_gps_.y + dy_rot;
        aligned_estimated_pose.position.z = initial_gps_.z + dz;


        tf2::Quaternion estimator_q;
        tf2::fromMsg(pose.orientation, estimator_q);
        tf2::Quaternion gps_initial_rotation_q;
        gps_initial_rotation_q.setRPY(0, 0, initial_gps_yaw_);
        tf2::Quaternion final_q = gps_initial_rotation_q * estimator_q;
        final_q.normalize(); 
        aligned_estimated_pose.orientation = tf2::toMsg(final_q);

        // Guardar poses para cálculo de métricas
        PoseData est_pose_data;
        est_pose_data.transform = create_transform_matrix(aligned_estimated_pose);
        est_pose_data.timestamp = rclcpp::Time(stamp).seconds();

        PoseData gt_pose_data;
        gt_pose_data.transform = create_transform_matrix_gps(last_gps_pose_.pose.pose);
        gt_pose_data.timestamp = rclcpp::Time(last_gps_pose_.header.stamp).seconds();

        estimated_poses_.push_back(est_pose_data);
        ground_truth_poses_.push_back(gt_pose_data);

        // Calcular métricas actuales
        double ape = calculate_ape(est_pose_data.transform, gt_pose_data.transform);
        double rpe = 0.0;
        
        // if (estimated_poses_.size() > 1)
        // {
        //     rpe = calculate_rpe(
        //         estimated_poses_[estimated_poses_.size()-2].transform,
        //         estimated_poses_.back().transform,
        //         ground_truth_poses_[ground_truth_poses_.size()-2].transform,
        //         ground_truth_poses_.back().transform
        //     );
        // }

        //double ate = calculate_ate();

        // RCLCPP_INFO(this->get_logger(), "APE: %.3f m | RPE: %.3f m | ATE: %.3f m | Est(x,y): (%.2f, %.2f) | GT(x,y): (%.2f, %.2f)",
        //             ape, rpe, ate, estimated_pose.pose.pose.position.x, estimated_pose.pose.pose.position.y, 
        //             last_gps_pose_.pose.pose.position.x, last_gps_pose_.pose.pose.position.y);

        // Guardar para compatibilidad con el plotting
        est_traj_.push_back(aligned_estimated_pose.position);
        gt_traj_.push_back(last_gps_pose_.pose.pose.position);

        // --- GUARDAR EN FORMATO TUM ---
        // Estimada
        {
            std::ofstream est_tum("/home/david/ros2_ws/traj_est.txt", std::ios::app);
            double t = stamp.seconds();
            const auto& p = aligned_estimated_pose.position;
            const auto& q = aligned_estimated_pose.orientation;
            est_tum << std::fixed << std::setprecision(9)
                    << t << " "
                    << p.x << " " << p.y << " " << p.z << " "
                    << q.x << " " << q.y << " " << q.z << " " << q.w << std::endl;
        }
        // Ground truth
        {
            std::ofstream gt_tum("/home/david/ros2_ws/traj_gt.txt", std::ios::app);
            double t = last_gps_pose_.header.stamp.sec + last_gps_pose_.header.stamp.nanosec * 1e-9;
            const auto& p = last_gps_pose_.pose.pose.position;
            const auto& q = last_gps_pose_.pose.pose.orientation;
            gt_tum << std::fixed << std::setprecision(9)
                  << t << " "
                  << p.x << " " << p.y << " " << p.z << " "
                  << q.x << " " << q.y << " " << q.z << " " << q.w << std::endl;
        }
    }

    double extract_yaw_from_quaternion(const geometry_msgs::msg::Quaternion& quat)
    {
        tf2::Quaternion q(quat.x, quat.y, quat.z, quat.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        return yaw;
    }

    Eigen::Matrix4d create_transform_matrix(const geometry_msgs::msg::Pose& pose)
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        
        // Posición
        T(0, 3) = pose.position.x;
        T(1, 3) = pose.position.y;
        T(2, 3) = pose.position.z;
        
        // Rotación
        tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
        tf2::Matrix3x3 rot_matrix(q);
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T(i, j) = rot_matrix[i][j];
            }
        }
        
        return T;
    }
    Eigen::Matrix4d create_transform_matrix_gps(const geometry_msgs::msg::Pose& pose)
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        
        // Posición
        T(0, 3) = pose.position.x;
        T(1, 3) = pose.position.y;
        T(2, 3) = pose.position.z;
        
        // Rotación usando solo la componente Z del GPS (en grados, convertir a radianes)
        double yaw_rad = pose.orientation.z * M_PI / 180.0;
        
        // matriz de rotación solo en Z (yaw)
        // T(0, 0) = cos(yaw_rad);
        // T(0, 1) = -sin(yaw_rad);
        // T(1, 0) = sin(yaw_rad);
        // T(1, 1) = cos(yaw_rad);
        // T(2, 2) = 1.0; // Sin rotación en Z


        // Matriz de rotación usando quaternios
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw_rad); // Roll=0, Pitch=0, solo Yaw
        tf2::Matrix3x3 rot_matrix(q);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                T(i, j) = rot_matrix[i][j];
            }
        }

        return T;
    }

    double calculate_ape(const Eigen::Matrix4d& T_est, const Eigen::Matrix4d& T_gt)
    {
        // --- Error de Posición ---
        Eigen::Vector3d pos_est = T_est.block<3,1>(0, 3);
        Eigen::Vector3d pos_gt = T_gt.block<3,1>(0, 3);
        
        // Calcular el vector de diferencia de posición
        Eigen::Vector3d pos_diff = pos_gt - pos_est;
        
        // Obtener el error absoluto para cada componente de posición
        double error_x = pos_diff.x();
        double error_y = pos_diff.y();
        double error_z = pos_diff.z();

        // --- Error de Orientación ---
        Eigen::Matrix3d R_est = T_est.block<3,3>(0, 0);
        Eigen::Matrix3d R_gt = T_gt.block<3,3>(0, 0);
        
        // Calcular la matriz de rotación del error
        Eigen::Matrix3d R_err = R_gt * R_est.transpose();
        
        // Calcular el ángulo de error de la rotación (en radianes)
        // a partir de la traza de la matriz de error. trace(R) = 1 + 2*cos(theta)
        double trace = R_err.trace();
        // Asegurarse de que el argumento para acos esté en el rango [-1, 1]
        double cos_theta = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
        double orientation_error_rad = std::acos(cos_theta);

        
        // --- CÁLCULO DE ERROR DE YAW (MODIFICADO) ---
        // Extraer el yaw de cada matriz de rotación
        // atan2(R(1,0), R(0,0)) da el yaw
        double yaw_est = atan2(R_est(1, 0), R_est(0, 0));
        double yaw_gt = atan2(R_gt(1, 0), R_gt(0, 0));

        // Calcular la diferencia, asegurándose de manejar el salto de -PI a PI
        double yaw_error_rad = yaw_gt - yaw_est;
        if (yaw_error_rad > M_PI) {
            yaw_error_rad -= 2 * M_PI;
        } else if (yaw_error_rad < -M_PI) {
            yaw_error_rad += 2 * M_PI;
        }

        // Loguear los errores desglosados por componente
        RCLCPP_INFO(this->get_logger(), "APE components -> Pos(X,Y,Z): (%.3f, %.3f, %.3f) m | Ori: %.3f deg",
                    error_x, error_y, error_z, yaw_error_rad * 180.0 / M_PI);

        // Devolver el error de posición total (norma euclidiana), que es el valor principal para el log global
        return pos_diff.norm();
    }

    double calculate_rpe(const Eigen::Matrix4d& T_est_i, const Eigen::Matrix4d& T_est_j,
                        const Eigen::Matrix4d& T_gt_i, const Eigen::Matrix4d& T_gt_j)
    {
        Eigen::Matrix4d rel_est = T_est_i.inverse() * T_est_j;
        Eigen::Matrix4d rel_gt = T_gt_i.inverse() * T_gt_j;
        Eigen::Matrix4d diff = rel_gt - rel_est;
        return diff.norm();
    }

    double calculate_ate()
    {
        if (estimated_poses_.empty() || ground_truth_poses_.empty()) {
            return 0.0;
        }

        Eigen::Vector3d est_centroid = Eigen::Vector3d::Zero();
        Eigen::Vector3d gt_centroid = Eigen::Vector3d::Zero();
        
        size_t n = std::min(estimated_poses_.size(), ground_truth_poses_.size());
        
        for (size_t i = 0; i < n; ++i) {
            est_centroid += estimated_poses_[i].transform.block<3,1>(0,3);
            gt_centroid += ground_truth_poses_[i].transform.block<3,1>(0,3);
        }
        
        est_centroid /= n;
        gt_centroid /= n;

        // Calcular ATE después de alineación
        double sum_error = 0.0;
        for (size_t i = 0; i < n; ++i) {
            Eigen::Vector3d est_pos = estimated_poses_[i].transform.block<3,1>(0,3) - est_centroid;
            Eigen::Vector3d gt_pos = ground_truth_poses_[i].transform.block<3,1>(0,3) - gt_centroid;
            sum_error += (gt_pos - est_pos).norm();
        }
        
        return sum_error / n;
    }

    void calculate_final_metrics()
    {
        if (estimated_poses_.empty() || ground_truth_poses_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No hay poses para calcular métricas finales.");
            return;
        }

        size_t n = std::min(estimated_poses_.size(), ground_truth_poses_.size());
        
        // Calcular estadísticas de APE
        std::vector<double> ape_values;
        for (size_t i = 0; i < n; ++i) {
            double ape = calculate_ape(estimated_poses_[i].transform, ground_truth_poses_[i].transform);
            ape_values.push_back(ape);
        }

        // Calcular estadísticas de RPE
        std::vector<double> rpe_values;
        for (size_t i = 1; i < n; ++i) {
            double rpe = calculate_rpe(
                estimated_poses_[i-1].transform, estimated_poses_[i].transform,
                ground_truth_poses_[i-1].transform, ground_truth_poses_[i].transform
            );
            rpe_values.push_back(rpe);
        }

        double final_ate = calculate_ate();

        // Calcular estadísticas
        double ape_mean = std::accumulate(ape_values.begin(), ape_values.end(), 0.0) / ape_values.size();
        double rpe_mean = rpe_values.empty() ? 0.0 : std::accumulate(rpe_values.begin(), rpe_values.end(), 0.0) / rpe_values.size();

        auto ape_minmax = std::minmax_element(ape_values.begin(), ape_values.end());
        auto rpe_minmax = rpe_values.empty() ? std::make_pair(rpe_values.end(), rpe_values.end()) : 
                         std::minmax_element(rpe_values.begin(), rpe_values.end());

        RCLCPP_INFO(this->get_logger(), "========== MÉTRICAS FINALES ==========");
        RCLCPP_INFO(this->get_logger(), "APE - Media: %.4f m, Min: %.4f m, Max: %.4f m", 
                    ape_mean, *ape_minmax.first, *ape_minmax.second);
        if (!rpe_values.empty()) {
            RCLCPP_INFO(this->get_logger(), "RPE - Media: %.4f m, Min: %.4f m, Max: %.4f m", 
                        rpe_mean, *rpe_minmax.first, *rpe_minmax.second);
        }
        RCLCPP_INFO(this->get_logger(), "ATE: %.4f m", final_ate);
        RCLCPP_INFO(this->get_logger(), "Total de poses procesadas: %zu", n);
        RCLCPP_INFO(this->get_logger(), "======================================");
    }

    void save_trajectories()
    {
        RCLCPP_INFO(this->get_logger(), "Guardando trayectorias en /home/david/ros2_ws/ ...");
        std::ofstream est_file("/home/david/ros2_ws/odom_est_traj.csv");
        std::ofstream gt_file("/home/david/ros2_ws/odom_gt_traj.csv");

        // Escribir cabeceras en los ficheros CSV
        est_file << "x,y,z,qx,qy,qz,qw\n";
        gt_file << "x,y,z,qx,qy,qz,qw\n";

        // No es necesario restar initial_gps_ aquí si el ploteo lo maneja
        // o si las poses ya están en un marco común.
        // Para la evaluación, es mejor guardar las poses globales alineadas.

        size_t n = std::min(estimated_poses_.size(), ground_truth_poses_.size());

        for (size_t i = 0; i < n; ++i)
        {
            // --- Pose Estimada ---
            const auto& est_transform = estimated_poses_[i].transform;
            const auto& est_position = est_transform.block<3, 1>(0, 3);
            
            // CORRECCIÓN: Convertir la matriz de rotación a cuaternión
            tf2::Matrix3x3 est_rot_matrix(
                est_transform(0, 0), est_transform(0, 1), est_transform(0, 2),
                est_transform(1, 0), est_transform(1, 1), est_transform(1, 2),
                est_transform(2, 0), est_transform(2, 1), est_transform(2, 2)
            );
            tf2::Quaternion est_orientation;
            est_rot_matrix.getRotation(est_orientation);
            est_orientation.normalize();

            est_file << std::fixed << std::setprecision(6)
                    << est_position.x() << "," << est_position.y() << "," << est_position.z() << ","
                    << est_orientation.x() << "," << est_orientation.y() << "," << est_orientation.z() << "," << est_orientation.w() << std::endl;

            // --- Pose Ground Truth ---
            const auto& gt_transform = ground_truth_poses_[i].transform;
            const auto& gt_position = gt_transform.block<3, 1>(0, 3);

            // CORRECCIÓN: Convertir la matriz de rotación a cuaternión
            tf2::Matrix3x3 gt_rot_matrix(
                gt_transform(0, 0), gt_transform(0, 1), gt_transform(0, 2),
                gt_transform(1, 0), gt_transform(1, 1), gt_transform(1, 2),
                gt_transform(2, 0), gt_transform(2, 1), gt_transform(2, 2)
            );
            tf2::Quaternion gt_orientation;
            gt_rot_matrix.getRotation(gt_orientation);
            gt_orientation.normalize();

            gt_file << std::fixed << std::setprecision(6)
                    << gt_position.x() << "," << gt_position.y() << "," << gt_position.z() << ","
                    << gt_orientation.x() << "," << gt_orientation.y() << "," << gt_orientation.z() << "," << gt_orientation.w() << std::endl;
        }

        est_file.close();
        gt_file.close();
        RCLCPP_INFO(this->get_logger(), "Trayectorias guardadas.");
    }

    void plot_trajectories()
    {
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

    // Variables miembro
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr lidar_odometry_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gps_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr lidar_odometry_sub_dlo_;
    std::mutex mutex_;

    // Datos iniciales
    geometry_msgs::msg::Point initial_lidar_;
    geometry_msgs::msg::Point initial_gps_;
    double initial_lidar_yaw_ = 0.0;
    double initial_gps_yaw_ = 0.0;
    bool got_initial_lidar_ = false;
    bool got_initial_gps_ = false;

    // Última pose del GPS
    nav_msgs::msg::Odometry last_gps_pose_;

    // Vectores para guardar las trayectorias (compatibilidad)
    std::vector<geometry_msgs::msg::Point> est_traj_;
    std::vector<geometry_msgs::msg::Point> gt_traj_;

    // Nuevas estructuras para métricas avanzadas
    std::vector<PoseData> estimated_poses_;
    std::vector<PoseData> ground_truth_poses_;

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