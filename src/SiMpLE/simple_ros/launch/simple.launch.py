from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Genera la descripción del lanzamiento para el nodo de odometría LIDAR.
    """
    lidar_odometry_node = Node(
        package='simple_ros',
        executable='lidar_odometry',
        name='simple_lidar_odometry',
        output='screen',
        parameters=[{
            # Parámetros del algoritmo SiMpLE
            'rNew': 1.5,
            'rMap': 2.5,
            'rMin': 5.0,
            'rMax': 80.0,
            'sigma': 0.60,
            'epsilon': 1e-3,
            
            # Parámetros de ROS
            'odom_frame': 'odom_lidar'
        }]
    )

    return LaunchDescription([
        lidar_odometry_node
    ])
