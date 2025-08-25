import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Obtener la ruta al directorio 'share' de este paquete
    trajlo_ros_share_dir = get_package_share_directory('trajlo_ros')

    # --- Declaración de Argumentos ---
    # Argumento para la ruta del archivo de configuración
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value=os.path.join(trajlo_ros_share_dir, 'data', 'config_livox.yaml'),
        description='Ruta al archivo de configuración de Traj-LO.'
    )

    # Argumento para el tópico de entrada del LiDAR
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic',
        default_value='/rubyplus_points',
        description='Tópico de entrada para la nube de puntos del LiDAR.'
    )

    # Argumento para el tópico de salida de la odometría
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value= '/lidar_odometry/pose',
        description='Tópico de salida para la odometría.'
    )

    # --- Definición del Nodo ---
    trajlo_ros_node = Node(
        package='trajlo_ros',
        executable='trajlo_ros_node',
        name='trajlo_ros_node',
        output='screen',
        parameters=[{
            'config_path': LaunchConfiguration('config_path'),
            # 'use_sim_time': 'True',
        }],
        remappings=[
            # Remapea el tópico de entrada codificado en el nodo ('/points_raw') 
            # al valor proporcionado en el argumento 'lidar_topic'.
            ('/points_raw', LaunchConfiguration('lidar_topic')),
            
            # Remapea el tópico de salida codificado en el nodo ('/odom')
            # al valor proporcionado en el argumento 'odom_topic'.
            ('/odom', LaunchConfiguration('odom_topic'))
        ]
    )

    # --- Creación de la Descripción del Lanzamiento ---
    ld = LaunchDescription()

    # Añadir los argumentos y el nodo a la descripción
    ld.add_action(config_path_arg)
    ld.add_action(lidar_topic_arg)
    ld.add_action(odom_topic_arg)
    ld.add_action(trajlo_ros_node)

    return ld
