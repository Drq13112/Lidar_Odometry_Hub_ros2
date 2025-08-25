import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # 1. Publicador de la transformada estática del sensor
    static_tf_base_to_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_base_to_lidar',
        arguments=['0.20', '0', '0.30', '0', '0', '0', 'base_link', 'rubyplus'],
        output='screen'
    )

    # Odom Comparator Node for MOla
    odom_comparator_node = Node(
            package='lidar_odom_subscriber',
            executable='odom_gps_comparator',
            name='odom_gps_comparator',
            output='screen',
            parameters=[
                {'lidar_topic': '/lidar_odometry/pose'},
                {'gps_topic': '/zoe/localization/global'},
                {'calibrate_angle': False},
                #{'use_sim_time':True}, 
            ]
        )
    
    # Odom Comparator Node for Kinematic
    odom_comparator_node_kinematic = Node(
            package='lidar_odom_subscriber',
            executable='odom_gps_comparator',
            name='odom_gps_comparator',
            output='screen',
            parameters=[
                {'lidar_topic': '/kinematic_icp/lidar_odometry'},
                {'gps_topic': '/zoe/localization/global'},
                {'calibrate_angle': False},
                #{'use_sim_time':True}, 
            ]
        )

    # Odometry Comparator Node for Kiss-ICP
    odom_comparator_node_kiss = Node(
            package='lidar_odom_subscriber',
            executable='odom_gps_comparator',
            name='odom_gps_comparator',
            output='screen',
            parameters=[
                {'lidar_topic': '/kiss/odometry'},
                {'gps_topic': '/zoe/localization/global'},
                {'calibrate_angle': False} 
            ]
        )

    return LaunchDescription([

        # mola_launch,
        odom_comparator_node_kiss,
        static_tf_base_to_lidar
    ])
