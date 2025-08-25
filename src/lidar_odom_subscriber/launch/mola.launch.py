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
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'rubyplus'],
        output='screen'
    )

    # 2. Lanzador de MOLA (con los parámetros correctos)
    mola_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('mola_lidar_odometry'),'ros2-launchs', 'ros2-lidar-odometry.launch.py')
        ),
        launch_arguments={
            'lidar_topic_name': 'rubyplus_points',
            'ignore_lidar_pose_from_tf': 'False', 
            'publish_localization_following_rep105': 'False',
            'mola_lo_reference_frame': 'map',
            'use_rviz': 'False',
            'use_mola_gui': 'False',
            # 'use_state_estimator': 'True', -> No usar porque no publica en el topic /lidar_odometry/pose
            'enforce_planar_motion': 'True',
            # 'use_sim_time': 'True',
            # 'mola_lo_pipeline': '../pipelines/extras/lidar3d-kissicp-like.yaml'
            
        }.items()
    )

    # 3. Tu nodo comparador (sin el static_transform_publisher dentro)
    # Es mejor que la transformada odom->base_link la publique tu nodo de localización principal (GPS/EKF)
    odom_comparator_node = Node(
            package='lidar_odom_subscriber',
            executable='odom_gps_comparator',
            name='odom_gps_comparator',
            output='screen',
            parameters=[
                {'lidar_topic': '/lidar_odometry/pose'},
                {'gps_topic': '/zoe/localization/global'}
            ]
        )
    
    #Kiss-ICP launch
    kiss_icp_node = Node(
            package='kiss_icp',
            executable='kiss_icp_node',
            name='kiss_icp_node',
            output='screen',
            parameters=[
                {'input_topic': '/rubyplus_points'},
                {'output_topic': '/kiss/odometry'}
            ]
        )

    

    return LaunchDescription([

        mola_launch,
        # odom_comparator_node,
        # static_tf_base_to_lidar
    ])
