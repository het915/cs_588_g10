
import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import GroupAction
from launch.substitutions import Command
from launch_ros.actions import PushRosNamespace 
from launch.actions import IncludeLaunchDescription 
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # tf of front radar sensor
    tf2_umrr_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='front_radar_link_to_umrr',
        arguments=['0', '0', '0', '0', '0', '0', 'front_radar_link', 'umrr']
    )

    # tf of front camera sensor
    tf2_oak_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='front_camera_link_to_oak_d_base_frame',
        arguments=['0', '0', '0', '0', '0', '0', 'front_camera_link', 'oak-d-base-frame']
    )

    # tf of front lidar sensor
    tf2_livox_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='front_lidar_link_to_livox_frame',
        arguments=['0', '0', '0.03', '0', '0.32', '0', 'front_lidar_link', 'livox_frame']
    )

    # tf of top lidar sensor
    tf2_ouster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='top_lidar_link_to_os_sensor',
        arguments=['0', '0', '0.04', '0', '0', '0', 'top_lidar_link', 'os_sensor']
    )

    # ---------------------------------------------------------------------

    # launch front camera sensor
    front_camera_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('depthai_ros_driver'), 'launch'),
            '/rgbd_pcl.launch.py'])
    )

    # launch front radar sensor
    front_radar_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('umrr_ros2_driver'), 'launch'),
            '/radar.launch.py'])
    )

    # launch front lidar sensor
    front_lidar_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('livox_ros_driver2'), 'launch'),
            '/HAP_launch.py'])
    )

    # launch top lidar sensor
    top_lidar_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ouster_ros'), 'launch'),
            '/driver.launch.py'])
    )

    # launch corner camera sensors
    # corner_camera_launch = IncludeLaunchDescription(        
    #     PythonLaunchDescriptionSource([os.path.join(
    #         get_package_share_directory('basic_launch'), 'launch'),
    #         '/corner_camera_launch.launch.py'])
    # )


    # launch rviz
    rviz_display_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('basic_launch'), 'launch'),
            '/rviz_display.launch.py'])
    )

    # ---------------------------------------------------------------------

    # create and return launch description object
    return LaunchDescription(
        [            
            tf2_umrr_node,
            tf2_oak_node,
            tf2_livox_node,
            tf2_ouster_node,
            front_radar_launch,
            front_lidar_launch,
            top_lidar_launch,
            front_camera_launch,
            rviz_display_launch,
            # corner_camera_launch
        ]
        
    )