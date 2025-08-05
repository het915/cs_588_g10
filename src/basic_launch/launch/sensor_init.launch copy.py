
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