from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([FindPackageShare('gem_e2_description'), 'urdf', 'gem_e2.urdf.xacro'])
    ])

    robot_description = {'robot_description': robot_description_content}

    

    platform_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('platform_launch'),
                    'launch',
                    'white_e2',
                    'platform.launch.py'
                ])
            ),
            launch_arguments={'use_camera': 'true'}.items()
        )
    
    zed_camera_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('basic_launch'),
                    'launch',
                    'perception',
                    'zed_camera.launch.py'
                ])
            ),
            launch_arguments={
                'camera_model' : 'zed2'}.items(),
        )
    
    gnss_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('septentrio_gnss_driver'),
                    'launch',
                    'rover.launch.py'
                ])
            )
        )
    # Top Lidar
    ouster_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('basic_launch'),
                    'launch',
                    'perception',
                    'ouster_driver.launch.py'
                ])
            ),
            launch_arguments={
                'param_file' : 'ouster_config'}.items(),
        )
    
    lucid_cam_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('basic_launch'),
                    'launch',
                    'perception',
                    'corner_cameras.launch.py'
                ])
            )
        )
    
    rviz_display_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('basic_launch'), 'launch'),
            '/rviz_display.launch.py'])
    )

    return LaunchDescription([
        zed_camera_launch,
        ouster_launch,
        lucid_cam_launch,
        rviz_display_launch
    ])
