import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node

# This is the function launch  system will look for
def generate_launch_description():

    urdf_file = 'gem_e4.urdf'
    package_description = "gem_e4_description"

    print("Fetching URDF ==>")
    robot_desc_path = os.path.join(get_package_share_directory(package_description), "urdf", urdf_file)

    # Robot State Publisher
    # xacro ~/src/robot_description/urdf/simple.urdf
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher_node',
        # emulate_tty=True,
        parameters=[{'use_sim_time': False, 'robot_description': Command(['xacro ', robot_desc_path])}],
        output="screen"
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher_node',
        # emulate_tty=True,
        output="screen"
    )

    # RVIZ Configuration
    rviz_config_dir = os.path.join(get_package_share_directory("basic_launch"), 'rviz', 'gem_e4.rviz')

    rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            name='rviz_node',
            parameters=[{'use_sim_time': False}],
            arguments=['-d', rviz_config_dir])

    # create and return launch description object
    return LaunchDescription(
        [            
            robot_state_publisher_node,
            joint_state_publisher_node,
            rviz_node
        ]
        
    )