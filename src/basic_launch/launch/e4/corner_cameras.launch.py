import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import GroupAction
from launch.substitutions import Command
from launch_ros.actions import PushRosNamespace 
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # ---------------------------------------------------------------------

    # launch fl camera sensor
    camera_fl_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('basic_launch'), 'launch', 'e4' ),
            '/camera_fl.launch.py'])
    )

    # launch fr camera sensor
    camera_fr_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('basic_launch'), 'launch', 'e4' ),
            '/camera_fr.launch.py'])
    )

    # launch rl camera sensor
    camera_rl_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('basic_launch'), 'launch', 'e4'),
            '/camera_rl.launch.py'])
    )

    # launch rr camera sensor
    camera_rr_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('basic_launch'), 'launch', 'e4'),
            '/camera_rr.launch.py'])
    )

    # launch image combiner
    combiner_node = Node(
        package='corner_camera_convert',
        executable='corner_camera_combiner',
        name='corner_camera_combiner',
        output='screen'
    )

    # ---------------------------------------------------------------------

    # create and return launch description object
    return LaunchDescription(
        [            
            camera_fl_launch,
            camera_fr_launch,
            camera_rl_launch,
            camera_rr_launch,
            combiner_node
        ]
        
    )