from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Path to the conda python interpreter
    conda_python = "/home/gem/anaconda3/envs/adapt/bin/python" 

    # Path to the config file
    config_file = os.path.join(
        get_package_share_directory('mppi_controller'),
        'config',
        'mppi_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='mppi_controller',
            executable='adapt_mppi_node',
            name='adapt_mppi_node',
            output='screen',
            parameters=[config_file],
        )
    ])