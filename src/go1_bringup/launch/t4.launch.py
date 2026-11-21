"""T4: одометрия (/odom + TF odom → trunk). Запускается из /entrypoint.sh T4."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('go1_odometry'), 'config', 'odometry.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('params_file', default_value=params),
        Node(package='go1_odometry', executable='odometry', name='go1_odometry', output='screen',
             parameters=[LaunchConfiguration('params_file'), {'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
