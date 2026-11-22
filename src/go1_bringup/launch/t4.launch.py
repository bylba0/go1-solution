"""T4: одометрия (/odom + TF odom → trunk). Запускается из /entrypoint.sh T4.

estimator_mode:=legs|lidar — переопределяет estimator.mode из params_file (пусто — как в файле параметров).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _odometry(context):
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() in ('true', '1')
    params = [LaunchConfiguration('params_file').perform(context), {'use_sim_time': use_sim_time}]
    mode = LaunchConfiguration('estimator_mode').perform(context).strip()
    if mode:
        params.append({'estimator.mode': mode})
    return [Node(package='go1_odometry', executable='odometry', name='go1_odometry', output='screen',
                 parameters=params)]


def generate_launch_description():
    params = os.path.join(get_package_share_directory('go1_odometry'), 'config', 'odometry.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('params_file', default_value=params),
        DeclareLaunchArgument('estimator_mode', default_value='', description='legs | lidar (пусто — из params_file)'),
        OpaqueFunction(function=_odometry),
    ])
