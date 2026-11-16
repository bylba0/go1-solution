"""Заглушка для заданий, не заявленных в supported_tasks: один пустой узел."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('task', default_value='unknown'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='go1_bringup', executable='idle_node', name='go1_idle', output='screen',
             parameters=[{'task': LaunchConfiguration('task'),
                          'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
