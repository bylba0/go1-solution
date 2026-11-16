"""T2: Safety Block (/lowcmd_bare → /lowcmd). Запускается из /entrypoint.sh T2."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='go1_safety', executable='safety_block', name='go1_safety_block', output='screen',
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]),
    ])
