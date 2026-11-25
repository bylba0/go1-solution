"""T3: ходьба на обученной политике. Запускается из /entrypoint.sh T3.

/cmd_vel + /lowstate → go1_walk (политика, цели 200 Гц) → /lowcmd_bare → go1_safety (Safety Block T2) → /lowcmd.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='go1_safety', executable='safety_block', name='go1_safety_block', output='screen',
             respawn=True, respawn_delay=0.1,
             parameters=[{'use_sim_time': use_sim_time}]),
        # kp 45 вместо обученных 50 (в обучении kp рандомизирован ±10 %): меньше шагов с моментом в пределе
        Node(package='go1_walk', executable='walk_node', name='go1_walk', output='screen',
             respawn=True, respawn_delay=0.1,
             parameters=[{'use_sim_time': use_sim_time, 'heading_hold': True, 'kp': 45.0}]),
    ])
