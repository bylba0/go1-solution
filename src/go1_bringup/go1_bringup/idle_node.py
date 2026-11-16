"""Узел-заглушка: задание ещё не реализовано. Ничего не публикует, живёт до остановки контейнера."""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class IdleNode(Node):
    def __init__(self):
        super().__init__('go1_idle')
        task = self.declare_parameter('task', 'unknown').value
        self.get_logger().warn(f'{task}: задание не реализовано (нет в supported_tasks) — узел-заглушка, ничего не публикую')


def main(args=None):
    rclpy.init(args=args)
    node = IdleNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
