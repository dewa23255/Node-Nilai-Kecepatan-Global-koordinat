#!/usr/bin/env python3
"""
Node Global Velocity robot roda diferensial (Robin).

Menghitung kecepatan robot dalam KOORDINAT GLOBAL dari kecepatan angular roda,
mengikuti materi Local Coordinate, Global Coordinate, dan Position Estimation.

Input  : /left_wheel/command   (std_msgs/Float64)  phi_dot_L (rad/s)
         /right_wheel/command  (std_msgs/Float64)  phi_dot_R (rad/s)

Output : terminal (log) dan topic /global_velocity (geometry_msgs/Twist)
         linear.x = x_dot, linear.y = y_dot, angular.z = theta_dot

Rumus (local coordinate / forward kinematics):
    V_L = r * phi_dot_L
    V_R = r * phi_dot_R
    V_B = (V_L + V_R) / 2
    omega = (V_R - V_L) / s

Rumus (global coordinate):
    x_dot     = V_B * cos(theta)
    y_dot     = V_B * sin(theta)
    theta_dot = omega

Nilai theta didapat dari integrasi numerik (position estimation):
    theta_k = theta_{k-1} + omega_k * dt
    x_k     = x_{k-1} + x_dot_k * dt
    y_k     = y_{k-1} + y_dot_k * dt

dengan r = jari-jari roda, s = jarak antar roda.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class GlobalVelocity(Node):
    def __init__(self):
        super().__init__('global_velocity')

        # Parameter robot (samakan dengan model Robin)
        self.declare_parameter('wheel_radius', 0.03)       # r (meter)
        self.declare_parameter('wheel_separation', 0.17)   # s (meter)
        self.declare_parameter('update_rate', 20.0)        # Hz, frekuensi hitung
        self.declare_parameter('print_rate', 2.0)          # Hz, frekuensi cetak

        self.r = self.get_parameter('wheel_radius').value
        self.s = self.get_parameter('wheel_separation').value
        update_rate = self.get_parameter('update_rate').value
        print_rate = self.get_parameter('print_rate').value
        self.print_every = max(1, int(update_rate / print_rate))
        self.tick = 0

        # Kecepatan angular roda terakhir yang diterima (rad/s)
        self.phi_dot_l = 0.0
        self.phi_dot_r = 0.0

        # Estimasi pose global (awal di 0, 0, 0)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = self.get_clock().now()

        self.create_subscription(
            Float64, '/left_wheel/command', self.left_callback, 10)
        self.create_subscription(
            Float64, '/right_wheel/command', self.right_callback, 10)

        self.pub = self.create_publisher(Twist, '/global_velocity', 10)
        self.timer = self.create_timer(1.0 / update_rate, self.update)

        self.get_logger().info(
            f'Global velocity aktif. r={self.r} m, s={self.s} m. '
            f'Pose awal x=0, y=0, theta=0.')

    def left_callback(self, msg):
        self.phi_dot_l = msg.data

    def right_callback(self, msg):
        self.phi_dot_r = msg.data

    def update(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0:
            return

        # Local coordinate: kecepatan roda -> kecepatan badan robot
        v_l = self.r * self.phi_dot_l
        v_r = self.r * self.phi_dot_r
        v_b = (v_l + v_r) / 2.0
        omega = (v_r - v_l) / self.s

        # Global coordinate: kecepatan badan -> kecepatan global
        x_dot = v_b * math.cos(self.theta)
        y_dot = v_b * math.sin(self.theta)
        theta_dot = omega

        # Position estimation: integrasi numerik (metode Euler)
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.theta += theta_dot * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Publish kecepatan global
        out = Twist()
        out.linear.x = x_dot
        out.linear.y = y_dot
        out.angular.z = theta_dot
        self.pub.publish(out)

        # Cetak ke terminal dengan laju yang lebih pelan
        self.tick += 1
        if self.tick % self.print_every == 0:
            if abs(omega) < 1e-6:
                radius = 'lurus'
            else:
                radius = f'{v_b / omega:.2f} m'
            self.get_logger().info(
                f'x_dot={x_dot:+.3f} m/s | y_dot={y_dot:+.3f} m/s | '
                f'theta_dot={theta_dot:+.3f} rad/s || '
                f'V_B={v_b:+.3f} m/s | pose: x={self.x:+.2f} m, '
                f'y={self.y:+.2f} m, theta={math.degrees(self.theta):+.1f} deg | '
                f'radius belok: {radius}')


def main(args=None):
    rclpy.init(args=args)
    node = GlobalVelocity()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
