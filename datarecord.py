#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16
from rosgraph_msgs.msg import Clock
from rcl_interfaces.msg import Log  # Added to read terminal logs!
import csv
import os
import math
import json

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

class CustomCSVRecorder(Node):
    def __init__(self, output_dir):
        super().__init__('custom_csv_recorder')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # --- RAW DATA RECORDING VARIABLES ---
        self.latest_odom = None
        self.latest_scan = None
        self.latest_cmd_vel = None
        self.current_episode = 0
        self.last_clock_time = None
        self.record_frequency_ms = 50.0 

        # --- LIVE METRIC EVALUATION VARIABLES ---
        self.last_omega = 0.0
        self.smoothness_penalty = 0.0
        
        self.collision_count = 0
        self.in_collision = False
        
        self.goals_sent = 0
        self.goals_reached = 0
        self.current_goal = None # Acts as a lock to prevent double-counting

        # Subscriptions mapped to your specific topics
        self.create_subscription(Odometry, '/task_generator_node/jackal/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/task_generator_node/jackal/lidar', self.scan_cb, 10)
        self.create_subscription(Twist, '/task_generator_node/jackal/cmd_vel', self.vel_cb, 10)
        self.create_subscription(Int16, '/task_generator_node/task_reset', self.reset_cb, 10)
        self.create_subscription(Clock, '/clock', self.clock_cb, 10)
        self.create_subscription(PoseStamped, '/task_generator_node/jackal/goal_pose', self.goal_cb, 10)
        
        # Subscribe to terminal logs to catch "Goal succeeded"
        self.create_subscription(Log, '/rosout', self.rosout_cb, 10)

        # Initialize CSV files for raw data
        self.write_csv("odom", ["time", "data"], "w")
        self.write_csv("scan", ["time", "data"], "w")
        self.write_csv("cmd_vel", ["time", "data"], "w")
        self.write_csv("episode", ["time", "episode"], "w")
        self.write_csv("start_goal", ["episode", "start", "goal"], "w")
        self.write_csv("start_goal", [0, "[0.0, 0.0, 0.0]", "[10.0, 10.0, 0.0]"], "a")
        
        with open(os.path.join(self.output_dir, "params.yaml"), "w") as f:
            f.write("model: 'jackal'\n")

        self.get_logger().info(f"All-in-One Recorder Active! Saving to: {self.output_dir}")
        self.get_logger().info("Send goals in RViz. Press Ctrl+C when finished to generate the final summary CSV!")

    def write_csv(self, filename, row, mode="a"):
        with open(os.path.join(self.output_dir, f"{filename}.csv"), mode, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def rosout_cb(self, msg):
        # Watch the terminal logs for the exact Nav2 success phrases
        if "Reached the goal!" in msg.msg or "Goal succeeded" in msg.msg:
            # If we currently have an active goal, mark it as a success!
            if self.current_goal is not None:
                self.goals_reached += 1
                self.get_logger().info(f" Confirmed by Nav2: Goal #{self.goals_reached} officially reached!")
                
                # Clear the current goal so the second log message doesn't double-count it
                self.current_goal = None 

    def goal_cb(self, msg):
        # Track when you send a new goal in RViz
        self.goals_sent += 1
        self.current_goal = True # Set the lock to True
        self.get_logger().info(f"Goal #{self.goals_sent} received! Waiting for Nav2 to finish...")

    def odom_cb(self, msg):
        pose = msg.pose.pose
        twist = msg.twist.twist
        _, _, yaw = euler_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        
        self.latest_odom = json.dumps({
            "position": [round(pose.position.x, 3), round(pose.position.y, 3), round(yaw, 3)],
            "velocity": [round(twist.linear.x, 3), round(twist.linear.y, 3), round(twist.angular.z, 3)]
        })

    def scan_cb(self, msg):
        # Metric: Collision checking
        valid_ranges = [r for r in msg.ranges if not math.isnan(r) and r > 0.0]
        if valid_ranges:
            min_dist = min(valid_ranges)
            # Jackal radius is ~0.25m. Less than 0.28m is a collision!
            if min_dist < 0.28:
                if not self.in_collision:
                    self.collision_count += 1
                    self.in_collision = True
                    self.get_logger().warn(f'COLLISION! Total so far: {self.collision_count}')
            else:
                self.in_collision = False

        ranges = [msg.range_max if math.isnan(val) else round(val, 3) for val in msg.ranges]
        self.latest_scan = json.dumps(ranges)

    def vel_cb(self, msg):
        # Metric: Path Smoothness
        delta_omega = msg.angular.z - self.last_omega
        self.smoothness_penalty += (delta_omega ** 2)
        self.last_omega = msg.angular.z

        self.latest_cmd_vel = json.dumps([round(msg.linear.x, 3), round(msg.linear.y, 3), round(msg.angular.z, 3)])

    def reset_cb(self, msg):
        self.current_episode = msg.data
        self.write_csv("start_goal", [self.current_episode, "[0.0, 0.0, 0.0]", "[10.0, 10.0, 0.0]"])

    def clock_cb(self, msg):
        current_time_ns = msg.clock.sec * 1e9 + msg.clock.nanosec
        if self.last_clock_time is None:
            self.last_clock_time = current_time_ns
            return
        time_diff_ms = (current_time_ns - self.last_clock_time) / 1e6
        if time_diff_ms < self.record_frequency_ms:
            return
        self.last_clock_time = current_time_ns
        if self.latest_odom: self.write_csv("odom", [int(current_time_ns), self.latest_odom])
        if self.latest_scan: self.write_csv("scan", [int(current_time_ns), self.latest_scan])
        if self.latest_cmd_vel: self.write_csv("cmd_vel", [int(current_time_ns), self.latest_cmd_vel])
        self.write_csv("episode", [int(current_time_ns), self.current_episode])

    def save_final_metrics(self):
        # Calculate Success Rate
        success_rate = 0.0
        if self.goals_sent > 0:
            success_rate = (self.goals_reached / self.goals_sent) * 100.0

        # Create a new, cleanly formatted CSV just for your report
        summary_file = os.path.join(self.output_dir, "final_summary_metrics.csv")
        with open(summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Goals Sent", self.goals_sent])
            writer.writerow(["Goals Reached", self.goals_reached])
            writer.writerow(["Success Rate (%)", round(success_rate, 2)])
            writer.writerow(["Total Collisions", self.collision_count])
            writer.writerow(["Path Smoothness Penalty", round(self.smoothness_penalty, 2)])

        # Print beautifully to terminal
        print("\n" + "="*40)
        print(" FINAL METRICS SAVED ")
        print("="*40)
        print(f"Goals Sent        : {self.goals_sent}")
        print(f"Goals Reached     : {self.goals_reached}")
        print(f"Success Rate      : {success_rate:.2f}%")
        print(f"Total Collisions  : {self.collision_count}")
        print(f"Path Smoothness   : {self.smoothness_penalty:.2f} (Lower is better)")
        print(f"-> Saved to: {summary_file}")
        print("="*40 + "\n")

def main():
    import sys
    rclpy.init()
    output_dir = "test_run_data"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        
    node = CustomCSVRecorder(output_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Trigger the final save right when you press Ctrl+C
        node.save_final_metrics()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
