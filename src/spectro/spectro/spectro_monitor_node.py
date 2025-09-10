#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import json

class SpectroMonitorNode(Node):
    def __init__(self):
        super().__init__('spectro_monitor_node')
        
        # Subscribers
        self.result_sub = self.create_subscription(
            String, 
            '/spectro_processor_node/processing_result', 
            self.result_callback, 
            10
        )
        
        self.average_sub = self.create_subscription(
            Float32,
            '/spectro_processor_node/average_derivative',
            self.average_callback,
            10
        )
        
        self.get_logger().info("Spectro Monitor Node started")
        self.get_logger().info("Monitoring processing results...")

    def result_callback(self, msg):
        """Callback for processing results"""
        try:
            data = json.loads(msg.data)
            filename = data.get('filename', 'Unknown')
            average = data.get('average_derivative', 0.0)
            plot_filename = data.get('plot_filename', 'Unknown')
            timestamp = data.get('timestamp', 'Unknown')
            
            self.get_logger().info(
                f"PROCESSED: {filename} | "
                f"Avg d²R/dλ²: {average:.6f} | "
                f"Plot: {plot_filename} | "
                f"Time: {timestamp}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error parsing result: {e}")

    def average_callback(self, msg):
        """Callback for average derivative values"""
        # This is redundant with result_callback but can be used separately
        pass


def main(args=None):
    rclpy.init(args=args)
    
    node = SpectroMonitorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutdown signal received")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
