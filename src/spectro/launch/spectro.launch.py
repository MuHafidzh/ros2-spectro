from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spectro',
            executable='spectro_node',
            name='spectro_node',
            output='screen',
            parameters=[
                # Parameter yang dapat disesuaikan
                {'integration_time_ms': 500},  # Default 500ms
                {'scans_to_average': 5},
                {'boxcar_width': 1},
                {'save_path': '~/spectro_data'},
                {'joystick_intensity_button': 0},   # X button untuk save intensity
                {'joystick_reflectance_button': 2}, # Triangle button untuk save reflectance
                {'joystick_ps_button': 10},         # PS button untuk connect/disconnect
                {'joystick_start_button': 9},       # Start button untuk toggle tab
            ]
        ),
    ])
