from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='spectro',
            executable='spectro_processor_node',
            name='spectro_processor_node',
            output='screen',
            parameters=[
                {'reflectance_folder': '/home/tank/spectro_data/reflectance'},
                {'reference_file': '/home/tank/ros2_spectra_ws/standard.txt'},
                {'process_folder_name': 'process'},
                {'wavelength_min': 400.0},
                {'wavelength_max': 700.0},
                {'smoothing_window': 100},
                {'smoothing_order': 2},
            ]
        ),
    ])
