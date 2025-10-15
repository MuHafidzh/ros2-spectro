from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Main spectrometer node
        Node(
            package='spectro',
            executable='spectro_node',
            name='spectro_node',
            output='screen',
            parameters=[
                {'integration_time_ms': 500},
                {'scans_to_average': 10},
                {'boxcar_width': 5},
                {'save_path': '~/spectro_data'},
                {'joystick_intensity_button': 0},  # X button
                {'joystick_reflectance_button': 2}, # Triangle button
                {'joystick_ps_button': 10},
                {'joystick_start_button': 9},
                {'joystick_square_button': 3},  # Square button for service toggle
                {'joystick_circle_button': 1},  # Circle button for threshold
            ]
        ),
        
        # Data processor node
        Node(
            package='spectro',
            executable='spectro_processor_node',
            name='spectro_processor_node',
            output='screen',
            parameters=[
                {'reflectance_folder': '/home/tank/spectro_data/reflectance'},
                {'reference_file': '/home/tank/ros2_spectra_ws/standard.txt'},
                {'process_folder_name': 'process'},
                {'use_absorbance_as_primary': True},  # Default to absorbance
                {'wavelength_min': 400.0},
                {'wavelength_max': 700.0},
                {'smoothing_window': 100},
                {'smoothing_order': 2},
            ]
        ),
    ])
