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
                {'scans_to_average': 5},
                {'boxcar_width': 1},
                {'save_path': '~/spectro_data'},
                {'joystick_intensity_button': 0},
                {'joystick_reflectance_button': 2},
                {'joystick_ps_button': 10},
                {'joystick_start_button': 9},
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
                {'wavelength_min': 400.0},
                {'wavelength_max': 700.0},
                {'smoothing_window': 100},
                {'smoothing_order': 2},
            ]
        ),
    ])
