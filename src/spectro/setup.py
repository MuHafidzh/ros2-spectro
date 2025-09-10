import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'spectro'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nakanomiku',
    maintainer_email='nakanomiku@todo.todo',
    description='ROS2 node for Ocean Insight Spectrometer',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spectro_node = spectro.spectro_node:main',
            'spectro_processor_node = spectro.spectro_processor_node:main',
            'spectro_monitor_node = spectro.spectro_monitor_node:main',
        ],
    },
)
