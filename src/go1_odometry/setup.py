from glob import glob

from setuptools import setup

package_name = 'go1_odometry'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bylba0',
    maintainer_email='176760298+bylba0@users.noreply.github.com',
    description='Hackathon Go1 T4 odometry',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['odometry = go1_odometry.node:main']},
)
