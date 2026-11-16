from setuptools import setup

package_name = 'go1_safety'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bylba0',
    maintainer_email='176760298+bylba0@users.noreply.github.com',
    description='Hackathon Go1 T2 Safety Block',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['safety_block = go1_safety.node:main']},
)
