from setuptools import setup

package_name = 'go1_walk'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/policy', ['policy/policy.npz', 'policy/policy.npz.sha256']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Walking policy executor for Go1',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['walk_node = go1_walk.walk_node:main']},
)
