from setuptools import find_packages, setup

package_name = 'rack_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammad',
    maintainer_email='s.mohammadreza.kasaei@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'service = rack_detection.service:main',
             'client = rack_detection.client_member_function:main',
             'visualizer = rack_detection.pointcloud:main',        
             'service_convnet = rack_detection.service_convnet:main',
             'client_grasp_predict = rack_detection.client_grasp_prediction:main',
                  
        ],
    },
)
