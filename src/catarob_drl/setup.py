from setuptools import find_packages, setup

package_name = 'catarob_drl'

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
    maintainer='dylan_trows',
    maintainer_email='TRWDYL001@myuct.ac.za',
    description='Catarob DRL package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drl_agent = catarob_drl.DRL_Agent:main',
            'vrx_controller = catarob_drl.VRX_Controller:main',
        ],
    },
)
