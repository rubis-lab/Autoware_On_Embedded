from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['lgsvl_rosbridge_library', 'lgsvl_rosbridge_library.internal', 'lgsvl_rosbridge_library.capabilities', 'lgsvl_rosbridge_library.util'],
    package_dir={'' : 'src'},
)

setup(**d)
