from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["cooperative_link", "cooperative_link.geometry", "cooperative_link.link"],
    package_dir={"": "src"},
)

setup(**d)
