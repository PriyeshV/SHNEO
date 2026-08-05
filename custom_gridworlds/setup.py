# run ' pip install -e . ' in the custom gridworlds directory

from setuptools import setup

setup(name="custom_gridworlds",
      version="0.4",
      license="MIT",
      packages=["simple_maze"],
      # package_data={"custom_envs.portal_maze.envs": ["maze_samples/*.npy"]},
      install_requires=["gymnasium", "pygame", "numpy"]
)
