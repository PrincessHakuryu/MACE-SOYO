"""Mark wheels containing the optional LASP-D3 library as platform-specific."""
from pathlib import Path
from setuptools import Distribution, setup


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return Path("mace_soyo/utils/libd3.so.1.0.0").is_file()


setup(distclass=NativeDistribution)
