"""
SETUPTOOLS FOR DWA PATH API.
"""

import io
import os
import sys
from configparser import SafeConfigParser
from shutil import copy
from distutils.command.install import install as _install
from distutils.dir_util import copy_tree
from setuptools import find_packages, setup
from setuptools.command.install import install

PATH = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(PATH, "README.md"), encoding="utf-8") as f:
    DESCRIPTION = "\n" + f.read()

class install_(install):
    """ Post-installation for installation mode. """

    def run(self):
        ret = None
        if self.old_and_unmanageable or self.single_version_externally_managed:
            ret = _install.run(self)
        else:
            caller = sys._getframe(2)
            caller_module = caller.f_globals.get("__name__", "")
            caller_name = caller.f_code.co_name

            if caller_module != "distutils.dist" or caller_name != "run_commands":
                _install.run(self)
            else:
                self.do_egg_install()

        return ret

setup(
    name="dwapath",
    version="1.0.0",
    description="Open source robotic path planning tool powered by Numpy and Pandas.",
    long_description=DESCRIPTION,
    maintainer="Chia-Hua Peng",
    maintainer_email="chia-hua.peng@noblis.org",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.6.5",
    install_requires=
    [
        "numpy",
        "pandas",
        "matplotlib"
    ],
    cmdclass={"install": install_},
)