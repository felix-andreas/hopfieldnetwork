from setuptools import setup, find_packages
import os

base_path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(base_path, "README.md")) as file:
    readme = file.read()

with open(os.path.join(base_path, "hopfieldnetwork", "__about__.py")) as file:
    about = {}
    exec(file.read(), about)

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    author=about["__author__"],
    license=about["__license__"],
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "pillow"],
    python_requires=">=2.7",
    package_data={
        "hopfieldnetwork": ["data/**/*"],
    },
    entry_points={
        "console_scripts": ["hopfieldnetwork-ui=hopfieldnetwork.gui:start_gui"]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
)
