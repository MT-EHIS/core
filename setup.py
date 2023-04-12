from os import path
from setuptools import setup, find_packages
from io import open

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "mt-ehis-core/version.py")) as f:
    exec(f.read())


setup(
    name="mtehis",
    version=__version__,
    description="Maintenance Tool for Evolving Hybrid Intelligent Systems: Core Package.",
    long_description=open("./README.md").read(),
    long_description_content_type="text/markdown",
    author="Oleksandr Pokhylenko",
    author_email="pokhilenko.alex@gmail.com",
    packages=find_packages(exclude=["test"]),
    platforms=["all"],
    install_requires=["streamad==0.3.0"],
    include_package_data=True,
    url="https://github.com/MT-EHIS/core",
    setup_requires=["setuptools==58.2.0"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
