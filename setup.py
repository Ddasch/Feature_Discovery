import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="featurediscovery",
    version="0.1.0",
    description="Discover hidden predictive qualities of features in your dataset by testing them for linear separability after applying different kernel transformations.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Ddasch/Feature_Discovery",
    author="Frederik Schadd",
    author_email="featurediscovery@outlook.com",
    license="LGPL v2.1",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Environment :: GPU :: NVIDIA CUDA :: 11.5",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    packages=find_packages(include=('featurediscovery*',)),
    include_package_data=True,
    install_requires=[
        "pandas>=1.0.0"
        , 'scikit-learn>=1.0.0'
        , 'cupy-cuda115>=9.6.0'
        , 'matplotlib>=3.0.0'
        , 'scipy>=1.7.0'
      ],
    entry_points={
        "console_scripts": [
        ]
    },
)