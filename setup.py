import pathlib
from setuptools import setup

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
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["featurediscovery"],
    include_package_data=True,
    install_requires=[
        "pandas>=1.0.0"
        , 'scikit-learn>=1.0.0'
        , 'cupy-cuda>=9.6.0'
        , 'matplotlib>=3.0.0'
        , 'scipy>=1.7.0'
      ],
    entry_points={
        "console_scripts": [
        ]
    },
)