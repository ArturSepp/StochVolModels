from setuptools import setup, find_packages


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


def read_file(file):
    with open(file) as f:
        return f.read()


long_description = read_file("README.md")
requirements = read_requirements("requirements.txt")

setup(
    name='stochvolmodels',
    version='1.0.24',
    author='Artur Sepp, Parviz Rakhmonov',
    author_email='artursepp@gmail.com, parviz.msu@gmail.com',
    url='https://github.com/ArturSepp/StochVolModels',
    description='Implementation of Stochastic Volatility Models',
    long_description_content_type="text/x-rst",  # If this causes a warning, upgrade your setuptools package
    long_description=long_description,
    license="MIT license",
    packages=find_packages(exclude=["docs", "stochvolmodels/my_papers"]),  # Don't include test directory in binary distribution
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)