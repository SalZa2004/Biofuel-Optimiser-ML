# setup.py
from setuptools import setup, find_packages
def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="biofuel-ml",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=parse_requirements("requirements.txt")
)