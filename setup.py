from setuptools import setup, find_packages

def get_requirements():
    with open("./requirements.txt", "r") as f:
        requirements = f.readlines()
    return requirements


setup(
    name="adabkb",
    version="1.1.0",
    description="Ada-BKB: Adaptive Budgeted Kernelized Bandit",
    python_requires='~=3.6',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
    install_requires=get_requirements(),
    include_package_data=True,
)
