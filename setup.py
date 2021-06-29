from setuptools import setup, find_packages

setup(
    name="adabbkb",
    version="0.0.1",
    description="Ada-BBKB: Adaptive Batch Budgeted Kernelized Bandit",
    python_requires='~=3.6',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'sklearn', 'pytictoc', 'matplotlib'],
    include_package_data=True,
)
