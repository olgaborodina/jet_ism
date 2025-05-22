from setuptools import setup, find_packages

setup(
    name="jet_ism",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "h5py>=3.6.0",
        "matplotlib>=3.4.0",
        "Pillow>=8.3.0",
        "natsort>=7.1.0",
    ],
    python_requires=">=3.7",
    author="Olga Borodina",
    description="ISM simulation analysis package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 
