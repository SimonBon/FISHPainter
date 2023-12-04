from setuptools import setup, find_packages

package_name = "FISHcreation"

def read_requirements():
    with open('requirements.txt', 'r') as file:
        return [line.strip() for line in file.readlines()]

setup(
    name=package_name,
    version='{{VERSION_PLACEHOLDER}}',
    packages=find_packages(),
    install_requires=read_requirements(),
    author="Simon Gutwein",
    include_package_data=True,
    author_email="simon.gutwein@ccri.at",
    description="",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/SimonBon/FISHcreation",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
