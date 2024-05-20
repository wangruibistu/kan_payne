"""
Author:         Rui Wang
Created:        2024-05-18
Modified-History:
    2024-05-18, Rui Wang, created
"""
import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


with open("requirements.txt", "r") as f:
    requirements = [
        req.strip()
        for req in f.readlines()
        if not req.startswith("#") and req.__contains__("==")
    ]


setuptools.setup(
    name="kan_payne",
    version="0.0.2",
    author="Wang Rui",
    author_email="wangrui@nao.cas.cn",
    description="The payne emulator based on KAN",  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangruibistu/KAN_payne",
    project_urls={
        "Source": "https://github.com/wangruibistu/KAN_payne",
    },
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    include_package_data=True,
    package_data={
    },
    install_requires=requirements,
    python_requires=">=3.11",
)
