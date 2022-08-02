import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()
long_description = ""

setuptools.setup(
    name="map-sort",
    version="0.1",
    author="Caleb Vatral",
    author_email="caleb.m.vatral@vanderbilt.edu",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy',
          'scipy',
          'filterpy',
          'shapely'
      ],
    python_requires='>=3.5',
)