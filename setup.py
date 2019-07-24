import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coach-ml",
    version="0.11",
    author="Loren Kuich",
    author_email="loren@lkuich.com",
    description="Python client for coach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lkuich/coach-python",
    packages=setuptools.find_packages(),
    install_requires=['requests', 'numpy==1.16.4', 'tensorflow==1.12.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
