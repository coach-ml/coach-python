import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lkuich",
    version="0.0.1",
    author="Loren Kuich",
    author_email="loren@lkuich.com",
    description="Python client for coach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lkuich/coach-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)