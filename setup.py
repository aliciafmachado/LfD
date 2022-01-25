import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lfd",
    version="0.0.1",
    author="Alicia Fortes Machado",
    author_email="aliciafortesmachado@gmail.com",
    description="Package with Learning from Demonstration (LFD) algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aliciafmachado/LfD",
    project_urls={
        "Bug Tracker": "https://github.com/aliciafmachado/LfD/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"lfd": "src"},
    # where="src"
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
)

print(setuptools.find_packages())