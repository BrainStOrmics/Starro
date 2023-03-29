from setuptools import find_packages, setup


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]


with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()


setup(
    name="starro",
    version="1.0.0",
    python_requires=">=3.7",
    install_requires=read_requirements("requirements.txt"),
    packages=find_packages(exclude=("tests", "docs")),
    author="Hailin Pan, Kyung Hoi (Joseph) Min, Zehua Jing",
    author_email="panhailin@genomics.cn",
    description="Starro: a uniform framework of cell segmentation on spatially resolved transcriptomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL",
    url="https://github.com/Bai-Lab/Starro",
)
