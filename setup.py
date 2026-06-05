from pathlib import Path

from setuptools import find_packages, setup


FILE = Path(__file__).resolve()
PARENT = FILE.parent
README = (PARENT / "README.md").read_text(encoding="utf-8")


def parse_requirements(file_path):
    requirements = []
    for line in Path(file_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line.split("#")[0].strip())
    return requirements


setup(
    name="ml-detr",
    python_requires=">=3.8",
    license="AGPL-3.0",
    description="ML-DETR based on Ultralytics RT-DETR.",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["ultralytics", "ultralytics.*"]),
    package_data={"ultralytics": ["**/*.yaml"]},
    include_package_data=True,
    install_requires=parse_requirements(PARENT / "requirements.txt"),
    entry_points={"console_scripts": ["yolo = ultralytics.cfg:entrypoint", "ultralytics = ultralytics.cfg:entrypoint"]},
)
