from setuptools import find_packages, setup

setup(
    name="GPAT",
    version="1.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "scipy",
        'matplotlib',
        'japanize-matplotlib',
        'plotly',
    ],
    license=(
        "The MIT License applies only to members of the Owada Laboratory "
        "of the Department of Industrial Administration Faculty of Science"
        "andTechnology, Tokyo University of Science, and members of "
        "the laboratory's development team Fabre, or persons authorized "
        "by those members."
    ),
    description="A library for human pose estimation using mmpose.",
)