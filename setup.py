from setuptools import find_packages, setup

setup(
    name="ges3vig",
    py_modules=["ges3vig"],
    version="1.0",
    author="AnonCVPRSubmitter",
    description="Ges3Vig",
    packages=find_packages(include=("ges3vig*")),
    install_requires=[
        f"clip @ git+https://github.com/eamonn-zh/CLIP.git", "lightning", "wandb", "scipy", "hydra-core",
        "h5py", "open3d", "pandas"
    ]
)
