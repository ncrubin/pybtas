from skbuild import setup

setup(
    name="pybtas",
    version="0.0.1",
    author="Nicholas C. Rubin",
    author_email="rubinnc0@gmail.com",
    description="Exposing BTAS to python",
    packages=["pybtas"],
    package_dir={"": "src"},
    cmake_install_dir="src/pybtas",
)