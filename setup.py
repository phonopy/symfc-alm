"""Setup script."""
from setuptools import setup

setup(
    name="symfc-alm",
    version="0.1",
    setup_requires=["numpy", "setuptools"],
    description="This is the symfc-alm module.",
    author="Atsushi Togo",
    author_email="atz.togo@gmail.com",
    install_requires=["numpy", "alm"],
    provides=["symfc_alm"],
    platforms=["all"],
)
