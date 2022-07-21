from setuptools import setup

long_description = open('README.md').read()

setup(
    name="pygmmis",
    version='1.2.3',
    description="Gaussian mixture model for incomplete, truncated, and noisy data",
    long_description = long_description,
    long_description_content_type='text/markdown',
    author="Peter Melchior",
    author_email="peter.m.melchior@gmail.com",
    license='MIT',
    py_modules=["pygmmis"],
    url="https://github.com/pmelchior/pygmmis",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    install_requires=["numpy","scipy","parmap>=1.5.2"]
)
