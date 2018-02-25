from setuptools import setup

setup(
    name="pygmmis",
    description="Gaussian mixture model for incomplete, truncated, and noisy data",
    long_description="Gaussian mixture model for incomplete, truncated, and noisy data",
    version='1.0.1',
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
    requires=["numpy","scipy","multiprocessing","parmap"]
)
