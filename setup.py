from distutils.core import setup

setup(
    name="pygmmis",
    description="Gaussian mixture model for incomplete, truncated, and noisy data",
    long_description="Gaussian mixture model for incomplete, truncated, and noisy data",
    version='1.0',
    author="Peter Melchior",
    author_email="peter.m.melchior@gmail.com",
    py_modules=["pygmmis"],
    url="https://github.com/pmelchior/pygmmis",
    requires=["numpy","scipy","multiprocessing","parmap"]
)
