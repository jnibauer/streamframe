from setuptools import setup, find_packages

setup(
    name='streamframe',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'jax',
    ],
    url='https://github.com/jnibauer/streamframe',
    license='MIT',
    author='Your Name',
    author_email='jnibauer@princeton.edu',
    description='A small package for generating stream frames with Jax'
)