from setuptools import setup, find_packages

setup(
    name='tmdprocessor',
    version='0.1.0',
    description='A library for processing TMD files (TrueMap v6 and GelSight) and visualizing height maps.',
    author='Antoine Boucher',
    author_email='antoine@antoineboucher.info',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'plotly',
        'Pillow'
    ],
    entry_points={
        'console_scripts': [
            'tmdprocessor=tmdprocessor.cli:main'
        ]
    },
)
