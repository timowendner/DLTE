from setuptools import setup, find_packages

setup(
    name='DLTE',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'mido',
        'torch',
    ],
    author='Timo Wendner',
    author_email='timo.wendner@gmail.com',
    description='Torch Project to predict the Tempo for Midi files',
    url='https://github.com/timowendner/MeterDetection/settings',
)
