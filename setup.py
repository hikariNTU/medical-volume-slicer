from setuptools import setup

import pathlib

description = (
    'A convenient tool to generate medical images into different directional slice.'
)

try:
    here = pathlib.Path(__file__).parent.resolve()
    # Get the long description from the README file
    long_description = (here / 'README.md').read_text(encoding='utf-8')
except FileNotFoundError:
    long_description = description
except:
    raise


setup(
    name='medical-volume-slicer',
    version='0.0.1',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='git@github.com:hikariNTU/medical-volume-slicer.git',
    author='Lian Chung, HikariTW',
    # author_email='emailaddress@example.com',
    license='GNU General Public License v3.0',
    packages=['medical_volume_slicer'],
    python_requires='>=3.6, <4',
    install_requires=['medpy'],
    zip_safe=False,
)
