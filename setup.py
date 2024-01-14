from setuptools import setup, find_packages

with open("requirements.txt", "r") as fp:
    pkg_requirements = [i.strip() for i in fp.readlines() if len(i.strip()) != 0]


setup(
    name='gbox',
    version='0.1.1',
    author='Rajesh Nakka',
    author_email='33rajesh@gmail.com',
    description='Geometry Box: A simple package for working with basic geometry shapes',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10.*',
    instal_requires=pkg_requirements,
)
