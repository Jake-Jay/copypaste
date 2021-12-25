from setuptools import setup

setup(
    name='copypaste',
    version='0.0.1',
    description='Copy Paste For Semi Supervised Representation Learning',
    url='https://github.com/shuds13/pyexample',
    author='Jake Pencharz',
    author_email='jake.pencharz@bayer.com',
    license='BSD 2-clause',
    packages=['pyexample'],
    install_requires=[
        'pytorch>=1.10.0',
        'numpy',
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ],
)
