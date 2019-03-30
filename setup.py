from setuptools import setup

setup(
    name='dmc',
    version='0.1',
    packages=['dmc'],
    package_dir={'': 'dmc'},
    install_requires=[],
    url='https://github.com/marvinlwenzel/thb_dmc_ss19',
    license='MIT',
    author='Marvin Lukas Wenzel',
    author_email='wenzel@th-brandenburg.de',
    description='Data Mining related implementations',
    long_description='Implementations for the 2019 Data Mining lecture at the University of Applied Sciences Brandenburg.',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',

    ],
    keywords='thb data datamining',
    python_requires='>=3.6.*, <4',
    extras_require={
        'test': ['pyhamcrest'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/marvinlwenzel/thb_dmc_ss19/issues',
        'Our university department': 'https://informatik.th-brandenburg.de/',
        'Moodle Course': 'https://moodle.th-brandenburg.de/course/view.php?id=2137',
    }
)
