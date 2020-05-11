import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bio_rtd',
    author='Jure Sencar',
    author_email='jure@timetools.eu',
    version='0.7.2',
    description='Library for modeling residence time distributions (RTD)'
                ' of integrated continuous biomanufacturing processes.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url='https://github.com/open-biotech/bio-rtd/',
    project_urls={
        "Bug Tracker": "https://github.com/open-biotech/bio-rtd/issues/",
        "Documentation": "https://open-biotech.github.io/bio-rtd-docs/",
        "Source Code": "https://github.com/open-biotech/bio-rtd/",
    },

    packages=["bio_rtd",
              "bio_rtd.chromatography",
              "bio_rtd.uo",
              "bio_rtd.utils"],

    keywords=[
        'residence time distribution',
        'biomanufacturing'
    ],
    install_requires=[
        'numpy',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Manufacturing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.7',
)
