# bio-rtd library
bio-rtd library is a python library for modeling
residence time distributions (RTD)
of integrated continuous biomanufacturing processes.

## Version
Current version: 0.7.1

## Requirements

Python 3.7.+

Core dependencies:

    numpy
    scipy

Packages for visual representation in examples:

    bokeh
    xlrd

Packages for importing data from Excel in examples:

    pandas

## Set up

Download or clone the github repo:

    git clone https://github.com/open-biotech/bio-rtd.git

Set local destination (`/path_to/bio-rtd`) as working directory
or add it to the python path.


## Getting started

Examples can be run as scripts:

    python examples/models/single_pcc.py

Model examples ending with `_gui.py`
can be run via `bokeh serve` command:

    bokeh serve examples/models/integrated_mab_gui.py
    
or
    
    python bokeh serve examples/models/integrated_mab_gui.py

For more information see the [Documentation](https://open-biotech.github.io/bio-rtd-docs/).

## Documentation

The documentation can be accessed at https://open-biotech.github.io/bio-rtd-docs/ or by building a local version.

To build a local version of documentation install the following packages:

    pip install sphinx sphinx_autodoc_typehints sphinx_rtd_theme

run command:

    make html

and open `docs/build/html/index.html` with a web browser.


## Meta information

Distributed under the MIT license. See ``LICENSE`` for more information.

For technical issues, bug reports and feature request use
[issue tracker](https://github.com/open-biotech/bio-rtd/issues).

If you want to contribute to the code, see
[Developers guide](https://open-biotech.github.io/bio-rtd-docs/user_guide/development.html).

If you are using the library in your projects please let 
[us know](mailto:jure.sencar@boku.ac.at).
This way we know how much interest there is, what is the scope of usage,
the needs, etc. This information influences the future development of the project.


E-mail: [jure.sencar@boku.ac.at](mailto:jure.sencar@boku.ac.at)

University of Natural Resources and Life Sciences (BOKU), Vienna

## Referencing the library

 If you are using this package for a scientific publication,
 please add the following reference:

* Senčar, J., (2020) GitHub Repository,
   https://github.com/open-biotech/bio-rtd.

## Acknowledgements
This work was supported by:

- The Federal Ministry for Digital and
  Economic Affairs (bmwd), the Federal Ministry for Transport,
  Innovation and Technology (bmvit),
  the Styrian Business Promotion Agency SFG,
  the Standortagentur Tirol,
  Government of Lower Austria,
  and ZIT - Technology Agency of the City of Vienna
  through the COMET-Funding Program managed
  by the Austrian Research Promotion Agency FFG
- Baxalta Innovations GmbH (now part of Takeda)
- Bilfinger Industrietechnik Salzburg GmbH
