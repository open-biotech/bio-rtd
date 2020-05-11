Development
===========

Unit tests
----------

Each file in ``rtd_lib`` has a corresponding test file with prefix ``test_`` placed in ``rtd_lib_test`` folder.

To run all tests and asses code coverage (the share of code tested) using the ``coverage`` package, run the following
command in terminal:

.. code-block:: bash

    coverage run --source=./bio_rtd -m unittest discover bio_rtd_test; coverage report

To see the detailed coverage analysis (e.g. to discover non-covered lines), run:

.. code-block:: bash

   coverage html

and open ``htmlcov/index.html`` in web browser.

Running tests without code coverage:

.. code-block:: bash

  python -m unittest discover bio_rtd_test

If you create a pull request, please add appropriate tests, make sure all tests succeed and keep complete (100 %)
code coverage. If needed, also update the documentation.


Documentation
-------------

Dependencies (``pip`` packages):

.. code-block:: bash

   sphinx
   sphinx_autodoc_typehints
   sphinx_rtd_theme
   rst2pdf

To generate the documentation from script, run:

.. code-block:: bash

   make html

and open ``docs/build/html/index.html``.

To generate the documentation in form of PDF, run:

.. code-block:: bash

    sphinx-build -b pdf docs/source docs/build

To clean the build directory, run:

.. code-block:: bash

    make clean



Pull requests
-------------

To contribute to the library, please `create a pull request <https://help.github
.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull
-request>`_ on `GitHub <https://github.com/open-biotech/bio-rtd.git>`_.

Checklist before making a pull request:

* All unit tests need to succeed.
* Ensure 100 % code coverage with unit tests.
* Update docstrings and documentation.


