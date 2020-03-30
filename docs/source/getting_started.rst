Getting Started
===============

Requirements
------------
Python 3.7.+

Packages::

 numpy
 scipy
 bokeh
 xlrd
 pandas

* ``numpy`` and ``scipy`` are basic requirements.
* ``bokeh`` and ``xlrd`` are needed for plotting and/or GUI.
* ``pandas`` is needed for reading data from Excel in one of the examples.

It is generally recommended to use `python virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ or
`conda
virtual
environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Setup
-----

Clone git repo:

.. code-block:: bash

   git clone https://github.com/open-biotech/bio-rtd.git

Go to repo folder:

.. code-block:: bash

   cd bio-rtd

Install requirements:

.. code-block:: bash

   pip install -r requirements.txt

Running first example
---------------------

Run an example:

.. code-block:: bash

   python example/models/single_pcc.py

If you see:

.. code-block:: python

    ModuleNotFoundError: No module named 'bio_rtd'

then you need to add repo to PYTHONPATH (either in your IDE or in terminal):

PyCharm

.. code-block::

   Check "Add source root to PYTHONPATH" under
   "Run/Debug configuration".


Terminal

.. code-block:: bash

   export PYTHONPATH=${PYTHONPATH}:`pwd`

Running example with bokeh serve
--------------------------------

Examples that end with ``_gui.py`` are
python scripts for creating an interactive web application.

In background a `Bokeh Server` instance is created. The server connects UI elements with python script (see
more at `Building Bokeh Applications <https://docs.bokeh
.org/en/latest/docs/user_guide/server
.html#userguide-server-applications>`_).


----

**Terminal**

.. code-block:: bash

   bokeh serve --show example/models/integrated_mab_gui.py

``bokeh`` can also run as a python script (good for debugging):


.. code-block:: bash

   python `which bokeh` serve --show example/models/integrated_mab_gui.py

----

**PyCharm**

To run ``python bokeh serve`` with PyCharm, set the following *Run Configuration*:

.. code-block::

    Configuration: python
    Script path: /path_to/bokeh
    Parameters: serve --show /example/models/integrated_mab_gui.py

where ``/path_to/bokeh`` can be obtained by running ``which bokeh`` command in terminal in PyCharm.

----

Also make sure the repo is added to the PYTHONPATH as described in :ref:`Running first example`.


Flag ``--show`` is optional and runs the newly created instance in a new tab in a web browser.
Only one server instance can be run (on the same port). If you try to run another ``bokeh serve`` command while one
is running, you will see the following exception:

.. code-block:: bash

   Cannot start Bokeh server, port 5006 is already in use

In that case find and close the existing running ``bokeh serve`` process.
