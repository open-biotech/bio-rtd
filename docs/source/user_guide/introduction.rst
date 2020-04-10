Introduction
============

Residence time distribution (RTD)
---------------------------------

Let's simulate a protein pulse response measurement on a small flow-through column:

.. bokeh-plot:: ../../examples/documentation/tutorial_1a.py

Let's fit an exponentially modified gaussian distribution to the peak using
the reference points in blue circles.

    Probability distribution classes, such as
    :class:`bio_rtd.pdf.ExpModGaussianFixedDispersion`
    allow the probability distribution functions (pdf)
    to be dependent on process parameters and inlet flow rate.

.. bokeh-plot:: ../../examples/documentation/tutorial_1b.py

Let's expand the example by introducing flow-through unit operation
which uses pdf:

.. bokeh-plot:: ../../examples/documentation/tutorial_1c.py

Simulating breakthrough profile with the same unit operation:

.. bokeh-plot:: ../../examples/documentation/tutorial_1d.py

Simulating breakthrough profile with  :class:`bio_rtd.core.RtdModel` with inlet and two unit operations.

.. bokeh-plot:: ../../examples/documentation/tutorial_1e.py

See :ref:`Features` section for features.

See :ref:`Examples` section for more examples.

See :ref:`Templates` section on how to set up specific unit operation.

See :ref:`API Reference` for detailed info about parameters,
attributes and methods for each unit operation.

Creating custom rtd model
-------------------------

We recommend checking :ref:`Models` and making a local copy of one that
most closely resembles you needs and modify it accordingly.
To define new instances of unit operations, use the parameter
and attribute list from :ref:`Templates`.

Also check :ref:`Coding` section in order to better
understand the coding style in *bio_rtd*.
