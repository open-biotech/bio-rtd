Features
========

Inlet profiles
--------------

Available inlet profiles

- Constant flow rate, constant concentration :class:`bio_rtd.inlet.ConstantInlet`
- Constant flow rate, box-shaped concentration profile :class:`bio_rtd.inlet.IntervalInlet`
- Custom flow rate, custom concentration profile :class:`bio_rtd.inlet.CustomInlet`

Unit operations
---------------

Unit operations are split in following groups:

- Fully-continuous :mod:`bio_rtd.uo.fc_uo` (accept and provide constant flow rate)
- Semi-continuous :mod:`bio_rtd.uo.sc_uo` (accept constant and provide periodic flow rate)
- Surge tank :mod:`bio_rtd.uo.surge_tank` (accept periodic and constant flow rate)
- Special :mod:`bio_rtd.uo.special_uo` (the ones that do not fit in categories above)


All unit operations can be instructed to discard parts of inlet or outlet process fluid stream
in oder to optimize the start-up phase.

For common attributes among unit operations check :class:`bio_rtd.core.UnitOperation` class.
For complete parameter set of individual
unit operation, check its API by clicking the class name.

Here are listed key features of unit operations:

:class:`bio_rtd.uo.fc_uo.Dilution`

- Instant dilution of the process fluid stream.

:class:`bio_rtd.uo.fc_uo.Concentration`

- Instant concentration of the process fluid stream.
- One can specify retained species and losses during concentration step.

:class:`bio_rtd.uo.fc_uo.BufferExchange`

- Instant inline buffer exchange.
- One can specify retained species, losses and efficiency.

:class:`bio_rtd.uo.fc_uo.FlowThrough`

- Propagation of the process fluid stream through a fixed unit operation (most common use case).
- A probability distribution function is specified to describe the propagation dynamics.
- Offers setting equilibration and wash buffer composition (for more parameters check the class link).

:class:`bio_rtd.uo.fc_uo.FlowThroughWithSwitching`

- Extension of the :class:`bio_rtd.uo.fc_uo.FlowThrough`.
- Allows switching unit operations during run (e.g. for alternating flow-through chromatography).

:class:`bio_rtd.uo.sc_uo.ACC`

- Alternating column chromatography (without recycling of the overloaded material).
- Describing binding dynamics via :class:`bio_rtd.core.BreakthroughProfile`.
- Option to specify load duration based on breakthrough material.
- Material in elution peak is homogenized. Various peak cut methods are available.

:class:`bio_rtd.uo.sc_uo.PCC`

- Periodic counter-current chromatography.
- Extension of :class:`bio_rtd.uo.sc_uo.ACC`.
- Option to recycle breakthrough material during load and/or wash step.

:class:`bio_rtd.uo.sc_uo.PCCWithWashDesorption`

- Extension of :class:`bio_rtd.uo.sc_uo.PCC`.
- Option to simulate desorption of captured material during wash step.

:class:`bio_rtd.uo.surge_tank.CSTR`

- Ideal CSTR.
- Offers specifying initial fill level.
- Size can be determined based on specified 'safety margin', e.g. 10 %

:class:`bio_rtd.uo.special_uo.ComboUO`

- Unit operation that combines several unit operations and presents them as one.

Some unit operations can be described with a set of simpler unit operation,
but we might want to have them appear (e.g. in plots) as one.
Typical use-case would be describing filtration or diafiltration step with a
combination of Concentration, Dilution and/or FlowThrough unit operations.
In such case, one can use ComboUO as a container.

Probability distribution functions
----------------------------------

Available pdf peak shapes:

- Gaussian: :func:`bio_rtd.peak_shape.gaussian`
- Exponentially modified Gaussian: :func:`bio_rtd.peak_shape.emg`
- Skewed normal: :func:`bio_rtd.peak_shape.skew_normal`
- N tanks in series (N = 1 for exponential decay): :func:`bio_rtd.peak_shape.tanks_in_series`

Available PDF classes (wrappers around pdf peak shapes):

- :class:`bio_rtd.pdf.GaussianFixedDispersion`
- :class:`bio_rtd.pdf.GaussianFixedRelativeWidth`
- :class:`bio_rtd.pdf.ExpModGaussianFixedDispersion`
- :class:`bio_rtd.pdf.ExpModGaussianFixedRelativeWidth`
- :class:`bio_rtd.pdf.TanksInSeries`

Fixed dispersion:

   sigma = (void_volume * dispersion_index) ** 0.5

Fixed relative width:

   sigma = void_volume * relative_width

Logging
-------

Custom loggers are implemented in order to provide
control over log messages and storing intermediate data.

See :ref:`RtdLogger` API for more info.


