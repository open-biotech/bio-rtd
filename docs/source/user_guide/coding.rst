Coding
======

General guidelines
------------------

We recommend copying one of the :ref:`Models` that most closely resembles your needs and modifying it accordingly.

Use :ref:`Templates` to create instances of individual unit operations.

If you need to define new unit operations classes (or other elements of the library),
then make sure they extend proper base classes.

Conventions
-----------

We follow ``yx`` convention for shaping numpy arrays. In our case ``x`` is typically time axis (``t``)
and ``y`` corresponds to process fluid species.

Simulation time vector (``t``) is a 1D ``np.ndarray``. It should start with ``0`` and have a fixed step size. The same
time
vector should be used across the model (for inlet and all the unit operations).

Vector for flow rate ``f`` is also a 1D ``np.ndarray``.

Array for concentration profile is 2D ``np.ndarray`` with shape ``(n_species, n_time_steps)``. In case of single
specie,
the shape is ``(1, n_time_steps)`` and not ``(n_time_steps,)``.

Single underscore prefix ``_`` annotates private functions and variables which should be only used inside the class
or function.

**Variable names**

* ``t`` - simulation time vector
* ``dt`` - time step size
* ``i`` - time step index on time vector (``t[i] == dt * i``)
* ``f`` - process fluid flow rate
* ``rt`` - residence time
* ``rt_mean`` - mean residence time (= flow-through time)
* ``rt_target`` - target ``rt_mean`` at steady-state; typically used to determine the size of unit operations
* ``v`` - volume
*  ``v_void`` - void volume; usually effective void volume, thus excluding hold-up zones (``rt_mean`` = ``v_void`` / ``f``)
* ``v_init`` - initial volume of fluid in unit operation (e.g. if surge tank starts with 50 % pre-fill, the ``v_init
  `` = 0.5 * ``v_void``; 0.5 could also be specified as ``v_init_ratio`` = 0.5)
* ``m`` - mass
* ``uo`` - unit operation
* ``fc_uo`` - fully-continuous unit operation (accepts and provides constant flow rate)
* ``sc_uo`` - semi-continuous unit operation (accepts constant flow rate and provides irregular flow rate)
* ``surge_tank`` - surge tank (accepts irregular or constant flow rates and provides constant* flow rate)
* ``pdf`` - probability distribution function; ``sum(pdf * time_step) == 1``
* ``p`` - probability distribution vector; ``p = pdf(t)``

All names of the time dependent vectors or arrays
are thus starting with ``f_``, ``c_``, ``m_`` or ``p_``.

Constant flow rate profile can be clipped at the beginning or at the end, resulting
in a box-shaped profile.

