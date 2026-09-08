"""avsim -- a modular 2-D autonomous-driving simulator.

Layers, bottom to top:

``avsim.core``        conventions, integrators, SE(2), planar geometry
``avsim.models``      parameters, tires, suspension, powertrain, vehicle models
``avsim.world``       paths, road network, signals, other traffic, the container
``avsim.perception``  a simple vision sensor and a multi-object tracker
``avsim.planning``    prediction, velocity profile, behaviour, Frenet lattice
``avsim.control``     geometric baselines, constrained iLQR, the vehicle NMPC
``avsim.autonomy``    the integrated stack
``avsim.eval``        KPIs, scenarios, the runner, the lecture's study tasks
``avsim.viz``         figures and animations (needs matplotlib)

Start with :func:`avsim.eval.runner.run_scenario` or the ``avsim`` CLI.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
