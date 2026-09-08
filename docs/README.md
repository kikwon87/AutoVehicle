# Documentation

* [`design.md`](design.md) — why the code is shaped the way it is: twenty-eight
  decisions, each with its alternative and what that alternative costs. Most of
  the entries were written after the alternative had actually been tried and had
  produced a specific, measured failure.
* [`controller_api.md`](controller_api.md) — the **control algorithm interface
  report** (제어 알고리즘 인터페이스 규격서): everything a controller receives,
  everything it may command, how a `.py` plug-in is found and loaded, and how
  the run is scored. This is the file to hand to somebody writing their own
  algorithm for the test platform. Templates to copy are in
  [`../examples/controllers/`](../examples/controllers/).
* The project [`README`](../README.md) covers install, architecture, the test
  platform, conventions, the reproduced lecture figures, the optimal-control
  problem, and the known limitations.
* `avsim contract` prints the **model contract** for the running example:
  coordinates, model regime, integrator, step, horizon, parameters, constraints,
  transcription, derivatives and the inter-sample treatment — Study Task 5.
* `avsim study` runs the other four study tasks and prints their numbers.
* `avsim platform` starts the test platform; `--headless` scores one preset from
  the command line, which is how a controller gets into a CI job.
