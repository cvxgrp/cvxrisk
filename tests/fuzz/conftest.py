"""Keep the Atheris fuzz harnesses out of the normal pytest run.

The ``fuzz_*.py`` modules import ``atheris`` and are meant to be compiled and
driven by ClusterFuzzLite, not collected by pytest. The project enables
``--doctest-modules``, which would otherwise import every module here looking
for doctests and fail with ``ModuleNotFoundError: No module named 'atheris'``
in environments where the fuzzing extras are not installed.
"""

from __future__ import annotations

collect_ignore_glob = ["fuzz_*.py"]
