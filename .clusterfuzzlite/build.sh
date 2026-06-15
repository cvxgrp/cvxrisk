#!/bin/bash -eu
# ClusterFuzzLite build script — compiles each Atheris harness in tests/fuzz/
# via OSS-Fuzz's compile_python_fuzzer helper.

cd "$SRC/cvxrisk"

# Install cvxrisk and its runtime dependencies (clarabel, cvx-linalg, numpy,
# scipy) so PyInstaller can discover and bundle them into the frozen fuzzer
# binaries. Without this the harness imports would be unresolved at runtime.
pip3 install .

# numpy/scipy/clarabel ship C/Rust extensions whose submodules PyInstaller's
# static analysis does not fully discover on its own (e.g. numpy._core._*),
# which makes the frozen binary crash on import. --collect-all forces every
# submodule, data file, and shared library of these packages to be bundled.
for fuzzer in tests/fuzz/fuzz_*.py; do
  compile_python_fuzzer "$fuzzer" \
    --collect-all numpy \
    --collect-all scipy \
    --collect-all clarabel
done
