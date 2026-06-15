#!/bin/bash -eu
# ClusterFuzzLite build script — compiles each Atheris harness in tests/fuzz/
# via OSS-Fuzz's compile_python_fuzzer helper.

cd "$SRC/cvxrisk"

# Install cvxrisk and its runtime dependencies (clarabel, cvx-linalg, numpy,
# scipy) so PyInstaller can discover and bundle them into the frozen fuzzer
# binaries. Without this the harness imports would be unresolved at runtime.
pip3 install .

for fuzzer in tests/fuzz/fuzz_*.py; do
  compile_python_fuzzer "$fuzzer"
done
