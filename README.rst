.. raw:: html

  <div align="center">
    <h6>Tools required for Galaxy Image Analysis</h6>
    <h1>
      <a href="https://github.com/BMCV/giatools">giatools</a><br>
      <a href="https://github.com/BMCV/giatools/actions/workflows/testsuite.yml"><img src="https://github.com/BMCV/giatools/actions/workflows/testsuite.yml/badge.svg" /></a>
      <a href="https://github.com/BMCV/giatools/actions/workflows/testsuite.yml"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/kostrykin/07509ac0c0aa1d5a65ca03806bd3600b/raw/giatools.json" /></a>
      <a href="https://anaconda.org/bioconda/giatools"><img src="https://img.shields.io/badge/Install%20with-conda-%2387c305" /></a>
      <a href="https://anaconda.org/bioconda/giatools"><img src="https://img.shields.io/conda/v/bioconda/giatools.svg?label=Version" /></a>
      <a href="https://anaconda.org/bioconda/giatools"><img src="https://img.shields.io/conda/dn/bioconda/giatools.svg?label=Downloads" /></a>
    </h1>
  </div>

Galaxy Image Analysis: https://github.com/BMCV/galaxy-image-analysis

Use ``python -m unittest`` in the root directory of the repository to run the test suite.

Use ``coverage run -m unittest && coverage html`` to generate a coverage report.

Use ``cd docs && PYTHONPATH=".." sphinx-build -b doctest . _build`` to run doctest.

Use ``cd docs && PYTHONPATH=".." sphinx-build -b html . _build`` to build the docs.
