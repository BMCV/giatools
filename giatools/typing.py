#from __future__ import annotations  # properly format TypeAliases in Sphinx

import sys

import numpy as np

if sys.version_info < (3, 11):
    from typing_extensions import *  # noqa: F401, F403
else:
    from typing import *  # noqa: F401, F403

if sys.version_info < (3, 10):
    from typing import Iterator  # noqa: F401, F403
else:
    from collections.abc import Iterator  # noqa: F401, F403

try:
    from dask.array import Array as _DaskArray
    class DaskArray(_DaskArray): ...  # noqa: E701
except ImportError:
    class DaskArray: ...  # noqa: E701

NDArray: TypeAlias = Union[np.ndarray, DaskArray]  # noqa: F405
#NDArray.__doc__ = 'Type alias for NumPy arrays and Dask arrays.'
