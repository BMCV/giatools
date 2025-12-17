import pathlib as _pathlib
import sys as _sys

import numpy as _np

if _sys.version_info < (3, 11):
    from typing_extensions import *  # noqa: F401, F403
else:
    from typing import *  # noqa: F401, F403

if _sys.version_info < (3, 10):
    from typing import Iterator  # noqa: F401, F403
else:
    from collections.abc import Iterator  # noqa: F401, F403

try:
    from dask.array import Array as _DaskArray
    class DaskArray(_DaskArray): ...  # noqa: E701
except ImportError:
    class DaskArray: ...  # noqa: E701

NDArray: TypeAlias = Union[_np.ndarray, DaskArray]  # noqa: F405

PathLike: TypeAlias = Union[str, _pathlib.Path]  # noqa: F405
