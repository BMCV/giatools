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

if TYPE_CHECKING:  # noqa: F405
    from dask.array import Array as DaskArray
    NDArray: TypeAlias = Union[np.ndarray, DaskArray]  # noqa: F405
else:
    NDArray = np.ndarray

x: NDArray
