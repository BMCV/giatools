"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

from .image import (  # noqa: F401
    Image,
    default_normalized_axes,
)
from .version import __version__ as VERSION  # noqa: F401


def require_backend(name: str):
    """
    Ensures that the backend with the given `name` is available.

    Raises:
        ImportError: If the backend is not available.
    """
    from .io import backends

    for backend in backends:
        if backend.name == name:
            break
    else:
        raise ImportError(f'The backend "{name}" is not available.')
