from .backend import (  # noqa: F401
    Backend,
    UnsupportedFileError,
)
from .omezarr import OMEZarrReader
from .tiff import (
    TiffReader,
    TiffWriter,
)
from .skimage import (
    SKImageReader,
    SKImageWriter,
)

backends = [
    Backend('tifffile', TiffReader, TiffWriter),
    Backend('omezarr', OMEZarrReader),
    Backend('skimage', SKImageReader, SKImageWriter),
]
"""
Defines the supported backends for reading and writing image files.

For reading, the backends are tried in succession until one is successful. For writing, the appropriate backend is
selected based on the file extension.
"""
