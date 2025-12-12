from .backend import Backend
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
