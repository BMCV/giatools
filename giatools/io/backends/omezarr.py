import numpy as np
import ome_zarr.io
import ome_zarr.reader

from ...typing import (
    Any,
    Dict,
)
from ..backend import (
    Reader,
    UnsupportedFileError,
)


class OMEZarrReader(Reader):

    def open(self, *args, **kwargs) -> Any:
        omezzarr_store = ome_zarr.io.parse_url(*args, **kwargs)
        if omezzarr_store is None:
            raise UnsupportedFileError()
        else:
            omezarr_reader = ome_zarr.reader.Reader(omezzarr_store)
            return list(omezarr_reader())

    def get_num_images(self) -> int:
        return len(self.file)

    def select_image(self, position: int) -> Any:
        return self.file[position]

    def get_axes(self, image: Any) -> str:
        assert 'axes' in image.metadata, 'OME-Zarr node is missing axes information.'
        return ''.join(axis['name'].upper() for axis in image.metadata['axes'])

    def get_image_data(self, image: Any) -> np.ndarray:
        return image.data[0]  # top-level of the pyramid (dask array)

    def get_image_metadata(self, image: Any) -> Dict[str, Any]:
        # TODO: Read the metadata
        return dict()  # _get_omezarr_metadata(omezarr_node)
