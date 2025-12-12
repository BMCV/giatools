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

normalized_unit_representations = {

    # nanometer representations
    'nm': 'nm',
    'nanometer': 'nm',

    # micrometer representations
    'um': 'um',
    'micrometer': 'um',

    # millimeter representations
    'mm': 'mm',
    'millimeter': 'mm',

}


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
        return _get_omezarr_axes(image)

    def get_image_data(self, image: Any) -> np.ndarray:
        return image.data[0]  # top-level of the pyramid (dask array)

    def get_image_metadata(self, image: Any) -> Dict[str, Any]:
        return _get_omezarr_metadata(image)


def _get_omezarr_axes(omezarr_node: ome_zarr.reader.Node) -> str:
    """
    Extract axes string from an `ome_zarr.reader.Node` object.
    """
    assert 'axes' in omezarr_node.metadata, 'OME-Zarr node is missing axes information.'
    return ''.join(axis['name'].upper() for axis in omezarr_node.metadata['axes'])


def _get_omezarr_metadata(omezarr_node: ome_zarr.reader.Node) -> Dict[str, Any]:
    """
    Extract metadata from an `ome_zarr.reader.Node` object.
    """
    axes = _get_omezarr_axes(omezarr_node)
    metadata: Dict[str, Any] = dict()

    # Extract the `unit`, if it is constant across all axes
    units = frozenset((axis['unit'] for axis in omezarr_node.metadata.get('axes', [])))
    if len(units) == 1 and (unit := normalized_unit_representations.get(next(iter(units))), ''):
        metadata['unit'] = unit

    # Extract the pixel/voxel sizes
    try:
        transformations = (
            omezarr_node.metadata['coordinateTransformations'][0]  # sizes for the top-level of the pyramid
        )
        for transformation in transformations:
            if transformation['type'] == 'scale':
                scales = transformation['scale']

                # Only include spacing information if it matches the number of axes
                if len(scales) == len(axes):
                    metadata['resolution'] = (
                        1 / scales[axes.index('X')],
                        1 / scales[axes.index('Y')],
                    )
                    if 'Z' in axes:
                        metadata['z_spacing'] = scales[axes.index('Z')]

                # Only consider the first `scale` transformation
                break

    # Ignore if the metadata is malformed
    except (KeyError, IndexError):
        pass

    return metadata
