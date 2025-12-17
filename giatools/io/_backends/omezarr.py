import ome_zarr.io as _ome_zarr_io
import ome_zarr.reader as _ome_zarr_reader

from ... import (
    metadata as _metadata,
    typing as _T,
)
from .. import backend as _backend


class OMEZarrReader(_backend.Reader):

    def open(self, filepath: str, *args, **kwargs) -> _T.Any:
        try:
            omezarr_store = _ome_zarr_io.parse_url(filepath, *args, **kwargs)
        except TypeError:  # this is too generic to be added to `unsupported_file_errors`
            raise _backend.UnsupportedFileError(
                filepath,
                f'This backend does not accept the given arguments: args={args}, kwargs={kwargs}',
            )
        if omezarr_store is None:
            raise _backend.UnsupportedFileError(filepath=filepath)
        else:
            omezarr_reader = _ome_zarr_reader.Reader(omezarr_store)
            return list(omezarr_reader())

    def get_num_images(self) -> int:
        return len(self.file)

    def select_image(self, position: int) -> _T.Any:
        return self.file[position]

    def get_axes(self, image: _T.Any) -> str:
        return _get_omezarr_axes(image)

    def get_image_data(self, image: _T.Any) -> _T.NDArray:
        return image.data[0]  # top-level of the pyramid (dask array)

    def get_image_metadata(self, image: _T.Any) -> _metadata.Metadata:
        return _get_omezarr_metadata(image)


def _get_omezarr_axes(omezarr_node: _ome_zarr_reader.Node) -> str:
    """
    Extract axes string from an `ome_zarr.reader.Node` object.
    """
    if 'axes' not in omezarr_node.metadata:
        raise _backend.CorruptFileError('OME-Zarr node is missing axes information.')
    return ''.join(axis['name'].upper() for axis in omezarr_node.metadata['axes'])


def _get_omezarr_metadata(omezarr_node: _ome_zarr_reader.Node) -> _metadata.Metadata:
    """
    Extract metadata from an `ome_zarr.reader.Node` object.
    """
    axes = _get_omezarr_axes(omezarr_node)
    metadata = _metadata.Metadata()

    # Extract the `unit`, if it is constant across all axes
    units = frozenset((axis['unit'] for axis in omezarr_node.metadata.get('axes', [])))
    if (
        len(units) == 1 and (unit := next(iter(units))) and
        (normalized_unit := _backend.normalize_unit(unit)) is not None
    ):
        metadata.unit = normalized_unit

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
                    metadata.pixel_size = (
                        float(scales[axes.index('X')]),
                        float(scales[axes.index('Y')]),
                    )
                    if 'Z' in axes:
                        metadata.z_spacing = float(scales[axes.index('Z')])

                # Only consider the first `scale` transformation
                break

    # Ignore if the metadata is malformed
    except (KeyError, IndexError, TypeError):
        pass

    return metadata
