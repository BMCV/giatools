import ome_zarr.io
import ome_zarr.reader

from ... import typing as _typing
from ..backend import (
    CorruptFileError,
    Reader,
    UnsupportedFileError,
    normalize_unit,
)


class OMEZarrReader(Reader):

    def open(self, filepath: str, *args, **kwargs) -> _typing.Any:
        try:
            omezarr_store = ome_zarr.io.parse_url(filepath, *args, **kwargs)
        except TypeError:  # this is too generic to be added to `unsupported_file_errors`
            raise UnsupportedFileError(
                filepath,
                f'This backend does not accept the given arguments: args={args}, kwargs={kwargs}',
            )
        if omezarr_store is None:
            raise UnsupportedFileError(filepath=filepath)
        else:
            omezarr_reader = ome_zarr.reader.Reader(omezarr_store)
            return list(omezarr_reader())

    def get_num_images(self) -> int:
        return len(self.file)

    def select_image(self, position: int) -> _typing.Any:
        return self.file[position]

    def get_axes(self, image: _typing.Any) -> str:
        return _get_omezarr_axes(image)

    def get_image_data(self, image: _typing.Any) -> _typing.NDArray:
        return image.data[0]  # top-level of the pyramid (dask array)

    def get_image_metadata(self, image: _typing.Any) -> _typing.Dict[str, _typing.Any]:
        return _get_omezarr_metadata(image)


def _get_omezarr_axes(omezarr_node: ome_zarr.reader.Node) -> str:
    """
    Extract axes string from an `ome_zarr.reader.Node` object.
    """
    if 'axes' not in omezarr_node.metadata:
        raise CorruptFileError('OME-Zarr node is missing axes information.')
    return ''.join(axis['name'].upper() for axis in omezarr_node.metadata['axes'])


def _get_omezarr_metadata(omezarr_node: ome_zarr.reader.Node) -> _typing.Dict[str, _typing.Any]:
    """
    Extract metadata from an `ome_zarr.reader.Node` object.
    """
    axes = _get_omezarr_axes(omezarr_node)
    metadata: _typing.Dict[str, _typing.Any] = dict()

    # Extract the `unit`, if it is constant across all axes
    units = frozenset((axis['unit'] for axis in omezarr_node.metadata.get('axes', [])))
    if len(units) == 1 and (unit := next(iter(units))) and (normalized_unit := normalize_unit(unit)) is not None:
        metadata['unit'] = normalized_unit

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
    except (KeyError, IndexError, TypeError):
        pass

    return metadata
