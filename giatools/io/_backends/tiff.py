import json as _json
from xml.etree import ElementTree as _ElementTree

import attrs as _attrs
import tifffile as _tifffile

from ... import (
    metadata as _metadata,
    typing as _T,
)
from .. import backend as _backend


class TiffReader(_backend.Reader):

    unsupported_file_errors = (
        _tifffile.TiffFileError,
        IsADirectoryError,
    )

    def open(self, filepath: str, *args, **kwargs) -> _T.Any:
        try:
            return _tifffile.TiffFile(filepath, *args, **kwargs)
        except TypeError:  # this is too generic to be added to `unsupported_file_errors`
            raise _backend.UnsupportedFileError(
                filepath,
                f'This backend does not accept the given arguments: args={args}, kwargs={kwargs}',
            )

    def get_num_images(self) -> int:
        return len(self.file.series)

    def select_image(self, position: int) -> _T.Any:
        return self.file.series[position]

    def get_axes(self, image: _T.Any) -> str:
        return image.axes.upper()

    def get_image_data(self, image: _T.Any) -> _T.NDArray:
        return image.asarray()

    def get_image_metadata(self, image: _T.Any) -> _T.Dict[str, _T.Any]:
        return _get_tiff_metadata(self.file, image)


class TiffWriter(_backend.Writer):

    supported_extensions = (
        'tiff',
        'tif',
    )

    def write(
        self,
        data: _T.NDArray,
        filepath: str,
        axes: str,
        metadata: _metadata.Metadata,
        **kwargs: _T.Any,
    ):
        metadata_dict =  _attrs.asdict(metadata, filter=lambda attr, value: value is not None)
        metadata_dict['axes'] = axes

        # Update the metadata structure to what `tifffile` expects
        kwargs = dict(metadata=metadata_dict) | dict(kwargs)
        if 'resolution' in metadata_dict:
            kwargs['resolution'] = metadata_dict.pop('resolution')
        if 'z_spacing' in metadata_dict:
            metadata_dict['spacing'] = metadata_dict.pop('z_spacing')

        # Write the image using tifffile
        _tifffile.imwrite(filepath, data, **kwargs)


def _get_tiff_metadata(tif: _T.Any, series: _T.Any) -> _T.Dict[str, _T.Any]:
    """
    Extract metadata from a `tifffile.TiffFile` object.
    """

    metadata: _T.Dict[str, _T.Any] = dict()

    # Extract pixel resolution, if available
    page0 = series.pages[0]
    if 'XResolution' in page0.tags and 'YResolution' in page0.tags:
        x_res = page0.tags['XResolution'].value
        y_res = page0.tags['YResolution'].value
        if x_res[1] != 0 and y_res[1] != 0:
            metadata['resolution'] = (
                x_res[0] / x_res[1],  # pixels per unit in X, numerator / denominator
                y_res[0] / y_res[1],  # pixels per unit in Y, numerator / denominator
            )

    # Read `ImageDescription` tag
    if 'ImageDescription' in page0.tags:
        description = page0.tags['ImageDescription'].value
        description_format = _guess_tiff_description_format(description)

        # Parse as JSON (giatools-style)
        if description_format == 'json':
            description_json = _json.loads(description)

            # Extract z-slice spacing, if available
            if (val := description_json.get('spacing', None)) is not None:
                try:
                    metadata['z_spacing'] = float(val)
                except ValueError:
                    pass  # ignore invalid values

            # Extract z-position, if available (this is a custom field written by giatools)
            if (val := description_json.get('z_position', None)) is not None:
                try:
                    metadata['z_position'] = float(val)
                except ValueError:
                    pass  # ignore invalid values

            # Extract unit, if available
            if (val := description_json.get('unit', None)) is not None and isinstance(val, str) and val != 'pixel':
                metadata['unit'] = val

        # Parse as XML (OME-style)
        elif description_format == 'xml':
            ome_xml = _ElementTree.fromstring(description)
            ome_ns = dict(ome='http://www.openmicroscopy.org/Schemas/OME/2016-06')
            ome_pixels = ome_xml.find('.//ome:Pixels', ome_ns)

            # Extract z-slice spacing, if available
            if ome_pixels is not None and (val := ome_pixels.attrib.get('PhysicalSizeZ', None)) is not None:
                try:
                    metadata['z_spacing'] = float(val)
                except ValueError:
                    pass  # ignore invalid values

            # OME-TIFF allows different units for z-, x, and y-axes. This needs to be handled properly in the future.
            # For now, we only read the global unit of the z-axis and ignore the others.

            # Extract unit, if available
            if ome_pixels is not None and 'PhysicalSizeZUnit' in ome_pixels.attrib:
                metadata['unit'] = str(ome_pixels.get('PhysicalSizeZUnit'))

            # We currently do not read the `z_position` here, because OME-TIFF only allows per-plane z-positions.

        # Perform line-by-line parsing (ImageJ-style)
        elif description_format == 'line':
            for line in description.splitlines():

                # Extract z-slice spacing, if available
                if line.startswith('spacing='):
                    try:
                        spacing = float(line.split('=')[1])
                        metadata['z_spacing'] = spacing
                    except ValueError:
                        pass  # Ignore lines where spacing value is not a valid float

                # Extract unit, if available
                if line.startswith('unit='):
                    unit = line.split('=')[1]
                    if unit.startswith('"') and unit.endswith('"'):
                        unit = unit[1:-1]  # remove quotes
                    metadata['unit'] = unit

            # We currently do not read the `z_position` here (not implemented yet)

    # As a fallback, read unit from the dedicated tag, if available
    if 'unit' not in metadata and 'ResolutionUnit' in page0.tags:
        res_unit = page0.tags['ResolutionUnit'].value
        if res_unit == 2:
            metadata['unit'] = 'inch'
        elif res_unit == 3:
            metadata['unit'] = 'cm'

    # Normalize unit representation
    if (unit := metadata.get('unit', None)) is not None:
        if unit == r'\u00B5m':
            metadata['unit'] = 'um'
        elif (normalized_unit := _backend.normalize_unit(unit)) is not None:
            metadata['unit'] = normalized_unit
        else:
            del metadata['unit']  # remove unrecognized unit

    return metadata


def _guess_tiff_description_format(description: str) -> _T.Literal['json', 'xml', 'line']:
    """
    Guess the format of the given TIFF `ImageDescription` string.
    """

    # Try to parse as JSON first
    try:
        _json.loads(description)
        return 'json'
    except _json.JSONDecodeError:
        pass

    # Try to parse as XML next
    try:
        _ElementTree.fromstring(description)
        return 'xml'
    except _ElementTree.ParseError:
        pass

    # Fall back to line-by-line parsing
    return 'line'
