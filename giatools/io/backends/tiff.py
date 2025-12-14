import json
from xml.etree import ElementTree

import tifffile

from ...typing import (
    Any,
    Dict,
    Literal,
    NDArray,
)
from ..backend import (
    Reader,
    Writer,
)


class TiffReader(Reader):

    unsupported_file_errors = (
        tifffile.TiffFileError,
        IsADirectoryError,
    )

    def open(self, *args, **kwargs) -> Any:
        return tifffile.TiffFile(*args, **kwargs)

    def get_num_images(self) -> int:
        return len(self.file.series)

    def select_image(self, position: int) -> Any:
        return self.file.series[position]

    def get_axes(self, image: Any) -> str:
        return image.axes.upper()

    def get_image_data(self, image: Any) -> NDArray:
        return image.asarray()

    def get_image_metadata(self, image: Any) -> Dict[str, Any]:
        return _get_tiff_metadata(self.file, image)


class TiffWriter(Writer):

    supported_extensions = (
        'tiff',
        'tif',
    )

    def write(self, im_arr: NDArray, filepath: str, metadata: dict, **kwargs):

        # Update the metadata structure to what `tifffile` expects
        kwargs = dict(kwargs)
        kwargs['metadata'] = metadata
        if 'resolution' in metadata:
            kwargs['resolution'] = metadata.pop('resolution')
        if 'z_spacing' in metadata:
            metadata['spacing'] = metadata.pop('z_spacing')

        # Write the image using tifffile
        tifffile.imwrite(filepath, im_arr, **kwargs)


def _get_tiff_metadata(tif: Any, series: Any) -> Dict[str, Any]:
    """
    Extract metadata from a `tifffile.TiffFile` object.
    """

    metadata: Dict[str, Any] = dict()

    # Extract pixel resolution, if available
    page0 = series.pages[0]
    if 'XResolution' in page0.tags and 'YResolution' in page0.tags:
        x_res = page0.tags['XResolution'].value
        y_res = page0.tags['YResolution'].value
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
            description_json = json.loads(description)

            # Extract z-slice spacing, if available
            if 'spacing' in description_json:
                metadata['z_spacing'] = float(description_json['spacing'])

            # Extract z-position, if available (this is a custom field written by giatools)
            if 'z_position' in description_json:
                metadata['z_position'] = float(description_json['z_position'])

            # Extract unit, if available
            if 'unit' in description_json:
                metadata['unit'] = str(description_json['unit'])

        # Parse as XML (OME-style)
        elif description_format == 'xml':
            ome_xml = ElementTree.fromstring(description)
            ome_ns = dict(ome='http://www.openmicroscopy.org/Schemas/OME/2016-06')
            ome_pixels = ome_xml.find('.//ome:Pixels', ome_ns)

            # Extract z-slice spacing, if available
            if ome_pixels is not None and 'PhysicalSizeZ' in ome_pixels.attrib:
                metadata['z_spacing'] = float(ome_pixels.get('PhysicalSizeZ'))

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
                    if unit != 'pixel':
                        metadata['unit'] = unit

    # As a fallback, read unit from the dedicated tag, if available
    if 'unit' not in metadata and 'ResolutionUnit' in page0.tags:
        res_unit = page0.tags['ResolutionUnit'].value
        if res_unit == 2:
            metadata['unit'] = 'inch'
        elif res_unit == 3:
            metadata['unit'] = 'cm'

    # Normalize unit representation
    if metadata.get('unit', None) in (r'\u00B5m', 'µm'):
        metadata['unit'] = 'um'

    return metadata


def _guess_tiff_description_format(description: str) -> Literal['json', 'xml', 'line']:
    """
    Guess the format of the given TIFF `ImageDescription` string.
    """

    # Try to parse as JSON first
    try:
        json.loads(description)
        return 'json'
    except json.JSONDecodeError:
        pass

    # Try to parse as XML next
    try:
        ElementTree.fromstring(description)
        return 'xml'
    except ElementTree.ParseError:
        pass

    # Fall back to line-by-line parsing
    return 'line'
