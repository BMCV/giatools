import os

from ..typing import (
    Any,
    Dict,
    NDArray,
    Optional,
    Self,
    Tuple,
    Type,
)


class UnsupportedFileError(Exception):
    """
    Raised when a file cannot be read or written.
    """
    def __init__(self, filepath: str, *args, **kwargs):
        self.filepath = filepath
        super().__init__(*args, **kwargs)


class CorruptFileError(Exception):
    """
    Raised when a file is corrupted (or follows an unexpected internal format flavor) and cannot be read.
    """
    def __init__(self, filepath: str, *args, **kwargs):
        self.filepath = filepath
        super().__init__(*args, **kwargs)


class IncompatibleDataError(Exception):
    """
    Raised when a file cannot be written because the data or metadata is incompatible with the file format.
    """
    def __init__(self, filepath: str, *args, **kwargs):
        self.filepath = filepath
        super().__init__(*args, **kwargs)


class Reader:

    unsupported_file_errors = tuple()

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.file = None

    def __enter__(self) -> Self:
        self.file = self.open(*self._args, **self._kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self.file, 'close'):
            self.file.close()
        self.file = None

    def open(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def get_num_images(self) -> int:
        raise NotImplementedError()

    def select_image(self, position: int) -> Any:
        raise NotImplementedError()

    def get_axes(self, image: Any) -> str:
        raise NotImplementedError()

    def get_image_data(self, image: Any) -> NDArray:
        raise NotImplementedError()

    def get_image_metadata(self, image: Any) -> Dict[str, Any]:
        raise NotImplementedError()


class Writer:

    supported_extensions = tuple()

    def write(self, im_arr: NDArray, filepath: str, metadata: dict, **kwargs):
        raise NotImplementedError()


class Backend:

    def __init__(self, name: str, reader_class: Type[Reader], writer_class: Optional[Type[Writer]] = None):
        self.name = name
        self.reader_class = reader_class
        self.writer_class = writer_class

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'<{self.name} Backend>'

    def peek_num_images_in_file(self, filepath: str, *args, **kwargs) -> Optional[int]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'File not found: {filepath}')
        try:
            with self.reader_class(filepath, *args, **kwargs) as reader:
                return reader.get_num_images()
        except tuple(list(self.reader_class.unsupported_file_errors) + [UnsupportedFileError]):
            return None  # Indicate that the file is unsupported

    def read(self, filepath: str, *args, position: int = 0, **kwargs) -> Optional[Tuple[NDArray, str, Dict[str, Any]]]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'File not found: {filepath}')
        try:
            with self.reader_class(filepath, *args, **kwargs) as reader:

                # Handle files with multiple images
                num_images = reader.get_num_images()
                if 0 <= position < num_images:
                    image = reader.select_image(position)
                else:
                    raise IndexError(f'Image {position} is out of range for file with {num_images} images.')
                im_axes = reader.get_axes(image)

                # Verify that the image format is supported
                if 'Y' not in im_axes or 'X' not in im_axes:
                    raise CorruptFileError(filepath, f'Image is missing required X or Y axes (found {im_axes}).')
                if (
                    not (frozenset('YX') <= frozenset(im_axes) <= frozenset('QTZYXCS'))
                    or len(im_axes) != len(frozenset(im_axes))
                ):
                    raise CorruptFileError(filepath, f'Image has unsupported axes: {im_axes}')

                # Treat sample axis "S" as channel axis "C" and fail if both are present
                if 'C' in im_axes and 'S' in im_axes:
                    raise CorruptFileError(
                        filepath,
                        f'Image has both channel and sample axes which is not supported: {im_axes}',
                    )
                im_axes = im_axes.replace('S', 'C')

                # Get the reference to the image data
                im_arr = reader.get_image_data(image)

                # Read the metadata
                metadata = reader.get_image_metadata(image)

                # Return the image data, axes, and metadata
                return im_arr, im_axes, metadata

        except tuple(list(self.reader_class.unsupported_file_errors) + [UnsupportedFileError]):
            return None  # Indicate that the file is unsupported

    def write(self, im_arr: NDArray, filepath: str, metadata: dict, **kwargs):
        writer = self.writer_class()

        # Create a copy of the metadata to avoid modifying the original
        metadata = dict(metadata) if metadata is not None else dict()

        # Validate metadata
        if metadata is None:
            raise ValueError('Metadata must be provided when writing images.')
        im_axes = metadata.get('axes', '')
        if 'Y' not in im_axes or 'X' not in im_axes:
            raise ValueError(f'Image is missing required X or Y axes (found {im_axes}).')
        if (
            not (frozenset('YX') <= frozenset(im_axes) <= frozenset('QTZYXCS'))
            or len(im_axes) != len(frozenset(im_axes))
        ):
            raise ValueError(f'Image has unsupported axes: {im_axes}')

        # Treat sample axis "S" as channel axis "C" and fail if both are present
        if 'C' in im_axes and 'S' in im_axes:
            raise ValueError(
                f'Image has both channel and sample axes which is not supported: {im_axes}',
            )
        im_axes = im_axes.replace('S', 'C')

        # Delegate the writing to the writer class
        writer.write(im_arr, filepath, metadata, **kwargs)


def normalize_unit(unit: str) -> Optional[str]:
    """
    Normalizes a unit string to a standard representation.
    """
    unit = unit.strip().lower()
    if unit in ('nm', 'nanometer', 'nanometers'):
        return 'nm'
    elif unit in ('um', 'micrometer', 'micrometers', 'µm'):
        return 'um'
    elif unit in ('mm', 'millimeter', 'millimeters'):
        return 'mm'
    elif unit in ('cm', 'centimeter', 'centimeters'):
        return 'cm'
    elif unit in ('m', 'meter', 'meters'):
        return 'm'
    elif unit in ('km', 'kilometer', 'kilometers'):
        return 'km'
    elif unit in ('inch', 'inches'):
        return 'inch'
    else:
        return None  # unknown unit
