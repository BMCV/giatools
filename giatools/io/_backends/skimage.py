import warnings

import skimage.io

from ... import (
    metadata as _metadata,
    typing as _typing,
    util as _util,
)
from ..backend import (
    IncompatibleDataError,
    Reader,
    UnsupportedFileError,
    Writer,
)

# https://gist.github.com/leommoore/f9e57ba2aa4bf197ebc5
supported_magic_numbers = (
    b'\x89\x50\x4e\x47',  # PNG
    b'\xff\xd8\xff',      # JPEG
    b'\x4d\x4d\x00\x2a',  # TIFF (big-endian)
    b'\x49\x49\x2a\x00',  # TIFF (little-endian)
)


def _can_read_file(self, filepath: str) -> bool:
    max_prefix_length = max(len(magic) for magic in supported_magic_numbers)
    with open(filepath, 'rb') as f:
        prefix = f.read(max_prefix_length)
        return any(prefix.startswith(magic) for magic in supported_magic_numbers)


class SKImageReader(Reader):

    unsupported_file_errors = (
        OSError,  # raised by _skimage_io_imread
    )

    def open(self, filepath: str, *args, **kwargs) -> _typing.Any:
        if _can_read_file(self, filepath):
            return (filepath, args, kwargs)  # deferred loading
        else:
            raise UnsupportedFileError(filepath, 'File format not supported by this backend.')

    @property
    def filepath(self) -> str:
        return self.file[0]

    def get_num_images(self) -> int:
        return 1

    def select_image(self, position: int) -> _typing.Any:
        image = _skimage_io_imread(self.filepath, *self.file[1], **self.file[2])
        if image.ndim not in (2, 3):
            raise UnsupportedFileError(self.filepath, f'Image has unsupported dimension: {image.ndim}')
        return image

    def get_axes(self, image: _typing.Any) -> str:
        if image.ndim == 2:
            return 'YX'
        else:
            return 'YXC'

    def get_image_data(self, image: _typing.Any) -> _typing.NDArray:
        return image

    def get_image_metadata(self, image: _typing.Any) -> _typing.Dict[str, _typing.Any]:
        return dict()


class SKImageWriter(Writer):

    supported_extensions = (
        'png',
        'jpg',
        'jpeg',
    )

    def write(self, data: _typing.NDArray, filepath: str, axes: str, metadata: _metadata.Metadata, **kwargs):
        suffix = filepath.split('.')[-1].lower()

        # Validate that the image data is compatible with the file format
        error = None
        if suffix == 'png':
            error = self._validate_png(data, axes)
        if suffix in ('jpg', 'jpeg'):
            error = self._validate_jpg(data, axes)
        if error:
            raise IncompatibleDataError(filepath, error)

        # Write the image using skimage
        #
        # The plugin infrastructure in `skimage.io` is deprecated since version 0.25 and will be removed in 0.27 (or
        # later). To avoid this warning, please do not pass additional keyword arguments for plugins (`**plugin_args`).
        #
        # TODO: Instead, use `imageio` or other I/O packages directly.
        #
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            skimage.io.imsave(filepath, data.squeeze(), check_contrast=False, **kwargs)

    def _validate_png(self, im_arr: _typing.NDArray, axes: str) -> _typing.Union[str, None]:
        if not (
            (axes == 'YX' and im_arr.ndim == 2) or (axes == 'YXC' and im_arr.ndim in (1, 3, 4))
        ):
            return 'PNG files only support single-channel, RGB, and RGBA images (YX or YXC axes layout).'

    def _validate_jpg(self, im_arr: _typing.NDArray, axes: str):
        if not (
            axes == 'YXC' and im_arr.ndim == 3 and im_arr.shape[2] == 3
        ):
            return 'JPEG files only support RGB images (YXC axes layout).'


@_util.silent
def _skimage_io_imread(*args, **kwargs) -> _typing.NDArray:
    """
    Wrapper for skimage.io.imread that suppresses non-fatal errors on stdout and stderr.

    When using ``skimage.io.imread`` to read an image file, sometimes errors can be reported albeit the image file will
    be read successfully. In those cases, Galaxy might detect the errors on stdout or stderr, and assume that the tool
    has failed: https://docs.galaxyproject.org/en/latest/dev/schema.html#error-detection To prevent this, this wrapper
    around ``skimage.io.imread`` will mute all non-fatal errors.

    Raises:
        OSError: If the image file cannot be read.
    """
    return skimage.io.imread(*args, **kwargs)
