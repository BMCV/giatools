import warnings

import skimage.io

from ...typing import (
    Any,
    Dict,
    NDArray,
    Union,
)
from ...util import silent
from ..backend import (
    IncompatibleDataError,
    Reader,
    UnsupportedFileError,
    Writer,
)


class SKImageReader(Reader):

    def open(self, *args, **kwargs) -> Any:
        return (args, kwargs)  # deferred loading

    def get_num_images(self) -> int:
        return 1

    def select_image(self, position: int) -> Any:
        image = _skimage_io_imread(*self.file[0], **self.file[1])
        if image.ndim not in (2, 3):
            raise UnsupportedFileError(
                f'Image has unsupported dimension: {image.ndim}',
                filepath=self.file[0][0],
            )
        return image

    def get_axes(self, image: Any) -> str:
        if image.ndim == 2:
            return 'YX'
        else:
            return 'YXC'

    def get_image_data(self, image: Any) -> NDArray:
        return image

    def get_image_metadata(self, image: Any) -> Dict[str, Any]:
        return dict()


class SKImageWriter(Writer):

    supported_extensions = (
        'png',
        'jpg',
        'jpeg',
    )

    def write(self, im_arr: NDArray, filepath: str, metadata: dict, **kwargs):
        suffix = filepath.split('.')[-1].lower()

        # Validate that the image data is compatible with the file format
        error = None
        if suffix == 'png':
            error = self._validate_png(im_arr, metadata)
        if suffix in ('jpg', 'jpeg'):
            error = self._validate_jpg(im_arr, metadata)
        if error:
            raise IncompatibleDataError(error, filepath=filepath)

        # Write the image using skimage
        #
        # The plugin infrastructure in `skimage.io` is deprecated since version 0.25 and will be removed in 0.27 (or
        # later). To avoid this warning, please do not pass additional keyword arguments for plugins (`**plugin_args`).
        #
        # TODO: Instead, use `imageio` or other I/O packages directly.
        #
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            skimage.io.imsave(filepath, im_arr.squeeze(), check_contrast=False, **kwargs)

    def _validate_png(self, im_arr: NDArray, metadata: dict) -> Union[str, None]:
        if not (
            metadata['axes'] == 'YX' or (metadata['axes'] == 'YXC' and im_arr.ndim in (1, 3, 4))
        ):
            return 'PNG files only support single-channel, RGB, and RGBA images (YX or YXC axes layout).'

    def _validate_jpg(self, im_arr: NDArray, metadata: dict):
        if not (
            metadata['axes'] == 'YXC' and im_arr.ndim == 3
        ):
            return 'JPEG files only support RGB images (YXC axes layout).'


@silent
def _skimage_io_imread(*args, **kwargs) -> NDArray:
    """
    Wrapper for skimage.io.imread that suppresses non-fatal errors on stdout and stderr.

    When using ``skimage.io.imread`` to read an image file, sometimes errors can be reported albeit the image file will
    be read successfully. In those cases, Galaxy might detect the errors on stdout or stderr, and assume that the tool
    has failed: https://docs.galaxyproject.org/en/latest/dev/schema.html#error-detection To prevent this, this wrapper
    around ``skimage.io.imread`` will mute all non-fatal errors.
    """
    return skimage.io.imread(*args, **kwargs)
