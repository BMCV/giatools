import skimage.io

from ...typing import (
    Any,
    Dict,
    NDArray,
)
from ...util import silent
from ..backend import (
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
    )

    def write(self, im_arr: NDArray, filepath: str, metadata: dict):
        skimage.io.imsave(filepath, im_arr, check_contrast=False)


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
