import numpy as np
import skimage.io

from ...typing import (
    Any,
    Dict,
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
            raise UnsupportedFileError(f'Image has unsupported dimension: {image.ndim}')
        return image

    def get_axes(self, image: Any) -> str:
        if image.ndim == 2:
            return 'YX'
        else:
            return 'YXC'

    def get_image_data(self, image: Any) -> np.ndarray:
        return image

    def get_image_metadata(self, image: Any) -> Dict[str, Any]:
        return dict()


class SKImageWriter(Writer):

    supported_extensions = (
        'png',
        'jpg',
    )

    def write(self, im_arr: np.ndarray, filepath: str, metadata: dict):
        skimage.io.imsave(filepath, im_arr, check_contrast=False)


@silent
def _skimage_io_imread(*args, **kwargs) -> np.ndarray:
    return skimage.io.imread(*args, **kwargs)
