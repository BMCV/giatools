import json
import os
import pathlib
import tempfile
import unittest

import attrs
import numpy as np
import scipy.ndimage as ndi
import skimage.io
import tifffile

import giatools.io
import giatools.metadata
from giatools.typing import (
    Literal,
    Optional,
    PathLike,
    Tuple,
    Type,
    Union,
)

from .. import tools


class imwrite(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def _read_image(self, filepath: str) -> Tuple[np.ndarray, str]:
        """
        Read an image file and return the image data and axes.

        This is a helper function to read image files for validation.
        """
        try:
            with tifffile.TiffFile(filepath) as tif:
                axes = tif.series[0].axes
                data = tif.asarray()
        except tifffile.TiffFileError:
            data = skimage.io.imread(filepath, as_gray=False)
            axes = 'YXC'
        return data, axes

    def _test(
        self,
        data_shape: Tuple,
        axes: str,
        dtype: np.dtype,
        metadata: Optional[giatools.metadata.Metadata] = None,
        *,
        filepath_type: Type[PathLike] = str,
        sigma: float = 0,
        ext: str,
        backend: str = 'auto',
        validate_axes: bool = True,
        validate_metadata: Union[bool, Literal['auto']] = 'auto',
        rms_tol: Optional[float] = None,
        **kwargs,
    ):
        # Create random image data
        data = np.random.rand(*data_shape)
        if sigma > 0:
            data = ndi.gaussian_filter(data, sigma=sigma)
        if not np.issubdtype(dtype, np.floating):
            data = (data * np.iinfo(dtype).max).astype(dtype)

        # Write the image to a temporary file
        filepath = os.path.join(self.tempdir.name, f'test.{ext}')
        metadata = giatools.metadata.Metadata() if metadata is None else metadata
        metadata_copy = attrs.asdict(metadata)
        giatools.io.imwrite(data, filepath_type(filepath), backend=backend, axes=axes, metadata=metadata, **kwargs)

        # Validate immutability of metadata
        tools.validate_metadata(self, metadata, **metadata_copy)

        # Read back the image data and the axes, and validate, if applicable
        data1, axes1 = self._read_image(filepath)
        if rms_tol is None:
            np.testing.assert_array_equal(data1, data)
        else:
            self.assertLessEqual(np.sqrt((data1 - data) ** 2).mean(), rms_tol)
        if validate_axes:
            self.assertEqual(axes1, axes)

        # Validate the metadata (written as JSON), if applicable
        if validate_metadata is True or (validate_metadata == 'auto' and ext in ('tif', 'tiff')):
            with tifffile.TiffFile(filepath) as im_file:
                page0 = im_file.series[0].pages[0]
                description = json.loads(page0.tags['ImageDescription'].value)
                x_res = page0.tags['XResolution'].value
                y_res = page0.tags['YResolution'].value

            if metadata.resolution is not None:
                np.testing.assert_allclose(
                    (
                        x_res[0] / x_res[1],
                        y_res[0] / y_res[1],
                    ),
                    metadata.resolution,
                )
            if metadata.z_spacing is not None:
                self.assertEqual(float(description['spacing']), metadata.z_spacing)
            if metadata.z_position is not None:
                self.assertEqual(float(description['z_position']), metadata.z_position)
            if metadata.unit is not None:
                self.assertEqual(description['unit'], metadata.unit)

    def test__unsupported_backend(self):
        with self.assertRaises(ValueError):
            self._test(
                data_shape=(10, 10, 2),
                axes='YXC',
                dtype=np.float32,
                ext='tiff',
                backend='unsupported_backend',
            )

    def test__unsupported_file_error(self):
        with self.assertRaises(giatools.io.UnsupportedFileError):
            self._test(
                data_shape=(10, 10, 2),
                axes='YXC',
                dtype=np.float32,
                ext='unsupported_extension',
                backend='auto',
            )

    def test__incompatible_data_error(self):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self._test(
                data_shape=(10, 10, 2, 2),
                axes='YXCZ',  # Invalid axes for PNG
                dtype=np.uint8,
                ext='png',
                backend='auto',
            )

    def test__float32__tifffile__tif(self):
        with self.assertWarns(DeprecationWarning):
            self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tif', backend='tifffile')

    def test__float32__tifffile__tiff(self):
        for filetype_path in (str, pathlib.Path):
            with self.subTest(filepath_type=filetype_path):
                self._test(
                    data_shape=(10, 10, 5, 2),
                    axes='YXZC',
                    dtype=np.float32,
                    ext='tiff',
                    backend='tifffile',
                    filepath_type=filetype_path,
                )

    def test__float32__tifffile__tiff__metadata(self):
        self._test(
            data_shape=(10, 10, 5),
            axes='YXZ',
            dtype=np.float32,
            ext='tiff',
            backend='tifffile',
            metadata=giatools.metadata.Metadata(
                resolution=(0.3, 0.4),
                z_spacing=0.5,
                z_position=0.8,
                unit='um',
            ),
        )

    def test__float32__skimage__tif(self):
        with self.assertWarns(DeprecationWarning):
            self._test(
                data_shape=(10, 10, 5, 2),
                axes='YXZC',
                dtype=np.float32,
                ext='tif',
                backend='skimage',
                validate_axes=False,
            )

    def test__float32__skimage__tiff(self):
        self._test(
            data_shape=(10, 10, 5, 2),
            axes='YXZC',
            dtype=np.float32,
            ext='tiff',
            backend='skimage',
            validate_axes=False,
        )

    def test__float32__auto__tif(self):
        with self.assertWarns(DeprecationWarning):
            self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tif', backend='auto')

    def test__float32__auto__tiff(self):
        self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tiff', backend='auto')

    def test__uint8__auto__png(self):
        self._test(data_shape=(10, 10, 2), axes='YXC', dtype=np.uint8, ext='png', backend='auto')

    def test__uint8__auto__jpg(self):
        for ext in ('jpg', 'jpeg'):
            with self.subTest(ext=ext):
                self._test(
                    data_shape=(100, 150, 3),
                    axes='YXC',
                    dtype=np.uint8,
                    ext=ext,
                    backend='auto',
                    sigma=3,
                    rms_tol=0.1,
                    quality=100,
                )
