"""
Module tests for the `giatools.image` module.
"""

import unittest
import unittest.mock

import numpy as np

import giatools.image
from giatools.typing import (
    Optional,
    Tuple,
)

from ..tools import (
    minimum_python_version,
    permute_axes,
)


class ImageTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.io_imwrite = unittest.mock.patch(
            'giatools.io.imwrite'
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)

        self.img1_data = np.random.randint(0, 255, (1, 2, 26, 32, 3), dtype=np.uint8)
        self.img1_axes = 'TZYXC'
        self.img1_original_axes = 'ZXYC'
        self.img1 = giatools.image.Image(
            data=self.img1_data.copy(),
            axes=self.img1_axes,
            original_axes=self.img1_original_axes,
        )

        self.img2_data = np.random.randint(0, 255, (1, 1, 32, 26, 1), dtype=np.uint8)
        self.img2_axes = 'ZTYXC'
        self.img2_original_axes = 'YXC'
        self.img2 = giatools.image.Image(
            data=self.img2_data.copy(),
            axes=self.img2_axes,
            original_axes=self.img2_original_axes,
        )


class Image__write(ImageTestCase):

    def test__immutability(self):
        self.img1.write('test_output.tiff')
        np.testing.assert_array_equal(self.img1.data, self.img1_data)


class Image__reorder_axes_like(ImageTestCase):

    def test(self):
        img_reordered = self.img1.reorder_axes_like('ZTCYX')
        self.assertEqual(img_reordered.axes, 'ZTCYX')
        self.assertEqual(img_reordered.data.shape, (2, 1, 3, 26, 32))
        self.assertEqual(img_reordered.original_axes, self.img1_original_axes)

    def test__immutability(self):
        """
        Verify that the original image is not modified.
        """
        self.img1.reorder_axes_like('ZTCYX')
        np.testing.assert_array_equal(self.img1.data, self.img1_data)
        self.assertEqual(self.img1.axes, self.img1_axes)
        self.assertEqual(self.img1.original_axes, self.img1_original_axes)


class Image__squeeze(ImageTestCase):

    def test__immutability(self):
        """
        Verify that the original image is not modified.
        """
        self.img1.squeeze()
        np.testing.assert_array_equal(self.img1.data, self.img1_data)
        self.assertEqual(self.img1.axes, self.img1_axes)
        self.assertEqual(self.img1.original_axes, self.img1_original_axes)


class Image__squeeze_like(ImageTestCase):

    def test__identity(self):
        img_squeezed = self.img1.squeeze_like('ZCYX')
        self.assertEqual(img_squeezed.original_axes, self.img1_original_axes)
        self.assertTrue(np.shares_memory(img_squeezed.data, self.img1.data))
        self.assertIs(img_squeezed.metadata, self.img1.metadata)

    def test__no_squeeze(self):
        img_squeezed = self.img1.squeeze_like('ZTCYX')
        self.assertEqual(img_squeezed.axes, 'ZTCYX')
        self.assertEqual(img_squeezed.data.shape, (2, 1, 3, 26, 32))
        self.assertEqual(img_squeezed.original_axes, self.img1_original_axes)

    def test__squeeze_1(self):
        img_squeezed = self.img1.squeeze_like('ZCYX')
        self.assertEqual(img_squeezed.axes, 'ZCYX')
        self.assertEqual(img_squeezed.data.shape, (2, 3, 26, 32))
        self.assertEqual(img_squeezed.original_axes, self.img1_original_axes)

    def test__squeeze_2(self):
        img_squeezed = self.img2.squeeze_like('XYC')
        self.assertEqual(img_squeezed.axes, 'XYC')
        self.assertEqual(img_squeezed.data.shape, (26, 32, 1))
        self.assertEqual(img_squeezed.original_axes, self.img2_original_axes)

    def test__squeeze_3(self):
        img_squeezed = self.img2.squeeze_like('XY')
        self.assertEqual(img_squeezed.axes, 'XY')
        self.assertEqual(img_squeezed.data.shape, (26, 32))
        self.assertEqual(img_squeezed.original_axes, self.img2_original_axes)

    def test__immutability(self):
        """
        Verify that the original image is not modified.
        """
        self.img1.squeeze_like('ZCYX')
        np.testing.assert_array_equal(self.img1.data, self.img1_data)
        self.assertEqual(self.img1.axes, self.img1_axes)
        self.assertEqual(self.img1.original_axes, self.img1_original_axes)


class Image__normalize_axes_like(ImageTestCase):

    def test__identity(self):
        img_normalized = self.img1.normalize_axes_like(self.img1_original_axes)
        self.assertEqual(img_normalized.original_axes, self.img1_original_axes)
        self.assertTrue(np.shares_memory(img_normalized.data, self.img1.data))
        self.assertIs(img_normalized.metadata, self.img1.metadata)

    def test1(self):
        img_normalized = self.img1.normalize_axes_like(self.img1_original_axes)
        self.assertEqual(img_normalized.axes, self.img1_original_axes)
        self.assertEqual(img_normalized.data.shape, (2, 32, 26, 3))
        self.assertEqual(img_normalized.original_axes, self.img1_original_axes)

    def test2(self):
        img_normalized = self.img2.normalize_axes_like(self.img2_original_axes)
        self.assertEqual(img_normalized.axes, self.img2_original_axes)
        self.assertEqual(img_normalized.data.shape, (32, 26, 1))
        self.assertEqual(img_normalized.original_axes, self.img2_original_axes)

    def test__immutability(self):
        """
        Verify that the original image is not modified.
        """
        self.img1.normalize_axes_like(self.img1_original_axes)
        np.testing.assert_array_equal(self.img1.data, self.img1_data)
        self.assertEqual(self.img1.axes, self.img1_axes)
        self.assertEqual(self.img1.original_axes, self.img1_original_axes)


class Image__data(unittest.TestCase):
    """
    Test the data property of the Image class.
    """

    @minimum_python_version(3, 11)
    def test__dask__filtering(self):
        import dask.array as da
        import scipy.ndimage as ndi
        np.random.seed(0)
        np_data = np.random.rand(40, 60)
        img = giatools.Image(data=da.from_array(np_data, chunks=(5, 5)), axes='YX')
        self.assertIsInstance(img.data, da.Array)
        np.testing.assert_almost_equal(ndi.gaussian_filter(img.data, sigma=3).mean(), 0.5, decimal=2)


class Image__iterate_jointly(unittest.TestCase):

    def _create_test_image(self, axes: str, shape: Tuple[int, ...]) -> giatools.image.Image:
        assert len(axes) == len(shape)
        np.random.seed(0)
        data = np.random.randint(0, 255, shape, dtype=np.uint8)
        return giatools.image.Image(data=data, axes=axes)

    def _test(self, axes: str, shape: Tuple[int, ...], joint_axes: str):
        assert set(joint_axes).issubset(set(axes))
        img = self._create_test_image(axes, shape)
        counter = np.zeros(img.data.shape, np.uint32)
        for source_slice, section in img.iterate_jointly(joint_axes):
            self.assertEqual(section.axes, joint_axes)
            counter[source_slice] += 1
            np.testing.assert_array_equal(
                section.reorder_axes_like(section.original_axes).data,
                img.data[source_slice],
            )
        np.testing.assert_array_equal(counter, np.ones(counter.shape, np.uint8))

    @minimum_python_version(3, 11)
    def test__img_yx__iterate_y(self):
        self._test('YX', (10, 11), 'Y')

    @minimum_python_version(3, 11)
    @permute_axes('YX', name='joint_axes')
    def test__img_yx__iterate_yx(self, joint_axes: str):
        self._test('YX', (10, 11), joint_axes)

    @minimum_python_version(3, 11)
    @permute_axes('YX', name='joint_axes')
    def test__img_zyx__iterate_yx(self, joint_axes: str):
        self._test('ZYX', (1, 11, 12), joint_axes)
        self._test('ZYX', (5, 11, 12), joint_axes)

    @minimum_python_version(3, 11)
    @permute_axes('ZYX', name='joint_axes')
    def test__img_zyx__iterate_zyx(self, joint_axes: str):
        self._test('ZYX', (1, 11, 12), joint_axes)
        self._test('ZYX', (5, 11, 12), joint_axes)

    @minimum_python_version(3, 11)
    @permute_axes('ZYX', name='joint_axes')
    def test__img_tzyxc__iterate_zyx(self, joint_axes: str):
        self._test('TZYXC', (1, 1, 11, 12, 3), joint_axes)
        self._test('TZYXC', (1, 5, 11, 12, 3), joint_axes)
        self._test('TZYXC', (5, 1, 11, 12, 3), joint_axes)
        self._test('TZYXC', (5, 5, 11, 12, 3), joint_axes)

    def _test_dask(self, axes: str, shape: Tuple[int, ...], joint_axes: str, chunks: Tuple[int, ...]):
        assert set(joint_axes).issubset(set(axes))
        import dask.array as da
        img = self._create_test_image(axes, shape)
        np_data, img.data = img.data, da.from_array(img.data, chunks=chunks)
        counter = np.zeros(img.data.shape, np.uint32)
        for source_slice, section in img.iterate_jointly(joint_axes):
            counter[source_slice] += 1
            np.testing.assert_array_equal(
                section.reorder_axes_like(section.original_axes).data,
                np_data[source_slice],
            )
        np.testing.assert_array_equal(counter, np.ones(counter.shape, np.uint8))

    @minimum_python_version(3, 11)
    @permute_axes('YX', name='joint_axes')
    def test__dask_array__zyx__iterate__yx(self, joint_axes: str):
        self._test_dask('ZYX', (10, 20, 30), joint_axes, (2, 5, 5))


class Image__astype(ImageTestCase):

    exact_dtype_list = [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ]

    inexact_dtype_list = [
        np.floating,
        np.integer,
        np.signedinteger,
        np.unsignedinteger,
    ]

    def _create_image(
        self,
        dtype: np.dtype,
        shape=(2, 3, 26, 32),
        axes='CYXZ',
        original_axes='QTCYXZ',
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        metadata: Optional[unittest.mock.Mock] = None,
    ) -> giatools.image.Image:
        """
        Create a test image with random data of the given data type.
        """
        np.random.seed(0)
        if np.issubdtype(dtype, np.integer):
            min_value = np.iinfo(dtype).min if min_value is None else min_value
            max_value = np.iinfo(dtype).max if max_value is None else max_value
            data = np.random.randint(min_value, max_value, shape, dtype=dtype)
        else:
            min_value = -1 if min_value is None else min_value
            max_value = +1 if max_value is None else max_value
            data = (
                np.random.rand(*shape) * (max_value - min_value) + min_value
            ).clip(min_value, max_value).astype(dtype)
            assert np.isinf(data).sum() == 0, f'min_value={min_value}, max_value={max_value}'  # sanity check
        return giatools.image.Image(
            data=data,
            axes=axes,
            original_axes=original_axes,
            metadata=metadata if metadata is not None else unittest.mock.Mock(),
        )

    def _test_conversion(
        self,
        src_dtype: np.dtype,
        dst_dtype: np.dtype,
        force_copy: bool,
        expected_dtype: Optional[np.dtype] = None,
    ):
        # The expected dtype is the same as the destination dtype if not specified
        if expected_dtype is None:
            expected_dtype = dst_dtype

        # Create test image
        img = self._create_image(src_dtype)
        original_dtype = img.data.dtype
        origianl_metadata = img.metadata
        original_axes = img.axes
        original_original_axes = img.original_axes

        # Determine if a `ValueError` is expected to be raised
        expect_value_error = False
        if (
            (  # conversion from signed to unsigned
                src_dtype in (np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64) and
                expected_dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
            ) or any(
                (  # conversion from larger unsigned to smaller signed
                    src_dtype == np.uint64 and expected_dtype in (np.int8, np.int16, np.int32, np.int64),
                    src_dtype == np.uint32 and expected_dtype in (np.int8, np.int16, np.int32),
                    src_dtype == np.uint16 and expected_dtype in (np.int8, np.int16),
                    src_dtype == np.uint8  and expected_dtype in (np.int8,),  # noqa: E272
                ),
            ) or any(
                (  # conversion from larger signed to smaller signed
                    src_dtype == np.int64 and expected_dtype in (np.int8, np.int16, np.int32),
                    src_dtype == np.int32 and expected_dtype in (np.int8, np.int16),
                    src_dtype == np.int16 and expected_dtype in (np.int8,),
                ),
            ) or any(
                (  # conversion from larger unsigned to smaller unsigned
                    src_dtype == np.uint64 and expected_dtype in (np.uint8, np.uint16, np.uint32),
                    src_dtype == np.uint32 and expected_dtype in (np.uint8, np.uint16),
                    src_dtype == np.uint16 and expected_dtype in (np.uint8,),
                ),
            ) or (  # conversion from larger integer to smaller float
                src_dtype not in (np.float16, np.float32, np.float64) and
                expected_dtype in (np.float16, np.float32, np.float64) and
                (
                    (img.data.max() > +np.finfo(expected_dtype).max).any() or
                    (img.data.min() < -np.finfo(expected_dtype).max).any()
                )
            )
        ):
            expect_value_error = True

            # Define a fallback image which *does not* raise an error under the tested conversion
            max_src_value = (
                np.finfo(src_dtype).max.item() if src_dtype in (
                    np.float16, np.float32, np.float64,
                ) else np.iinfo(src_dtype).max
            )
            max_dst_value = (
                np.finfo(expected_dtype).max.item() if expected_dtype in (
                    np.float16, np.float32, np.float64,
                ) else np.iinfo(expected_dtype).max
            )
            fallback_img = self._create_image(
                src_dtype,
                min_value=0,
                max_value=min((max_dst_value, max_src_value)),
                shape=img.data.shape,
                axes=img.axes,
                original_axes=img.original_axes,
                metadata=img.metadata,
            )

        # Define the conversion to be tested (and raise an error if unexpected overflows are encountered)
        def convert(_img):
            with np.errstate(invalid='raise', over='raise'):  # raise errors instead of printing warnings
                return _img.astype(dst_dtype, force_copy=force_copy)

        # Perform conversion (and verify that a `ValueError` is raised, if expected)
        if expect_value_error:
            with self.assertRaises(ValueError):
                img_converted = convert(img)
            img_converted = convert(fallback_img)
        else:
            img_converted = convert(img)

        # Verify properties of the converted image
        self.assertEqual(img.data.dtype, original_dtype)
        self.assertIs(img_converted.metadata, origianl_metadata)
        self.assertEqual(img_converted.axes, original_axes)
        self.assertEqual(img_converted.original_axes, original_original_axes)
        self.assertEqual(img_converted.data.dtype, expected_dtype)

        # Verify whether a copy was made or not
        if expected_dtype == original_dtype:
            if force_copy:
                self.assertIsNot(img.data, img_converted.data)
            else:
                self.assertIs(img.data, img_converted.data)

    def test__exact(self):
        for src_dtype in self.exact_dtype_list:
            for dst_dtype in self.exact_dtype_list:
                for force_copy in (False, True):
                    with self.subTest(f'from {src_dtype} to {dst_dtype} (force_copy={force_copy})'):
                        self._test_conversion(src_dtype, dst_dtype, force_copy=force_copy)

    def test__inexact(self):
        for src_dtype in self.exact_dtype_list:
            for dst_dtype in self.inexact_dtype_list:
                for force_copy in (False, True):
                    with self.subTest(f'from {src_dtype} to {dst_dtype} (force_copy={force_copy})'):

                        if dst_dtype == np.floating:
                            if src_dtype in (np.float16, np.float32, np.float64):
                                expected_dtype = src_dtype
                            else:
                                expected_dtype = np.float64

                        elif dst_dtype == np.integer:
                            if src_dtype in (np.int8, np.int16, np.int32, np.int64):
                                expected_dtype = src_dtype
                            elif src_dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                                expected_dtype = src_dtype
                            else:
                                expected_dtype = np.int64

                        elif dst_dtype == np.signedinteger:
                            if src_dtype in (np.int8, np.int16, np.int32, np.int64):
                                expected_dtype = src_dtype
                            else:
                                expected_dtype = np.int64

                        elif dst_dtype == np.unsignedinteger:
                            if src_dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                                expected_dtype = src_dtype
                            else:
                                expected_dtype = np.uint64

                        self._test_conversion(
                            src_dtype,
                            dst_dtype,
                            force_copy=force_copy,
                            expected_dtype=expected_dtype,
                        )
